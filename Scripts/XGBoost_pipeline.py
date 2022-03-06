import pandas as pd
import numpy as np
import tqdm
import wandb
from cross_validation import TimeCrossValidator
import xgboost as xgb
from metrics import compute_metrics
from collections import Counter

from time_encoding import encode_hour, encode_month, encode_weekday
from holidays import add_holiday_column
from id_selection import validation_ids_v1

if __name__ == "__main__":
    # Hyperparameter config
    model = xgb.XGBRegressor
    # https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
    hyperparameters_default = {
        "freq": "day",

        "model": model,
        "booster": "gbtree",
        "grow_policy": 0,  # 0: grow depth-wise, 1: favours splitting at nodes with the highest loss change
        "n_estimators": 25,
        "learning_rate": 0.11,
        "gamma": 16.52,
        "max_depth": 4,
        "min_child_weight": 0.21,
        "subsample": 0.35,
        "lambda": 0.01,
        "tree_method": "auto",

        "IDs_samples": validation_ids_v1,
        "n_lags": 14,
        "monthly_encoding": "Sine",
        "weekday_encoding": "None",
        "hourly_encoding": "RBF",
        "holidays": False,
    }

    # Init wandb
    wandb.init(project="UniversityHack", config=hyperparameters_default, mode="disabled")  # mode="disabled"
    config = wandb.config

    # Iterate over multiple IDs
    mean_IDs = Counter()
    for id_ in tqdm.tqdm(config.IDs_samples):

        # Read data
        assert config.freq in ["day", "hour"]
        df = pd.read_csv(f"../Data/water_meters/{str(id_).zfill(4)}.csv")
        # print(f"Data from meter no. {id_} loaded correctly.")

        # Feature independent preprocessing
        df["SAMPLETIME"] = pd.to_datetime(df["SAMPLETIME"], utc=True)
        temp = df.copy()
        temp.index = temp.SAMPLETIME
        if config.freq == "day":
            temp = temp.groupby(pd.Grouper(freq='D')).sum()
        temp["TARGET"] = temp["DELTA"]

        # Add lagged deltas
        lagged_cols = []
        for i in range(config.n_lags):
            lag_name = f"delta_lag_{i+1}"
            lagged_cols.append(lag_name)
            temp[lag_name] = temp["DELTA"].shift(i+1)

        # Add holiday data
        additional_cols = []
        if config.holidays:
            col_name = "holiday"
            additional_cols.append(col_name)
            temp = add_holiday_column(temp, col_name)

        # Time encoding
        # # Month encoding
        temp, month_encoding_cols = encode_month(temp, config.monthly_encoding)
        # # Weekday encoding
        temp, weekday_encoding_cols = encode_weekday(temp, config.weekday_encoding)
        # # Hour encoding
        temp, hour_encoding_cols = encode_hour(temp, config.hourly_encoding) if config.freq == "hour" else (temp, [])

        # Extract target and input columns.
        # Exclude rows with NaN values (due to shift operation)
        X = temp[[*lagged_cols,
                  *additional_cols,
                  *month_encoding_cols, *weekday_encoding_cols, *hour_encoding_cols]].to_numpy()[config.n_lags:]
        y = temp["TARGET"].to_numpy()[config.n_lags:]

        # print(f"X shape: {X.shape}, y shape: {y.shape}")

        # Time series cross validation
        # 13 splits of two weeks: first training uses the first 6 months of data.
        # Last split trains with all but the two last weeks of data.
        n_splits = 13
        tscv = TimeCrossValidator(data_freq=config.freq, n_splits=n_splits, test_days=14)
        mean_metrics = Counter()
        for split, (train_index, test_index) in enumerate(tscv.split(X)):
            # cv_split_title = f"  CV Split no. {split + 1}  "
            # print("="*len(cv_split_title))
            # print(cv_split_title)
            # print("="*len(cv_split_title))

            # Separate train and test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Fittable feature transformations
            pass
            # Training
            regressor = model(random_state=0, nthread=2,
                              learning_rate=config.learning_rate, n_estimators=config.n_estimators, max_depth=config.max_depth)
            regressor.fit(X_train, y_train)
            # Prediction
            predictions = np.zeros_like(y_test)
            n_iter = len(y_test)  # 14 if config.freq == "day" else 14*24
            for p in range(n_iter):
                predictions[p] = regressor.predict(X_test[p].reshape(1, -1))
                if (p+1) < n_iter:
                    for i in range(config.n_lags):
                        if i == 0:
                            X_test[p+1][i] = predictions[p]  # Overwrite delta-1 in next time step
                        else:
                            X_test[p+1][i] = X_test[p][i-1]

            # Print RMSE of 14 days predictions
            if config.freq == "hour":
                y_test = np.sum(y_test.reshape(-1, 24), axis=1)
                predictions = np.sum(predictions.reshape(-1, 24), axis=1)

            rmse_metrics = compute_metrics(predictions, y_test)
            # wandb.log({"cv_split": split+1, **rmse_metrics})
            mean_metrics.update(rmse_metrics)
        mean_metrics = {k: v/n_splits for k, v in mean_metrics.items()}
        # print(f"Mean of metrics along {n_splits} splits:")
        # print(mean_metrics)
        wandb.log({"ID": id_, **mean_metrics})
        mean_IDs.update(mean_metrics)
    mean_IDs = {f"mean_IDs_{k}": v / len(config.IDs_samples) for k, v in mean_IDs.items()}
    print(f"Mean of metrics along {len(config.IDs_samples)} IDs:")
    print(mean_IDs)
    wandb.log({**mean_IDs})
