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
        "n_estimators": 100,
        "learning_rate": 0.3,
        "gamma": 0,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1,
        "lambda": 1,
        "tree_method": "auto",

        "ID_sample": 100,  # TODO
        "n_lags": 1,
        "monthly_encoding": "RBF",
        "weekday_encoding": "RBF",
        "hourly_encoding": "RBF",
        "holidays": False,
    }

    # Init wandb
    wandb.init(project="UniversityHack", config=hyperparameters_default, mode="disabled")  # mode="disabled"
    config = wandb.config

    # Read data
    assert config.freq in ["day", "hour"]
    df = pd.read_csv(f"../Data/water_meters/{str(config.ID_sample).zfill(4)}.csv")
    print(f"Data from meter no. {config.ID_sample} loaded correctly.")

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

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Time series cross validation
    # 13 splits of two weeks: first training uses the first 6 months of data.
    # Last split trains with all but the two last weeks of data.
    n_splits = 13
    tscv = TimeCrossValidator(data_freq=config.freq, n_splits=n_splits, test_days=14)
    mean_metrics = Counter()
    for split, (train_index, test_index) in tqdm.tqdm(enumerate(tscv.split(X)), total=n_splits):
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
        regressor = model(random_state=0, learning_rate=config.learning_rate, n_estimators=config.n_estimators, max_depth=config.max_depth)
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
        wandb.log({"cv_split": split+1, **rmse_metrics})
        mean_metrics.update(rmse_metrics)
    mean_metrics = {k: v/n_splits for k, v in mean_metrics.items()}
    print(f"Mean of metrics along {n_splits} splits:")
    print(mean_metrics)
    wandb.log({"cv_split": "mean", **mean_metrics})
