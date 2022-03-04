import pandas as pd
import numpy as np
import wandb
from sklego.preprocessing import RepeatingBasisFunction
from cross_validation import TimeCrossValidator
import xgboost as xgb
from metrics import compute_metrics


def encode_rbf(df, column, n_periods, input_range, col_tag):
    rbf_cols = []
    rbf = RepeatingBasisFunction(n_periods=n_periods, column=column, input_range=input_range)
    rbf.fit(temp)
    rbf_results = rbf.transform(temp)
    for i, col in enumerate(rbf_results.T):
        col_name = f"{col_tag}_rbf_{i+1}"
        temp[col_name] = col
        rbf_cols.append(col_name)
    return df, rbf_cols


# Offset should not affect tree based models, but can become a hyperparameter in nn based models.
def encode_sine(df, column, n_periods, input_range, col_tag):
    sine_cols = []
    if n_periods == (input_range[1] - input_range[0] + 1):
        period = n_periods
    else:
        period = (input_range[1] - input_range[0] + 1) / n_periods  # Hacky solution to monthly encoding with day data
    angle_rad = df[column] / period * 2 * np.pi
    # Sine
    sin_col_name = f"{col_tag}_sin"
    df[sin_col_name] = np.sin(angle_rad)
    sine_cols.append(sin_col_name)
    # Cosine
    cos_col_name = f"{col_tag}_sin"
    df[cos_col_name] = np.cos(angle_rad)
    sine_cols.append(cos_col_name)
    return df, sine_cols


def encode_time(df, encoding_type, column, n_periods, input_range, col_tag):
    assert encoding_type in ["RBF", "Sine", "None"]
    if encoding_type == "RBF":
        return encode_rbf(df, column, n_periods, input_range, col_tag)
    elif encoding_type == "Sine":
        return encode_sine(df, column, n_periods, input_range, col_tag)
    elif encoding_type == "None":
        return df, []


def encode_month(df, encoding_type):
    df["day"] = df.index.dayofyear
    return encode_time(df, encoding_type, column="day", n_periods=12, input_range=(1, 365), col_tag="month")


def encode_weekday(df, encoding_type):
    df["weekday"] = df.index.weekday
    return encode_time(df, encoding_type, column="weekday", n_periods=7, input_range=(0, 6), col_tag="weekday")


def encode_hour(df, encoding_type):
    df["hour"] = df.index.hour
    return encode_time(df, encoding_type, column="hour", n_periods=24, input_range=(0, 23), col_tag="hour")


if __name__ == "__main__":
    # Hyperparameter config
    model = xgb.XGBRegressor
    # https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
    hyperparameters_default = {
        "freq": "hour",

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
        "monthly_encoding": "RBF",
        "weekday_encoding": "RBF",
        "hourly_encoding": "RBF",
        "Holidays": False,
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
    temp["YESTERDAY_DELTA"] = temp["DELTA"].shift(1)

    # Time encoding
    # # Month encoding
    temp, month_encoding_cols = encode_month(temp, config.monthly_encoding)
    # # Weekday encoding
    temp, weekday_encoding_cols = encode_weekday(temp, config.weekday_encoding)
    # # Hour encoding
    temp, hour_encoding_cols = encode_hour(temp, config.hourly_encoding) if config.freq == "hour" else (temp, [])

    # Extract target and input columns.
    # Exclude the row with NaN values (due to shift operation)
    X = temp[["YESTERDAY_DELTA", *month_encoding_cols, *weekday_encoding_cols, *hour_encoding_cols]].to_numpy()[1:]
    y = temp["TARGET"].to_numpy()[1:]

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Time series cross validation
    # 13 splits of two weeks: first training uses the first 6 months of data.
    # Last split trains with all but the two last weeks of data.
    tscv = TimeCrossValidator(data_freq=config.freq, n_splits=13, test_days=14)
    for split, (train_index, test_index) in enumerate(tscv.split(X)):
        cv_split_title = f"  CV Split no. {split + 1}  "
        print("="*len(cv_split_title))
        print(cv_split_title)
        print("="*len(cv_split_title))
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
                X_test[p+1][0] = predictions[p]  # Overwrite YESTERDAY_DELTA in next time step

        # Print RMSE of 14 days predictions
        if config.freq == "hour":
            y_test = np.sum(y_test.reshape(-1, 24), axis=1)
            predictions = np.sum(predictions.reshape(-1, 24), axis=1)

        rmse_metrics = compute_metrics(predictions, y_test)
        wandb.log({"cv_split": split+1, **rmse_metrics})
        print(rmse_metrics)
