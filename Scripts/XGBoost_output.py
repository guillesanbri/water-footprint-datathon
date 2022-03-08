import tqdm
import wandb
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from id_selection import get_all_ids
from holidays import add_holiday_column
from time_encoding import encode_hour, encode_month, encode_weekday


if __name__ == "__main__":
    # Hyperparameter config
    model = xgb.XGBRFRegressor  # XGBRegressor

    hyperparameters_default = {
        "freq": "day",

        "model": model,
        "booster": "gbtree",
        "grow_policy": "depthwise",
        "n_estimators": 25,
        "learning_rate": 0.11,
        "gamma": 16.52,
        "max_depth": 4,
        "min_child_weight": 0.21,
        "subsample": 0.35,
        "lambda": 0.01,
        "tree_method": "auto",

        "IDs_samples": sorted(get_all_ids("../Data/water_meters/")),
        "n_lags": 14,
        "monthly_encoding": "Sine",
        "weekday_encoding": "None",
        "hourly_encoding": "RBF",
        "holidays": False,
    }

    wandb.init(project="UniversityHack", config=hyperparameters_default, mode="disabled")
    config = wandb.config

    # Iterate over every ID
    assert len(config.IDs_samples) == 2747, "Not all IDs have been read"
    rows = []
    files_w = {}

    for id_ in config.IDs_samples:

        # Initialize a row dict
        row = {"ID": int(id_)}

        # Read data
        assert config.freq in ["day", "hour"]
        df = pd.read_csv(f"../Data/water_meters/{str(id_).zfill(4)}.csv")

        # Feature independent preprocessing
        df["SAMPLETIME"] = pd.to_datetime(df["SAMPLETIME"], utc=True)
        temp = df.copy()
        temp.index = temp.SAMPLETIME
        if config.freq == "day":
            temp = temp.groupby(pd.Grouper(freq='D')).sum()
        temp["TARGET"] = temp["DELTA"]

        if len(temp.index) != 365:
            print(f"W: ID: {id_} does not data corresponding to 365 days. Using data from {len(temp.index)} days.")
            files_w[id_] = 1

        if len(temp.index) > 20:

            if temp.index[-14] != pd.Timestamp("2020-01-18 00:00:00+00:00"):
                print(f"W: ID: {id_} does not have correct data in the lags. Using {temp.index[-14]} as first day.")
                files_w[id_] = 1

            prediction_df = pd.DataFrame(index=pd.date_range(start='2020-02-01', end='2020-02-14', freq="d"))
            prediction_df.index = pd.to_datetime(prediction_df.index, utc=True)
            temp = pd.concat([temp, prediction_df])

            # Add lagged deltas
            lagged_cols = []
            for i in range(config.n_lags):
                lag_name = f"delta_lag_{i + 1}"
                lagged_cols.append(lag_name)
                temp[lag_name] = temp["DELTA"].shift(i + 1)

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

            # Separate train and test
            X_train, X_test = X[:-14], X[-14:]
            y_train, y_test = y[:-14], y[-14:]

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
                if (p + 1) < n_iter:
                    for i in range(config.n_lags):
                        if i == 0:
                            X_test[p + 1][i] = predictions[p]  # Overwrite delta-1 in next time step
                        else:
                            X_test[p + 1][i] = X_test[p][i - 1]

        else:  # Number of data points below 20
            print("E")
            predictions = np.zeros(14)

        for i, d in enumerate([f"Dia_{i}" for i in range(1, 8)]):
            row[d] = predictions[i]
        row["Semana_1"] = np.sum(predictions[:7])
        row["Semana_2"] = np.sum(predictions[7:])
        rows.append(row)

    pd.DataFrame(rows).to_csv("debug.txt", index=False, header=False, decimal=".", sep="|", float_format="%.2f")
    print("Prediction csv file exported correctly!")

    print()
    print("="*80)
    print(" "*34 + "Output report" + " "*33)
    print("-" * 80)
    print(f"Found warnings in {len(files_w.keys())} files. ({len(files_w.keys())/2747*100:.2f} %)")
    print("="*80)
