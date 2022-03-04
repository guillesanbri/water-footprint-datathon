import pandas as pd
import numpy as np
import wandb
from sklego.preprocessing import RepeatingBasisFunction
from cross_validation import TimeCrossValidator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from metrics import compute_metrics

if __name__ == "__main__":
    # Hyperparameter config
    model = GradientBoostingRegressor
    hyperparameters_default = {
        "model": model,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 3,
        "freq": "hour",
        "ID_sample": 100,
        "Time encoding": "RBF",
        "weekday_rbf": True,
        "monthly_rbf": True,
        "hourly_rbf": True,
        "Holidays": False,
    }

    # Init wandb
    wandb.init(project="UniversityHack", config=hyperparameters_default)  # mode="disabled"
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
    monthly_rbf_cols = []
    if config.monthly_rbf:
        temp["day"] = temp.index.dayofyear
        monthly_rbf = RepeatingBasisFunction(n_periods=12, column="day", input_range=(1, 365))
        monthly_rbf.fit(temp)
        monthly_rbf_results = monthly_rbf.transform(temp)
        for i, col in enumerate(monthly_rbf_results.T):
            col_name = f"month_rbf_{i+1}"
            temp[col_name] = col
            monthly_rbf_cols.append(col_name)

    weekday_rbf_cols = []
    if config.weekday_rbf:
        temp["weekday"] = temp.index.weekday
        weekday_rbf = RepeatingBasisFunction(n_periods=7, column="weekday", input_range=(0, 6))
        weekday_rbf.fit(temp)
        weekday_rbf_results = weekday_rbf.transform(temp)
        for i, col in enumerate(weekday_rbf_results.T):
            col_name = f"weekday_rbf_{i + 1}"
            temp[col_name] = col
            weekday_rbf_cols.append(col_name)

    hour_rbf_cols = []
    if config.freq == "hour" and config.hourly_rbf:
        temp["hour"] = temp.index.hour
        hour_rbf = RepeatingBasisFunction(n_periods=24, column="hour", input_range=(0, 23))
        hour_rbf.fit(temp)
        hour_rbf_results = hour_rbf.transform(temp)
        for i, col in enumerate(hour_rbf_results.T):
            col_name = f"hour_rbf_{i + 1}"
            temp[col_name] = col
            hour_rbf_cols.append(col_name)

    # Extract target and input columns.
    # Exclude the row with NaN values (due to shift operation)
    X = temp[["YESTERDAY_DELTA", *monthly_rbf_cols, *weekday_rbf_cols, *hour_rbf_cols]].to_numpy()[1:]
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
