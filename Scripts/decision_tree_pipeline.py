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
    hyperparameters_default = {
        "model": "Gradient boosting",
        "freq": "day",
        "ID_sample": 10,
        "Time encoding": "RBF",
        "Holidays": False,
    }

    # Init wandb
    wandb.init(project="UniversityHack", config=hyperparameters_default,)  # mode="disabled"
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
    print(temp.columns)
    # Time encoding
    temp["day"] = temp.index.dayofyear
    monthly_rbf = RepeatingBasisFunction(n_periods=12, column="day", input_range=(1, 365))
    monthly_rbf.fit(temp)
    monthly_rbf_results = monthly_rbf.transform(temp)
    monthly_rbf_cols = []
    for i, col in enumerate(monthly_rbf_results.T):
        col_name = f"month_rbf_{i+1}"
        temp[col_name] = col
        monthly_rbf_cols.append(col_name)
    temp["weekday"] = temp.index.weekday
    weekday_rbf = RepeatingBasisFunction(n_periods=7, column="weekday", input_range=(0, 6))
    weekday_rbf.fit(temp)
    weekday_rbf_results = weekday_rbf.transform(temp)
    weekday_rbf_cols = []
    for i, col in enumerate(weekday_rbf_results.T):
        col_name = f"weekday_rbf_{i + 1}"
        temp[col_name] = col
        weekday_rbf_cols.append(col_name)

    # Extract target and input columns.
    # Exclude the row with NaN values (due to shift operation)
    X = temp[["YESTERDAY_DELTA", *monthly_rbf_cols, *weekday_rbf_cols]].to_numpy()[1:]
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
        regressor = GradientBoostingRegressor(random_state=0)
        regressor.fit(X_train, y_train)
        # Prediction
        predictions = np.zeros_like(y_test)
        for p in range(14):
            predictions[p] = regressor.predict(X_test[p].reshape(1, -1))
            if (p+1) < 14:
                X_test[p+1][0] = predictions[p]  # Overwrite YESTERDAY_DELTA in next time step
        # Print RMSE of 14 days predictions
        rmse_metrics = compute_metrics(predictions, y_test)
        wandb.log({"cv_split": split+1, **rmse_metrics})
        print(rmse_metrics)
