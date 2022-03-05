import pandas as pd
import numpy as np
import wandb
import tqdm
from cross_validation import TimeCrossValidator
from metrics import compute_metrics
from collections import Counter

if __name__ == "__main__":
    # Hyperparameter config
    hyperparameters_default = {
        "model": "Yesterday delta naive baseline",
        "freq": "day",
        "ID_sample": 100,
    }

    # Init wandb
    wandb.init(project="UniversityHack", config=hyperparameters_default, mode="disabled")
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
    temp["YESTERDAY_READING"] = temp["READING"].shift(1)

    # Extract target and columns.
    # Exclude the row with NaN values (due to shift operation)
    X = temp[["YESTERDAY_READING", "YESTERDAY_DELTA"]].to_numpy()[1:]
    y = temp["TARGET"].to_numpy()[1:]

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Time series cross validation
    # 13 splits of two weeks: first training uses the first 6 months of data.
    # Last split trains with all but the two last weeks of data.
    n_splits = 13
    tscv = TimeCrossValidator(data_freq=config.freq, n_splits=n_splits, test_days=14)
    mean_metrics = Counter()
    for split, (train_index, test_index) in tqdm.tqdm(enumerate(tscv.split(X)), total=n_splits):
        # Separate train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fittable feature transformations
        pass
        # Training
        pass  # Naive model
        # Prediction
        predictions = np.zeros_like(y_test)
        for p in range(14):
            predictions[p] = X_test[p][1]  # Naive baseline: Copy YESTERDAY_DELTA
            if (p+1) < 14:
                X_test[p+1][1] = predictions[p]  # Overwrite YESTERDAY_DELTA in next time step
        # Print RMSE of 14 days predictions
        rmse_metrics = compute_metrics(predictions, y_test)
        wandb.log({"cv_split": split+1, **rmse_metrics})
        mean_metrics.update(rmse_metrics)
    mean_metrics = {k: v / n_splits for k, v in mean_metrics.items()}
    print(f"Mean of metrics along {n_splits} splits:")
    print(mean_metrics)
    wandb.log({"cv_split": "mean", **mean_metrics})
