import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class TimeCrossValidator(TimeSeriesSplit):
    def __init__(self, data_freq="day", n_splits=13, test_days=14, **kwargs):
        assert data_freq in ["day", "hour"], 'freq must be "day" or "hour".'
        samples_per_day = 24 if data_freq == "hour" else 1
        super().__init__(n_splits=n_splits, test_size=test_days*samples_per_day, **kwargs)


if __name__ == "__main__":
    X = np.random.randn(365*24, 5)
    y = np.random.randn(365*24, 1)

    print(X.shape, y.shape)

    tscv = TimeCrossValidator(data_freq="hour", n_splits=13, test_days=14)
    for train_index, test_index in tscv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
