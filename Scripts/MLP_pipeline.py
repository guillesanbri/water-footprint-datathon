import pandas as pd
import numpy as np
import tqdm
import wandb
from cross_validation import TimeCrossValidator
from metrics import compute_metrics
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch import nn
from torch.nn import MSELoss
from torch.optim import SGD
import torch

from time_encoding import encode_hour, encode_month, encode_weekday
from holidays import add_holiday_column
from id_selection import validation_ids_v1


class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_layer_sizes=[], output_size=1):
        super().__init__()

        hidden_layers_list = [nn.Linear(input_dim, hidden_layer_sizes[0])]
        hidden_layers_list.append(nn.ReLU())
        for i in range(len(hidden_layer_sizes) - 1):
            hidden_layers_list.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            # hidden_layers_list.append(nn.ReLU())
        hidden_layers_list.append(nn.Linear(hidden_layer_sizes[-1], output_size))

        self.hidden_layers = nn.ModuleList(hidden_layers_list)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Hyperparameter config
    model = MLP
    loss_fn = {"MSE": MSELoss}
    optimizers = {"SGD": SGD}

    hyperparameters_default = {
        "device": "cuda",

        "freq": "hour",

        "model": model,
        "optimizer": "SGD",
        "criterion": "MSE",  # TODO: Test RMSE
        "learning_rate": 0.00002,
        "momentum": 0.9,

        "IDs_samples": validation_ids_v1,
        "n_lags": 0,
        "monthly_encoding": "RBF",
        "weekday_encoding": "Sine",
        "hourly_encoding": "Sine",
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
            # Separate train and test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Fittable feature transformations
            if config.n_lags > 0:
                rs = MinMaxScaler()
                X_train[:, :len(lagged_cols)] = rs.fit_transform(X_train[:, :len(lagged_cols)])
                X_test[:, :len(lagged_cols)] = rs.transform(X_test[:, :len(lagged_cols)])
            # Create tensors
            X_train = torch.Tensor(X_train).to(config.device)
            X_test = torch.Tensor(X_test).to(config.device)
            y_train = torch.Tensor(y_train).unsqueeze(-1).to(config.device)
            y_test = torch.Tensor(y_test).unsqueeze(-1).to(config.device)
            print(y_test.shape)
            # Training
            regressor = model(input_dim=len(X_train[0]), hidden_layer_sizes=[16, 16], output_size=1).to(config.device)  # 64
            criterion = loss_fn[config.criterion]()
            optimizer = optimizers[config.optimizer](regressor.parameters(), lr=config.learning_rate, momentum=config.momentum)
            # # Train loop
            for epoch in range(200000):  # loop over the dataset multiple times

                running_loss = 0.0

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = X_train, y_train

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = regressor(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if epoch % 1000 == 0:
                    print(f'[{epoch}] train loss: {running_loss:.3f}')

                    with torch.no_grad():
                        predictions = torch.zeros_like(y_test).to(config.device)
                        n_iter = len(y_test)  # 14 if config.freq == "day" else 14*24
                        for p in range(n_iter):
                            predictions[p] = regressor(X_test[p])
                            if (p + 1) < n_iter:
                                for i in range(config.n_lags):
                                    if i == 0:
                                        X_test[p + 1][i] = predictions[p]  # Overwrite delta-1 in next time step
                                    else:
                                        X_test[p + 1][i] = X_test[p][i - 1]
                        # Print RMSE of 14 days predictions
                        if config.freq == "hour":
                            y_test_temp = torch.sum(y_test.reshape(-1, 24), axis=1)
                            predictions = torch.sum(predictions.reshape(-1, 24), axis=1)

                        rmse_metrics = compute_metrics(predictions.cpu().detach().numpy(),
                                                       y_test_temp.cpu().detach().numpy())
                        print(rmse_metrics)

                running_loss = 0.0

            print('Finished Training')
            # Prediction
            predictions = torch.zeros_like(y_test)
            print(y_test, predictions)
            n_iter = len(y_test)  # 14 if config.freq == "day" else 14*24
            for p in range(n_iter):
                predictions[p] = regressor(X_test[p].reshape(1, -1))
                if (p+1) < n_iter:
                    for i in range(config.n_lags):
                        if i == 0:
                            X_test[p+1][i] = predictions[p]  # Overwrite delta-1 in next time step
                        else:
                            X_test[p+1][i] = X_test[p][i-1]
            print(predictions)
            print(y_test)
            # Print RMSE of 14 days predictions
            if config.freq == "hour":
                y_test = np.sum(y_test.reshape(-1, 24), axis=1)
                predictions = np.sum(predictions.reshape(-1, 24), axis=1)

            rmse_metrics = compute_metrics(predictions.cpu().detach().numpy(), y_test.cpu().detach().numpy())
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
