import numpy as np


def check_valid_y(*args):
    """
    Checks if any given parameter is an array with 14 values.
    :param args: Arrays of values.
    :return: True if all arguments are arrays of len==14, asserts otherwise.
    """
    for y in args:
        assert len(y) == 14, f"Y array with len == {len(y)}"
    return True


def rmse(y_pred, y_true):
    """
    Calculates the RMSE of two fiven values.
    :param y_pred: Predicted value.
    :param y_true: Target value.
    :return: Root Mean Squared Error.
    """
    return np.sqrt(np.mean((y_pred-y_true)**2))


def compute_metrics(y_pred, y_true):
    """
    Computes a dict of different metrics related to the Datathon.
    :param y_pred: Array with 14 values predicted by an algorithm.
    :param y_true: Array with 14 ground truth values.
    :return: Dict with keys 'RMSE_d<ii>' (01 <= ii <= 14); 'RMSE_w<e>' (1 <= e <= 2); and RMSE_Total.
     RMSE_Total is calulated as: 0.5 * daily_RMSE_mean + 0.5 * weekly_RMSE_mean.
    """
    n_days = len(y_true)  # should be 14
    n_weeks = n_days // 7
    check_valid_y(y_pred, y_true)
    metrics_dict = {}
    RMSEd_values = []
    RMSEw_values = []
    for i in range(n_days):
        RMSE_di = rmse(y_pred[i], y_true[i])
        metrics_dict[f"RMSE_d{str(i+1).zfill(2)}"] = RMSE_di
        if i < 7:
            RMSEd_values.append((RMSE_di))
    for e in range(n_weeks):
        week_pred = np.sum(y_pred[e * 7:(e + 1) * 7])
        week_true = np.sum(y_true[e * 7:(e + 1) * 7])
        RMSE_we = rmse(week_pred, week_true)
        metrics_dict[f"RMSE_w{str(e+1).zfill(2)}"] = RMSE_we
        RMSEw_values.append(RMSE_we)
    RMSE_total = 0.5*np.mean(RMSEd_values) + 0.5*np.mean(RMSEw_values)
    metrics_dict["RMSE_Total"] = RMSE_total
    return metrics_dict


if __name__ == "__main__":
    y_gt = [i+2 for i in range(14)]
    y_o = [i for i in range(14)]
    compute_metrics(y_o, y_gt)