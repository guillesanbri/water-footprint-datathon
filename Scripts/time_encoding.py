import numpy as np
from sklego.preprocessing import RepeatingBasisFunction


def encode_rbf(df, column, n_periods, input_range, col_tag):
    rbf_cols = []
    rbf = RepeatingBasisFunction(n_periods=n_periods, column=column, input_range=input_range)
    rbf.fit(df)
    rbf_results = rbf.transform(df)
    for i, col in enumerate(rbf_results.T):
        col_name = f"{col_tag}_rbf_{i+1}"
        df[col_name] = col
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
    cos_col_name = f"{col_tag}_cos"
    df[cos_col_name] = np.cos(angle_rad)
    sine_cols.append(cos_col_name)
    return df, sine_cols


def encode_time(df, encoding_type, column, n_periods, input_range, col_tag):
    assert encoding_type in ["RBF", "Sine", "Ordinal", "None"]
    if encoding_type == "RBF":
        return encode_rbf(df, column, n_periods, input_range, col_tag)
    elif encoding_type == "Sine":
        return encode_sine(df, column, n_periods, input_range, col_tag)
    elif encoding_type == "Ordinal":
        return df, [column]
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
