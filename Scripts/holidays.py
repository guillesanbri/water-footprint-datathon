import pandas as pd

holiday_dates = ["2019-01-01", "2019-01-22",
                 "2019-03-19",
                 "2019-04-19", "2019-04-22", "2019-04-29",
                 "2019-05-01",
                 "2019-06-24",
                 "2019-08-15",
                 "2019-10-09", "2019-10-12",
                 "2019-11-01",
                 "2019-12-06", "2019-12-25",
                 "2020-01-01", "2020-01-06", "2020-01-22"]


def add_holiday_column(df, col_name):
    holiday_df = pd.DataFrame(index=pd.date_range(start='2019-01-01', end='2020-03-01', freq="d"))
    holiday_df.index = pd.to_datetime(holiday_df.index, utc=True)
    holiday_df[col_name] = 0
    for d in holiday_dates:
        holiday_df.loc[d] = 1
    return pd.merge_asof(df, holiday_df, left_index=True, right_index=True,)