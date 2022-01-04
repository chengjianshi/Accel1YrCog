import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Union
from pathlib import Path

def hip_preprocess(path: Union[str, Path],  epoch: int,  sample_frequency: int) -> np.array:
    """
    Description:
        -> produce formatted output for each participants raw accelerameters data
    Parameters:
        @path: raw acc path
        @epoch: unit size of window
        @sample_frequency: device based
    Return:
        matrix with shape: (number_of_days, number_of_epoch_per_day, data_points_per_epoch)
    """

    if (epoch < 1):
        raise Exception("Sorry, no epoch below 1 second")
    
    f = open(path, "r").readlines()[:10]
    
    if (len(f) == 0):
       raise Exception(f"empty file: {str(path)}") 
    
    f = list(map(lambda x: x.strip(), f))
    stime = f[2].split(" ")[-1]

    df = pd.read_csv(path)
    df = df[len(df) % (epoch * sample_frequency):]

    df["vm"] = np.sqrt(np.sum(df**2, axis=1))
    df = df[["vm"]]

    # (number of epochs, epoch length)
    X = df.to_numpy().reshape(-1, epoch * sample_frequency)

    unit = timedelta(seconds=epoch)
    total_time = X.shape[0] * unit
    oneday = timedelta(days=1)
    lagged_time = str2tdelta(stime)

    if (total_time < oneday):
        raise Exception(f"Toatl time {total_time} less than a day.")

    fr = (oneday - lagged_time)
    to = (total_time - (total_time + lagged_time) % oneday)

    if (to - fr >= oneday):
        fr = fr // unit
        to = to // unit
        X = X[fr:to]
    else:
        X1 = X[fr // unit: (fr + lagged_time) // unit]
        X2 = X[:fr // unit]
        X = np.vstack((X1, X2))

    total_time = X.shape[0] * unit

    X = X.reshape(total_time // oneday, oneday // unit,
                  int(unit.total_seconds() * sample_frequency))

    return X


def str2tdelta(time):
    """
    input time in format: xx:xx:xx
    """

    time = [int(t) for t in time.split(":")]
    return timedelta(hours=time[0],
                     minutes=time[1],
                     seconds=time[2])

class accelTooShort(Exception):
    def __init__(self, id_, ndays, msg="accel effective wear time less than 1 day"):
        self.id_ = id_
        self.msg = msg
        self.ndays = ndays
        super().__init__(msg)

    def __str__(self):
        print(f"{self.id_}: {self.msg} -> {self.ndays}")


def test_effective_weartime(df, th):
    dt = pd.to_datetime(df.datetime).to_list()
    duration = dt[-1] - dt[0]
    wear_time_percent = sum(df.off_wrist_status) / len(df)
    if (duration * wear_time_percent < th):
        return False
    return True


def truncate_ineffective_date(df):
    dt = pd.to_datetime(df.datetime).to_list()
    df["date"] = list(map(lambda x: x.date(), dt))
    df_groupby_date = df.groupby("date", axis=0)
    date_to_use = []
    for date, dfg in df_groupby_date:
        if (test_effective_weartime(dfg, pd.Timedelta("23 hour"))):
            date_to_use.append(date)

    df = df[list(map(lambda x: x.date() in date_to_use, dt))]
    return df.groupby("date", axis=0)


def wrist_preprocess(path: Union[Path, str], epoch:str, sample_frequency:str)->np.array:
    """
    Description:
        -> produce formatted output for each participants raw accelerameters data
    Parameters:
        @path: raw acc path
        @epoch: unit size of window
        @sample_frequency: device based
    Return:
        matrix with shape: (number_of_days, number_of_epoch_per_day, data_points_per_epoch)
    """

    _id = path.split(".")[0]

    source = pd.read_csv(path)
    source = truncate_ineffective_date(source)
    ndays = len(source)

    if (ndays < 1):
        raise accelTooShort(_id, ndays)

    num_epoch = pd.Timedelta("1 day") // pd.Timedelta(epoch)
    epoch_size = pd.Timedelta(epoch) // pd.Timedelta(sample_frequency)

    res = np.zeros((ndays, num_epoch, epoch_size))

    for ix, (_, dfg) in enumerate(source):
        accel = dfg.activity.to_numpy()
        try:
            accel = accel.reshape((num_epoch, epoch_size))
            res[ix] = accel
        except:
            buffer = np.zeros(num_epoch * epoch_size)
            buffer[:accel.shape[0]] = accel
            res[ix] = buffer.reshape((num_epoch, epoch_size))

    return res

