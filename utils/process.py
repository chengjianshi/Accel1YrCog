from typing import Union, Dict
import numpy as np
from utils.reader import reader, accelType
from utils.feature_toolkit import *
import pandas as pd
from pathlib import Path

def get_level(x: float, a: Union[np.ndarray, pd.Series]):
    """
    determine the activity level of single participant
    :param x: single input activity value
    :param a: population-wise activity values
    :return: activity level
    """

    th1 = np.quantile(a, 0.25)
    th2 = np.quantile(a, 0.50)
    th3 = np.quantile(a, 0.75)

    if (x < th1):
        return 0
    elif (x < th2):
        return 1
    elif (x < th3):
        return 2
    else:
        return 3

def activity_level_df(df: pd.DataFrame):
    """
    generate activity level feature data
    :param df: input dataframe
    :return: data frame with activity level as feature included
    """
    cpm = df.CPM_75.to_numpy()
    vmc = df.VMC_75.to_numpy()

    df["cpm_level"] = [get_level(x, cpm) for x in cpm]
    df["vmc_level"] = [get_level(x, vmc) for x in vmc]
    return df

def process(clinical_df: pd.DataFrame, path: Union[str, Path], type: str)->Dict[str, float]: 

    """
    Description: 
        -> process raw accelerometer data and concate with clinical data 
    Parameters:
        @clinical_df: clinical data frame with index as key to match with current raw accel path 
        @path: path to raw accel data 
    Return:
        -> Diction of features with feature name as key and value of float 
    """
    
    f = open(path, "r")
    if (len(f.readlines()) < 100):
        raise Exception(f"Empty Accelerometer Data: {path}")
    
    if type == "hip":    
        read = reader(path, epoch = 60, type = accelType.HIP, sample_frequency = 30)
        ID = path.split("/")[-1]
    elif type == "wrist":
        read = reader(path,epoch = "1 min",type = accelType.WRIST, sample_frequency="15 second")
        ID = path.split("/")[-1].split(".")[0]
     
    if (ID not in clinical_df.index):
        raise Exception(f"Clinical Info non-exist: {ID}")

    FEATURES = clinical_df.loc[ID].to_dict()

    cpm, cpm_stat = CPM(read.signal)
    vmc, vmc_stat = VMC(read.signal)

    keys = ["mu", "std", "25", "50", "75", "range",
            "skew", "kurt", "beta(a)", "beta(b)", "entropy"]

    for i in range(len(vmc_stat)):

        FEATURES["CPM_" + keys[i]] = cpm_stat[i]
        FEATURES["VMC_" + keys[i]] = vmc_stat[i]

    cpm_paee = PAEE(cpm, 60)
    vmc_paee = PAEE(vmc, 60)

    FEATURES["CPM_PAEE_1"] = cpm_paee[0]
    FEATURES["CPM_PAEE_2"] = cpm_paee[1]
    FEATURES["VMC_PAEE_1"] = vmc_paee[0]
    FEATURES["VMC_PAEE_2"] = vmc_paee[1]

    cpm_fft = FFTstats(cpm, 60)
    vmc_fft = FFTstats(vmc, 60)

    keys = ["top15fft", "top15freq", "fentropy", "psd_mu", "psd_std",
            "rms_amplitude", "kurt", "skew", "mean_freq", "median_freq"]

    for i in range(len(cpm_fft)):
        if keys[i] == "top15freq" or keys[i] == "top15fft":
            for j in range(len(cpm_fft[i])):
                FEATURES["CPM_" + keys[i] + str(j)] = cpm_fft[i][j]
                FEATURES["VMC_" + keys[i] + str(j)] = vmc_fft[i][j]
        else:
            FEATURES["CPM_" + keys[i]] = cpm_fft[i]
            FEATURES["VMC_" + keys[i]] = vmc_fft[i]

    RES = activity_level_df(FEATURES)

    return RES

