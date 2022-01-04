import os
import pandas as pd
import numpy as np
from typing import Union
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split  
import xgboost as xgb
from pathlib import Path

np.random.seed(1234)
print(xgb.__version__)
SAVE_PATH = Path(os.getcwd()) / "models"
if not SAVE_PATH.exist():
    SAVE_PATH.mkdir(parents = True)

def CDPred(features_path: Union[str, Path], type: str, model_name: str):
    """
    generate clinical feature only classifier CDPred

    :param features_path: clinical data path
    :param type: hip / wrist data
    :param model_name: saved model name
    :return: xgb classifier
    """
    df = pd.read_csv(features_path)

    data = pd.get_dummies(df)
    data = data.loc[:, (data != 0).any(axis=0)]

    Xf = data.loc[:, data.columns != "Y"]
    Yf = data.loc[:, "Y"]
    xf = Xf.to_numpy()
    yf = Yf.to_numpy()

    xf_train, xf_test, yf_train, yf_test = train_test_split(xf, yf, test_size=0.15, random_state= 42)

    if type == "hip":    
    
        clf = XGBClassifier(max_depth = 3,
                            use_label_encoder=False,
                            objective = 'binary:logistic',
                            eval_metric = ['error', 'logloss'])
        
    elif type == "wrist":
        
        clf = XGBClassifier(max_depth = 10,
                            use_label_encoder=False,
                            objective = 'binary:logistic',
                            eval_metric = ['error', 'logloss'])
    
    else:
        
        print(f"Wrong type input {type}")
        exit()

    eval_set = [(xf_train, yf_train), (xf_test, yf_test)]
    clf.fit(xf_train, yf_train, eval_set = eval_set,early_stopping_rounds=10, verbose = 1)
    clf.save_model(SAVE_PATH / model_name)

    return clf

def CDPred4(features_path: Union[str, Path], type: str, model_name: str):
    """
    generate clinical + activity level feature only classifier CDPred-4

    :param features_path: clinical data path
    :param type: hip / wrist data
    :param model_name: saved model name
    :return: xgb classifier
    """
    df = pd.read_csv(features_path)

    data = pd.get_dummies(df)
    data = data.loc[:, (data != 0).any(axis=0)]

    Xf = data.loc[:, data.columns != "Y"]
    Yf = data.loc[:, "Y"]
    xf = Xf.to_numpy()
    yf = Yf.to_numpy()

    xf_train, xf_test, yf_train, yf_test = train_test_split(xf, yf, test_size=0.15, random_state= 42)

    if type == "hip":

        clf = XGBClassifier(n_estimators = 500,
                            max_depth = 3,
                            use_label_encoder=False,
                            objective = 'binary:logistic',
                            eval_metric = ['error', 'logloss'])

    elif type == "wrist":

        clf = XGBClassifier(max_depth = 9,
                            use_label_encoder=False,
                            objective = 'binary:logistic',
                            eval_metric = ['error', 'logloss'])

    else:

        print(f"Wrong type input {type}")
        exit()

    eval_set = [(xf_train, yf_train), (xf_test, yf_test)]
    clf.fit(xf_train, yf_train, eval_set = eval_set,early_stopping_rounds=10, verbose = 1)
    clf.save_model(SAVE_PATH / model_name)

    return clf

def CDPred4p(features_path: Union[str, Path], type: str, model_name: str):
    """
    generate complete feature only classifier CDPred-4+

    :param features_path: clinical data path
    :param type: hip / wrist data
    :param model_name: saved model name
    :return: xgb classifier
    """
    df = pd.read_csv(features_path)

    data = pd.get_dummies(df)
    data = data.loc[:, (data != 0).any(axis=0)]

    Xf = data.loc[:, data.columns != "Y"]
    Yf = data.loc[:, "Y"]
    xf = Xf.to_numpy()
    yf = Yf.to_numpy()

    xf_train, xf_test, yf_train, yf_test = train_test_split(xf, yf, test_size=0.15, random_state= 42)

    if type == "hip":

        clf = XGBClassifier(n_estimators = 50,
                            max_depth = 10,
                            subsample = .4,
                            colsample_bytree = .1,
                            gamma = .4,
                            learning_rate = .31,
                            use_label_encoder=False,
                            objective = 'binary:logistic',
                            eval_metric = ['error', 'logloss'])

    elif type == "wrist":

        clf = XGBClassifier(n_estimators = 500,
                            use_label_encoder= False,
                            min_child_weight = 1.2,
                            max_depth = 3,
                            learning_rate = .1,
                            subsample = 1,
                            colsample_bytree = .9,
                            gamma = 5,
                            nrounds = 1000,
                            objective = 'binary:logistic',
                            eval_metric = ['error', 'logloss'])

    else:

        print(f"Wrong type input {type}")
        exit()

    eval_set = [(xf_train, yf_train), (xf_test, yf_test)]
    clf.fit(xf_train, yf_train, eval_set = eval_set,early_stopping_rounds=10, verbose = 1)
    clf.save_model(SAVE_PATH / model_name)

    return clf

    