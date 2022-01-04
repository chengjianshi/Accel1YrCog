import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Any, List
from xgboost.sklearn import XGBClassifier
from tqdm import tqdm
from sklearn.metrics import plot_confusion_matrix, f1_score, accuracy_score, plot_roc_curve

def evaluate(m: XGBClassifier,
             x_train: Union[np.ndarray, pd.DataFrame],
             y_train: Union[np.ndarray, pd.DataFrame], 
             x_test: Union[np.ndarray, pd.DataFrame],
             y_test: Union[np.ndarray, pd.DataFrame]) -> None:
    
    result = m.evals_result()
    epochs = len(result["validation_0"]["error"])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(x_axis, result["validation_0"]["logloss"], label="Train")
    ax[0].plot(x_axis, result["validation_1"]["logloss"], label="Test")
    ax[0].legend()
    ax[0].set_ylabel("Log Loss")
    ax[0].set_title("XGBoost Classification logloss")

    ax[1].plot(x_axis, result["validation_0"]["error"], label="Train")
    ax[1].plot(x_axis, result["validation_1"]["error"], label="Test")
    ax[1].legend()
    ax[1].set_ylabel("Classification Error")
    ax[1].set_title("XGBoost Classification Error")
    
    fig, axes = plt.subplots(1,2, figsize = (10,5))
    
    yh = m.predict(x_train)
    
    print(f"Train f1 score : {f1_score(yh, y_train)}")
    
    accuracy = accuracy_score(yh, y_train)
    np.set_printoptions(precision=2)
    titles_options = [("Normalized confusion matrix (train)", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(m, x_train, y_train,
                                 display_labels=["cog-","cog+"],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, ax = axes[0])
    disp.ax_.set_title(title)
    
    yh = m.predict(x_test)
    
    print(f"Test f1 score : {f1_score(yh, y_test)}")
    
    accuracy = accuracy_score(yh, y_test)
    np.set_printoptions(precision=2)
    titles_options = [("Normalized confusion matrix (test)", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(m, x_test, y_test,
                                 display_labels=["cog-","cog+"],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize, ax = axes[1])
    disp.ax_.set_title(title)
    
    fig, axes = plt.subplots(1,1, figsize = (5,5))
    plot_roc_curve(m, x_train, y_train, ax = axes, name = "train") 
    plot_roc_curve(m, x_test, y_test, ax = axes, name = "test") 
    axes.set_title("ROC curve")
    axes.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle='dashed', color = "grey")

    
    plt.figure(figsize = (5,5))
    sorted_idx = m.feature_importances_.argsort()
    th = np.where(m.feature_importances_[sorted_idx] == 0)
    barx = x_train.columns[sorted_idx]
    bary = m.feature_importances_[sorted_idx]
    
    th = 0
    
    for idx in range(len(bary)):
        if bary[idx] != 0:
            th = idx
            break
    
    plt.barh(barx[th:], bary[th:])
    plt.xlabel("Xgboost Feature Importance")
    plt.show()


def bootstrap(model: Any, x: Union[np.array, pd.DataFrame], y: Union[np.array, pd.DataFrame],  metrics: Any, niters: int = 10000) -> List[float]:
    
    y_hat = model.predict(x)
    bootstrapped_scores = []
    rng = np.random.RandomState(1234)
    
    print("bootstrapping inference: ")
    
    y_hat = np.array(y_hat)
    y = np.array(y) 
     
    for _ in tqdm(range(niters)):
        indices = rng.randint(0, len(y_hat), len(y_hat))
        if len(np.unique(y[indices])) < 2:     
            continue
        score = metrics(y[indices], y_hat[indices])
        bootstrapped_scores.append(score)
    
    return bootstrapped_scores
    
    
    
