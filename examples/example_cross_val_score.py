from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))


if __name__ == '__main__':


    from fold import WrapperSplit, RandomNumberSplit
    
    wrapper = WrapperSplit(
        model=RandomNumberSplit,
        n=50,
        min_length=360,
        seed=42,
        split=0.5,
        freq='D',
        sample_labels=["IS", "OOS"]
    )

    # Example 1: Computing cross-validated metrics
    # --------------------------------------------
    
    length = 5000
    n_paths = 10

    end_date = datetime.now().date()

    start_date = end_date - timedelta(days=length - 1)

    X = pd.DataFrame(
        np.random.normal(size=(length, n_paths)),
        columns=[f'path_{n}' for n in range(1, n_paths + 1)],
        index=pd.date_range(start=start_date, end=end_date, freq="D")
    )

    y = pd.DataFrame(
        np.random.normal(size=(length, n_paths)),
        columns=[f'path_{n}' for n in range(1, n_paths + 1)],
        index=pd.date_range(start=start_date, end=end_date, freq="D")
    )
    
    # Estimator
    # =========
    estimator = LinearRegression()
    cross_val_score(estimator, X, y, cv=wrapper)
    # array([-0.07800858, -0.07658673, -0.0845051 , -0.08765034, -0.07774631,
    #        -0.06173064, -0.07189138, -0.09779912, -0.0564926 , -0.08828377,
    #        -0.08725884, -0.06885268, -0.03752609, -0.07538475, -0.05556132,
    #        -0.04854908, -0.08044684, -0.06788325, -0.10354424, -0.07574143,
    #        -0.08691646, -0.057517  , -0.10867424, -0.06416291, -0.06329993,
    #        -0.05605754, -0.07294834, -0.07718914, -0.08554365, -0.08536413,
    #        -0.07574143, -0.07954433, -0.09045388, -0.06616035, -0.08940118,
    #        -0.07274754, -0.05876806, -0.07969124, -0.08607394, -0.05323754,
    #        -0.08873512, -0.06894304, -0.08722127, -0.04268143, -0.07980813,
    #        -0.07845898, -0.06979668, -0.08932076, -0.07746637, -0.08359681])
    
    # Pipeline
    # ========
    
    pipe = make_pipeline(StandardScaler(),  LinearRegression())
    cross_val_score(pipe, X, y, cv=wrapper)
    # array([-0.07800858, -0.07658673, -0.0845051 , -0.08765034, -0.07774631,
    #        -0.06173064, -0.07189138, -0.09779912, -0.0564926 , -0.08828377,
    #        -0.08725884, -0.06885268, -0.03752609, -0.07538475, -0.05556132,
    #        -0.04854908, -0.08044684, -0.06788325, -0.10354424, -0.07574143,
    #        -0.08691646, -0.057517  , -0.10867424, -0.06416291, -0.06329993,
    #        -0.05605754, -0.07294834, -0.07718914, -0.08554365, -0.08536413,
    #        -0.07574143, -0.07954433, -0.09045388, -0.06616035, -0.08940118,
    #        -0.07274754, -0.05876806, -0.07969124, -0.08607394, -0.05323754,
    #        -0.08873512, -0.06894304, -0.08722127, -0.04268143, -0.07980813,
    #        -0.07845898, -0.06979668, -0.08932076, -0.07746637, -0.08359681])
    

    # Example 2: Cross validation iterators
    # -------------------------------------

    for i, (train, test) in enumerate(wrapper.split(X)):
        print("Split %d:" % i)
        X_train, X_test = X.iloc[train], X.iloc[test]
        print("  X:", X_train, X_test)
        y_train, y_test = y.iloc[train], y.iloc[test]
        print("  y:", y_train, y_test)