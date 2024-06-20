import pandas as pd
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

index = pd.date_range("2010", "2024", freq="D")


if __name__ == '__main__':
    
    ### Scikit-Learn Integration
    
    #### Example Model
    
    from fold import WrapperSplit, ExpandingSplit
    cv = WrapperSplit(
        model=ExpandingSplit,
        min_length=2,
        offset=1,
        split=-1,
        sample_labels=["IS", "OOS"]
    )
    
    #### Computing Cross-Validated Metrics
    
    # Pipeline and cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    pipe = make_pipeline(StandardScaler(),  LinearRegression())
    cross_val_score(pipe, X, y, cv=cv)
    # array([ ...
           
    #### Cross Validation Iterators
    
    model = cv.get_model(X)
    print(model.get_bounds(index_bounds=False))
    # bound         start  end
    # split sample            
    # 0     IS          0    1
    #       OOS         1    2
    # 1     IS          0    2
    #       OOS         2    3
    # 2     IS          0    3
    #       OOS         3    4
          
    for i, (train, test) in enumerate(cv.split(X)):
        print("Split %d:" % i)
        X_train, X_test = X[train], X[test]
        print("  X:", X_train.tolist(), X_test.tolist())
        y_train, y_test = y[train], y[test]
        print("  y:", y_train.tolist(), y_test.tolist())
    # Split 0:
    #   X: [[1, 2]] [[3, 4]]
    #   y: [1] [2]
    # Split 1:
    #   X: [[1, 2], [3, 4]] [[5, 6]]
    #   y: [1, 2] [3]
    # Split 2:
    #   X: [[1, 2], [3, 4], [5, 6]] [[7, 8]]
    #   y: [1, 2, 3] [4]
      
    #### Scikit-Learn models
    
    from sklearn.model_selection import TimeSeriesSplit
    from fold import SklearnFold
    
    model = SklearnFold(
        index,
        sk_model=TimeSeriesSplit(n_splits=2),
        sample_labels=["IS", "OOS"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2014-09-03
    #       OOS    2014-09-03 2019-05-04
    # 1     IS     2010-01-01 2019-05-04
    #       OOS    2019-05-04 2024-01-02
    
    ### Bespoke Models
    
    #### Expanding Number Split
    
    from fold import ExpandingNumberSplit
    model = ExpandingNumberSplit(
        index,
        n=5,
        min_length=360,
        split=-180,
        sample_labels=["IS", "OOS"],
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2010-06-30
    #       OOS    2010-06-30 2010-12-27
    # 1     IS     2010-01-01 2013-09-30
    #       OOS    2013-09-30 2014-03-29
    # 2     IS     2010-01-01 2017-01-01
    #       OOS    2017-01-01 2017-06-30
    # 3     IS     2010-01-01 2020-04-04
    #       OOS    2020-04-04 2020-10-01
    # 4     IS     2010-01-01 2023-07-06
    #       OOS    2023-07-06 2024-01-02
    
    #### Expanding Split
    
    from fold import ExpandingSplit
    model = ExpandingSplit(
        index,
        min_length=100,
        offset=50,
        split=-20,
        sample_labels=["IS", "OOS"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2010-03-22
    #       OOS    2010-03-22 2010-04-11
    # 1     IS     2010-01-01 2010-05-11
    #       OOS    2010-05-11 2010-05-31
    # 2     IS     2010-01-01 2010-06-30
    #                        ...
    # 98    OOS    2023-08-21 2023-09-10
    # 99    IS     2010-01-01 2023-10-10
    #       OOS    2023-10-10 2023-10-30
    # 100   IS     2010-01-01 2023-11-29
    #       OOS    2023-11-29 2023-12-19
    
    # [202 rows x 2 columns]
    
    #### Rolling Number Split
    
    from fold import RollingNumberSplit
    model = RollingNumberSplit(
        index,
        n=7,
        length=360,
        split=0.5,
        sample_labels=["IS", "OOS"]
    )
    model.get_bounds(index_bounds=True)
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2010-06-30
    #       OOS    2010-06-30 2010-12-27
    # 1     IS     2012-03-03 2012-08-30
    #       OOS    2012-08-30 2013-02-26
    # 2     IS     2014-05-05 2014-11-01
    #       OOS    2014-11-01 2015-04-30
    # 3     IS     2016-07-05 2017-01-01
    #       OOS    2017-01-01 2017-06-30
    # 4     IS     2018-09-05 2019-03-04
    #       OOS    2019-03-04 2019-08-31
    # 5     IS     2020-11-06 2021-05-05
    #       OOS    2021-05-05 2021-11-01
    # 6     IS     2023-01-07 2023-07-06
    #       OOS    2023-07-06 2024-01-02
    
    #### Rolling Split
    
    from fold import RollingSplit
    model = RollingSplit(
        index,
        60,
        split=1/2,
        sample_labels=["IS", "OOS"]
    )
    model.get_bounds(index_bounds=True)
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2010-01-31
    #       OOS    2010-01-31 2010-03-02
    # 1     IS     2010-01-31 2010-03-02
    #       OOS    2010-03-02 2010-04-01
    # 2     IS     2010-03-02 2010-04-01
    #                        ...
    # 166   OOS    2023-09-20 2023-10-20
    # 167   IS     2023-09-20 2023-10-20
    #       OOS    2023-10-20 2023-11-19
    # 168   IS     2023-10-20 2023-11-19
    #       OOS    2023-11-19 2023-12-19
    
    # [338 rows x 2 columns]
    
    #### Rolling Optimized Split
    
    from fold import RollingOptimizeSplit
    model = RollingOptimizeSplit(
        index,
        n=7,
        split=0.5,
        sample_labels=["IS", "OOS"]
    )
    model.get_bounds(index_bounds=True)
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2011-10-02
    #       OOS    2011-10-02 2013-07-02
    # 1     IS     2011-10-02 2013-07-02
    #       OOS    2013-07-02 2015-04-02
    # 2     IS     2013-07-02 2015-04-02
    #       OOS    2015-04-02 2016-12-31
    # 3     IS     2015-04-02 2016-12-31
    #       OOS    2016-12-31 2018-10-01
    # 4     IS     2016-12-31 2018-10-01
    #       OOS    2018-10-01 2020-07-01
    # 5     IS     2018-10-01 2020-07-01
    #       OOS    2020-07-01 2022-04-01
    # 6     IS     2020-07-01 2022-04-01
    #       OOS    2022-04-01 2023-12-31
    
    #### Interval Split
    
    from fold import IntervalSplit
    model = IntervalSplit(
        index,
        every="YS",
        closed_end=True,
        split=0.5,
        sample_labels=["IS", "OOS"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2010-07-03
    #       OOS    2010-07-03 2011-01-02
    # 1     IS     2011-01-01 2011-07-03
    #       OOS    2011-07-03 2012-01-02
    # 2     IS     2012-01-01 2012-07-02
    #       OOS    2012-07-02 2013-01-02
    # 3     IS     2013-01-01 2013-07-03
    #       OOS    2013-07-03 2014-01-02
    # 4     IS     2014-01-01 2014-07-03
    #       OOS    2014-07-03 2015-01-02
    # 5     IS     2015-01-01 2015-07-03
    #       OOS    2015-07-03 2016-01-02
    # 6     IS     2016-01-01 2016-07-02
    #       OOS    2016-07-02 2017-01-02
    # 7     IS     2017-01-01 2017-07-03
    #       OOS    2017-07-03 2018-01-02
    # 8     IS     2018-01-01 2018-07-03
    #       OOS    2018-07-03 2019-01-02
    # 9     IS     2019-01-01 2019-07-03
    #       OOS    2019-07-03 2020-01-02
    # 10    IS     2020-01-01 2020-07-02
    #       OOS    2020-07-02 2021-01-02
    # 11    IS     2021-01-01 2021-07-03
    #       OOS    2021-07-03 2022-01-02
    # 12    IS     2022-01-01 2022-07-03
    #       OOS    2022-07-03 2023-01-02
    # 13    IS     2023-01-01 2023-07-03
    #       OOS    2023-07-03 2024-01-02
    
    #### Calendar Split
    
    from fold import CalendarQuarterSplit
    model = CalendarQuarterSplit(
        index,
        n_train=2,
        n_test=1,
        sample_labels=["IS", "OOS"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2010-07-01
    #       OOS    2010-04-01 2010-07-01
    # 1     IS     2010-04-01 2010-10-01
    #       OOS    2010-07-01 2010-10-01
    # 2     IS     2010-07-01 2011-01-01
    #                        ...
    # 52    OOS    2023-04-01 2023-07-01
    # 53    IS     2023-04-01 2023-10-01
    #       OOS    2023-07-01 2023-10-01
    # 54    IS     2023-07-01 2024-01-01
    #       OOS    2023-10-01 2024-01-01
    
    # [110 rows x 2 columns]
    
    #### Grouper Split
    
    from fold import GrouperSplit
    model = GrouperSplit(
        index,
        by=index.year,
    #     split=0.5,  
        sample_labels=['IS', 'OOS']
    )
    print(model.get_bounds(index_bounds=True))
    # bound            start        end
    #      sample                      
    # 2010 IS     2010-10-13 2010-11-22
    #      OOS    2010-11-22 2011-01-01
    # 2011 IS     2011-01-01 2011-07-02
    #      OOS    2011-07-02 2012-01-01
    # 2012 IS     2012-01-01 2012-07-02
    #      OOS    2012-07-02 2013-01-01
    # 2013 IS     2013-01-01 2013-07-02
    #      OOS    2013-07-02 2014-01-01
    # 2014 IS     2014-01-01 2014-07-02
    #      OOS    2014-07-02 2015-01-01
    # 2015 IS     2015-01-01 2015-07-02
    #      OOS    2015-07-02 2016-01-01
    # 2016 IS     2016-01-01 2016-07-02
    #      OOS    2016-07-02 2017-01-01
    # 2017 IS     2017-01-01 2017-07-02
    #      OOS    2017-07-02 2018-01-01
    # 2018 IS     2018-01-01 2018-07-02
    #      OOS    2018-07-02 2019-01-01
    # 2019 IS     2019-01-01 2019-07-02
    #      OOS    2019-07-02 2020-01-01
    # 2020 IS     2020-01-01 2020-07-02
    #      OOS    2020-07-02 2021-01-01
    # 2021 IS     2021-01-01 2021-07-02
    #      OOS    2021-07-02 2022-01-01
    # 2022 IS     2022-01-01 2022-07-02
    #      OOS    2022-07-02 2023-01-01
    # 2023 IS     2023-01-01 2023-07-02
    #      OOS    2023-07-02 2024-01-01
    # 2024 IS     2024-01-01 2024-03-27
    
    #### Custom Function Split
    
    from fold import FunctionSplit, Key
    def split_func(index, prev_start):
        if prev_start is None:
            prev_start = index[0]
        new_start = prev_start + pd.offsets.MonthBegin(1)
        new_end = new_start + pd.DateOffset(years=1)
        if new_end > index[-1] + index.freq:
            return None
        return [
            slice(new_start, new_start + pd.offsets.MonthBegin(9)),
            slice(new_start + pd.offsets.MonthBegin(9), new_end)
        ]
    
    model = FunctionSplit(
        index,
        split_func=split_func,
        split_args=(Key("index"), Key("prev_start")),
        index_bounds=True,
        sample_labels=["IS", "OOS"],
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-02-01 2010-11-01
    #       OOS    2010-11-01 2011-02-01
    # 1     IS     2010-03-01 2010-12-01
    #       OOS    2010-12-01 2011-03-01
    # 2     IS     2010-04-01 2011-01-01
    #                        ...
    # 153   OOS    2023-08-01 2023-11-01
    # 154   IS     2022-12-01 2023-09-01
    #       OOS    2023-09-01 2023-12-01
    # 155   IS     2023-01-01 2023-10-01
    #       OOS    2023-10-01 2024-01-01
    
    # [312 rows x 2 columns]
    
    #### Random Split
    
    from fold import RandomNumberSplit
    model = RandomNumberSplit(
        index,
        50,
        min_length=360,
        split=0.5,
        sample_labels=["train", "test"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     train  2017-08-31 2018-02-27
    #       test   2018-02-27 2018-08-26
    # 1     train  2020-02-07 2020-08-05
    #       test   2020-08-05 2021-02-01
    # 2     train  2013-06-25 2013-12-22
    #                        ...
    # 47    test   2017-10-20 2018-04-18
    # 48    train  2017-10-20 2018-04-18
    #       test   2018-04-18 2018-10-15
    # 49    train  2014-08-29 2015-02-25
    #       test   2015-02-25 2015-08-24
    
    # [100 rows x 2 columns]
    
    #### Purged KFold
    
    from fold import PurgedKFold
    model = PurgedKFold(
        index,
        n_folds=10,
        n_test_folds=2,
        sample_labels=["IS", "OOS"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2021-03-16
    #       OOS    2021-03-16 2024-01-02
    # 1     IS     2010-01-01 2022-08-09
    #       OOS    2019-10-22 2024-01-02
    # 2     IS     2010-01-01 2024-01-02
    #                        ...
    # 42    OOS    2010-01-01 2015-08-11
    # 43    IS     2011-05-28 2024-01-02
    #       OOS    2010-01-01 2014-03-17
    # 44    IS     2012-10-21 2024-01-02
    #       OOS    2010-01-01 2012-10-21
    
    # [90 rows x 2 columns]
    
    #### Purged Walk-Forward
    
    from fold import PurgedWalkForwardSplit
    model = PurgedWalkForwardSplit(
        index,
        n_folds=10,
        n_test_folds=1,
        min_train_folds=2,
        max_train_folds=None,
        sample_labels=["IS", "OOS"]
    )
    print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2010-01-01 2012-10-21
    #       OOS    2012-10-21 2014-03-17
    # 1     IS     2010-01-01 2014-03-17
    #       OOS    2014-03-17 2015-08-11
    # 2     IS     2010-01-01 2015-08-11
    #       OOS    2015-08-11 2017-01-03
    # 3     IS     2010-01-01 2017-01-03
    #       OOS    2017-01-03 2018-05-29
    # 4     IS     2010-01-01 2018-05-29
    #       OOS    2018-05-29 2019-10-22
    # 5     IS     2010-01-01 2019-10-22
    #       OOS    2019-10-22 2021-03-16
    # 6     IS     2010-01-01 2021-03-16
    #       OOS    2021-03-16 2022-08-09
    # 7     IS     2010-01-01 2022-08-09
    #       OOS    2022-08-09 2024-01-02
    
    
    #### Train Test Split
    
    from fold import train_test_split
    
    ##### Create a 2-d dataset
    
    data = pd.DataFrame(
        np.random.normal(size=(len(index), 10)),
        index=index
    )
    
    ##### Select model
    
    model = PurgedWalkForwardSplit(index, sample_labels=["IS", "OOS"])
    
    
    ##### Get a pandas `DataFrame` object
    
    
    splitter = train_test_split(data, model)
    
    ##### Last valid split number
    
    last_split = splitter.last_valid_index()[0]
    
    ##### Train and test-set
    
    X_train = splitter[last_split, "IS"]
    y_test = splitter[last_split, 'OOS']
    
    ##### Apply transform
    
    
    from sklearn.preprocessing import StandardScaler
    transformer = StandardScaler().set_output(transform="pandas")
    transformer.fit(X_train).transform(y_test)
    #                   x0        x1        x2         x7        x8        x9
    # 2022-08-09 -0.576242 -0.225575  0.042182   0.520885 -0.610266 -0.194724
    # 2022-08-10  0.802063  0.434773  1.828127  -1.606077 -1.558970 -0.376618
    # 2022-08-11  0.391401  0.134264  2.245485   0.896719  0.684971 -0.846399
    # 2022-08-12  1.504788 -0.540543 -1.633605  -2.186249  1.044936 -0.853112
    # 2022-08-13 -0.749084  1.271873 -1.019302  -0.577152 -0.061101  0.526385
    #                                             ...
    # 2023-12-28 -1.143657 -0.639147 -2.203081  -0.028623 -0.756192 -0.558058
    # 2023-12-29 -0.183911  0.440218  0.670539   1.014517 -0.620626  0.175182
    # 2023-12-30 -0.416487 -0.570313  0.215304  -1.022257  0.240412  1.375879
    # 2023-12-31  1.897310  0.405948 -0.346925  -1.120385 -0.382428  0.381911
    # 2024-01-01  1.369621  1.100124  1.282244   1.018545  1.021911  1.326567
    
    # [511 rows x 10 columns]

