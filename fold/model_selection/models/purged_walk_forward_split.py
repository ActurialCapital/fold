import pandas as pd
import typing as tp

from fold.model_selection.models.purged_split import PurgedSplit
from fold.model_selection import PurgedWalkForwardCV
from fold.utils.datetime import prepare_dt_index


class PurgedWalkForwardSplit(PurgedSplit):
    """
    Create a purged walk-forward cross-validator.

    It splits time-series data into training and testing sets while purging 
    overlapping data.

    Parameters
    ----------
    index : range
        The time-series index to be split.
    n_folds : int, optional
        Number of folds for cross-validation. Defaults to 10.
    n_test_folds : int, optional
        Number of folds used for testing. Defaults to 1.
    min_train_folds : int, optional
        Minimum number of folds used for training. Defaults to 2.
    max_train_folds : int, optional
        Maximum number of folds used for training. If None, it is inferred.
    split_by_time : bool, optional
        Whether to split by time instead of index position. Defaults to False.
    purge_td : str, int, float, or pd.Timedelta, optional
        Time duration to purge overlapping data. Defaults to 0.
    pred_times : pd.Index or pd.Series, optional
        Prediction times.
    eval_times : pd.Index or pd.Series, optional
        Evaluation times.
    **kwargs : dict
        Additional keyword arguments passed to the `PurgedSplit` method.

    Notes
    -----
    This class leverages the `PurgedWalkForwardCV` from the `fold` library to 
    ensure there is no data leakage between training and testing sets by 
    purging overlapping periods.

    Examples
    --------
    Create a purged walk-forward splitter with default parameters:

    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = PurgedWalkForwardSplit(
    ...     index,
    ...     n_folds=10,
    ...     n_test_folds=1,
    ...     min_train_folds=2,
    ...     max_train_folds=None,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2010-10-03 2013-06-29
          OOS    2013-06-29 2014-11-11
    1     IS     2010-10-03 2014-11-11
          OOS    2014-11-11 2016-03-25
    2     IS     2010-10-03 2016-03-25
          OOS    2016-03-25 2017-08-07
    3     IS     2010-10-03 2017-08-07
          OOS    2017-08-07 2018-12-20
    4     IS     2010-10-03 2018-12-20
          OOS    2018-12-20 2020-05-03
    5     IS     2010-10-03 2020-05-03
          OOS    2020-05-03 2021-09-15
    6     IS     2010-10-03 2021-09-15
          OOS    2021-09-15 2023-01-28
    7     IS     2010-10-03 2023-01-28
          OOS    2023-01-28 2024-06-11
    ```
    """

    def __init__(
        self,
        index: range,
        n_folds: tp.Optional[int] = 10,
        n_test_folds: tp.Optional[int] = 1,
        min_train_folds: tp.Optional[int] = 2,
        max_train_folds: tp.Optional[int] = None,
        split_by_time: tp.Optional[bool] = False,
        purge_td: tp.Optional[str | int | float | pd.Timedelta] = 0,
        pred_times: tp.Optional[pd.Index | pd.Series] = None,
        eval_times: tp.Optional[pd.Index | pd.Series] = None,
        **kwargs,
    ):
        index = prepare_dt_index(index)
        purged_splitter = PurgedWalkForwardCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            min_train_folds=min_train_folds,
            max_train_folds=max_train_folds,
            split_by_time=split_by_time,
            purge_td=purge_td,
        )
        super().__init__(
            index,
            purged_splitter,
            pred_times=pred_times,
            eval_times=eval_times,
            **kwargs,
        )
