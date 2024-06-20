import pandas as pd
import typing as tp

from fold.model_selection.models.purged_kfold_cv import PurgedKFoldCV
from fold.model_selection.models.purged_split import PurgedSplit


class PurgedKFold(PurgedSplit):
    """
    A class used to represent a Purged K-Fold cross-validation splitter.

    Parameters
    ----------
    index : range
        The index range to be used for splitting.
    n_folds : int, optional
        Number of folds, by default 10.
    n_test_folds : int, optional
        Number of test folds, by default 2.
    purge_td : str, int, float, or pd.Timedelta, optional
        Time delta for purging, by default 0.
    embargo_td : str, int, float, or pd.Timedelta, optional
        Time delta for embargo, by default 0.
    pred_times : pd.Index or pd.Series, optional
        Prediction times, by default None.
    eval_times : pd.Index or pd.Series, optional
        Evaluation times, by default None.
    **kwargs : dict
        Additional keyword arguments to be passed to the PurgedSplit constructor.

    Notes
    -----
    This class initializes a `PurgedKFoldCV` instance and uses it to create a 
    `PurgedSplit` instance. The `PurgedKFoldCV` handles the logic for purged 
    K-Fold cross-validation, including the purging and embargoing of data.

    Examples
    --------
    Create a purged KFold splitter with default parameters:

    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = PurgedKFold(
    ...     index,
    ...     n_folds=10,
    ...     n_test_folds=2,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2020-01-01 2020-10-22
          OOS    2020-10-22 2021-01-02
    1     IS     2020-01-01 2020-11-27
          OOS    2020-09-16 2021-01-02
    2     IS     2020-01-01 2021-01-02
                    ...        ...
    42    OOS    2020-01-01 2020-05-28
    43    IS     2020-02-07 2021-01-02
          OOS    2020-01-01 2020-04-21
    44    IS     2020-03-15 2021-01-02
          OOS    2020-01-01 2020-03-15
    ```
    """
    def __init__(
        self,
        index: range,
        n_folds: tp.Optional[int] = 10,
        n_test_folds: tp.Optional[int] = 2,
        purge_td: tp.Optional[str | int | float | pd.Timedelta] = 0,
        embargo_td: tp.Optional[str | int | float | pd.Timedelta] = 0,
        pred_times: tp.Optional[pd.Index | pd.Series] = None,
        eval_times: tp.Optional[pd.Index | pd.Series] = None,
        **kwargs,
    ):
        purged_splitter = PurgedKFoldCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            purge_td=purge_td,
            embargo_td=embargo_td,
        )
        super().__init__(
            index,
            purged_splitter,
            pred_times=pred_times,
            eval_times=eval_times,
            **kwargs,
        )
