import math
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
from scipy.optimize import minimize_scalar
import typing as tp

from fold.model_selection.models.rolling_split import RollingSplit
from fold.tools import BaseTool, BasePeriod
from fold.utils.datetime import prepare_dt_index, infer_index_freq


def objective(
    length: int,
    index: range,
    ratio: float,
    n: int,
    split: int | float | slice | BaseTool | BasePeriod,
    optimize_anchor_set: int,
) -> int:
    """
    Compute the objective function for optimizing the length of rolling splits.

    Parameters
    ----------
    length : int
        The total length of the rolling window.
    index : range
        The index range.
    ratio : float
        The ratio for splitting the range.
    n : int
        The number of splits.
    split : int | float | slice | BaseTool | BasePeriod
        The split configuration.
    optimize_anchor_set : int
        Flag to determine optimization.

    Returns
    -------
    int
        The length of the empty part of the index or the total index length if
        the empty part is negative.
    """
    length = math.ceil(length)
    first_len = int(ratio * length)
    second_len = length - first_len
    if split is None or optimize_anchor_set == 0:
        empty_len = len(index) - (n * first_len + second_len)
    else:
        empty_len = len(index) - (n * second_len + first_len)
    if empty_len >= 0:
        return empty_len
    return len(index)


class RollingOptimizeSplit(RollingSplit):
    """
    Create rolling ranges of the same optimized length.

    Parameters
    ----------
    index : range
        The index range for the rolling split.
    n : int
        The number of rolling ranges.
    optimize_anchor_set : int, optional
        The index of a set to be non-overlapping, by default 1.
    split : int | float | slice | BaseTool | BasePeriod, optional
        The split ratio or configuration, by default None.
    allow_zero_len : bool, optional
        Whether to allow zero-length ranges, by default False.
    range_format : str, optional
        Format of the range, by default None.
    freq : str | int | float | Offset | pd.Timedelta, optional
        Frequency of the index, by default "auto".
    constraints : BaseTool, optional
        Additional constraints, by default None.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the split, by default None.
    sample_labels : range, optional
        Labels for the set, by default None.

    Notes
    -----
    This class searches for a range length that covers most of the index to 
    create `n` rolling ranges of the same length. The `optimize_anchor_set`
    parameter specifies the index of a set within the rolling ranges that 
    should become non-overlapping.

    Examples
    --------
    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = RollingOptimizeSplit(
    ...     index,
    ...     n=7,
    ...     split=0.5,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> model.get_bounds(index_bounds=True)
    bound          start        end
    split set                      
    0     IS  2020-01-01 2020-02-15
          OOS 2020-02-15 2020-04-01
    1     IS  2020-02-16 2020-04-01
          OOS 2020-04-01 2020-05-17
    2     IS  2020-04-02 2020-05-17
          OOS 2020-05-17 2020-07-02
    3     IS  2020-05-18 2020-07-02
          OOS 2020-07-02 2020-08-17
    4     IS  2020-07-03 2020-08-17
          OOS 2020-08-17 2020-10-02
    5     IS  2020-08-18 2020-10-02
          OOS 2020-10-02 2020-11-17
    6     IS  2020-10-03 2020-11-17
          OOS 2020-11-17 2021-01-02
    ```
    """

    def __init__(
        self,
        index: range,
        n: int,
        optimize_anchor_set: tp.Optional[int] = 1,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
    ):

        index = prepare_dt_index(index)

        try:
            freq = infer_index_freq(index, freq, allow_numeric=False)
        except Exception:
            freq = None

        if split is not None and not isinstance(split, (float, np.floating)):
            raise TypeError(
                "Split must be a float."
            )

        if optimize_anchor_set not in (0, 1):
            raise AssertionError(
                f"{optimize_anchor_set} not found in (0, 1)"
            )

        ratio = 1.0 if split is None else split
        length = math.ceil(
            minimize_scalar(objective, args=(
                index, ratio, n, split, optimize_anchor_set))
            .x
        )

        if split is None or optimize_anchor_set == 0:
            offset = int(ratio * length)

        else:
            offset = length - int(ratio * length)

        super().__init__(
            index,
            length=length,
            offset=offset,
            offset_anchor="prev_start",
            offset_anchor_set=None,
            split=split,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            backwards=backwards,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )

