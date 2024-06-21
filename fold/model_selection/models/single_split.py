import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, SplitPeriod


class SingleSplit(BaseModel):
    """
    Initializes a `SingleSplit` instance representing a single split of data.

    Parameters
    ----------
    index : range
        The indices or data points to be split.
    split : int, float, slice, BaseTool, BasePeriod
        The split indices or slice defining the split.
    fix_ranges : bool, optional
        Whether to fix the ranges of the split. Default is True.
    allow_zero_len : bool, optional
        Whether to allow zero-length splits. Default is False.
    range_format : str, optional
        The format string to use when displaying range labels.
    freq : str, int, float, Offset, pd.Timedelta, optional
        The frequency of the data points.
    constraints : BaseTool, optional
        Constraints to apply on the splits.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : list of str, optional
        Labels for each split.
    sample_labels : list of str, optional
        Labels for the entire set of splits.

    Notes
    -----
    This class represents a single split of data based on the provided 
    index and split parameters. It inherits from the `BaseModel` class 
    for split handling.

    The `SplitPeriod` class is used internally to calculate the split 
    based on the given parameters.

    Examples
    --------
    >>> index = range(5)
    >>> split = 0.5
    >>> model = SingleSplit(
    ...     index,
    ...     split=split,
    ...     sample_labels=["IS", "OOS"],
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound      start        end
    set                        
    IS    2010-08-23 2017-06-27
    OOS   2017-06-27 2024-05-01
    """

    def __init__(
        self,
        index: range,
        split: int | float | slice | BaseTool | BasePeriod,
        fix_ranges: tp.Optional[bool] = True,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None
    ):
        model = SplitPeriod(
            period=slice(None),
            index=index,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq
        )
        new_split = model.split(split)
        splits = [new_split]
        super().__init__(
            index,
            splits,
            fix_ranges=fix_ranges,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            backwards=backwards,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )
