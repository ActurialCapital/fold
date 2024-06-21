import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, RelativePeriod, SplitPeriod
from fold.utils.datetime import prepare_dt_index, to_freq, infer_index_freq
from fold.utils import checks


class ExpandingNumberSplit(BaseModel):
    """
    Split index with number of expanding ranges.

    Picks `n` evenly-spaced, expanding ranges. Argument `min_length` 
    defines the minimum length for each range.

    Parameters
    ----------
    index : range
        The range of the index to be split.
    n : int
        Number of expanding ranges.
    min_length : str, int, float, or pd.Timedelta, optional
        Minimum length for each range. Default is None.
    split : int, float, slice, BaseTool, or BasePeriod, optional
        Split criteria. Default is None.
    fix_ranges : bool, optional
        Whether to fix ranges. Default is True.
    allow_zero_len : bool, optional
        Allow zero length. Default is False.
    range_format : str, optional
        Format for the range. Default is None.
    freq : str, int, float, Offset, or pd.Timedelta, optional
        Frequency of the index. Default is "auto".
    constraints : BaseTool, optional
        Constraints to apply. Default is None.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the splits. Default is None.
    sample_labels : range, optional
        Labels for the samples. Default is None.

    Raises
    ------
    TypeError
        If `min_length` is a floating number and not between 0 and 1.
        If `min_length` is not within the valid range.
    ValueError
        If `freq` is not provided when necessary.
    
    Examples
    --------
    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = ExpandingNumberSplit(
    ...     data.index,
    ...     n=5,
    ...     min_length=360,
    ...     split=-180,
    ...     sample_labels=["IS", "OOS"],
    ...     fix_ranges=True
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2010-10-03 2011-04-01
          OOS    2011-04-01 2011-09-28
    1     IS     2010-10-03 2014-06-04
          OOS    2014-06-04 2014-12-01
    2     IS     2010-10-03 2017-08-07
          OOS    2017-08-07 2018-02-03
    3     IS     2010-10-03 2020-10-10
          OOS    2020-10-10 2021-04-08
    4     IS     2010-10-03 2023-12-14
          OOS    2023-12-14 2024-06-11
    ```
    """
    def __init__(
        self,
        index: range,
        n: int,
        min_length: tp.Optional[str | int | float | pd.Timedelta] = None,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        fix_ranges: tp.Optional[bool] = True,
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

        min_length = len(index) // n if min_length is None else min_length

        if checks.is_float(min_length):
            if 0 <= abs(min_length) <= 1:
                min_length = len(index) * min_length
            elif not min_length.is_integer():
                raise TypeError(
                    "Floating number for minimum length must be "
                    "between 0 and 1"
                )

        if checks.is_int(min_length):
            min_length = int(min_length)
            if min_length < 1 or min_length > len(index):
                raise TypeError(
                    "Minimum length must be within "
                    f"[{1}, {len(index)}]"
                )
            lengths = np.arange(1, len(index) + 1)
            lengths = lengths[lengths >= min_length]
        else:
            min_length = to_freq(min_length)

            if freq is None:
                raise ValueError("Must provide freq")
            if min_length < freq or min_length > index[-1] + freq - index[0]:
                raise TypeError(
                    "Minimum length must be within "
                    f"[{freq}, {index[-1] + freq - index[0]}]"
                )
            lengths = index[1:].append(index[[-1]] + freq) - index[0]
            lengths = lengths[lengths >= min_length]

        if n > len(lengths):
            n = len(lengths)

        rows = np.round(np.linspace(0, len(lengths) - 1, n)).astype(int)
        lengths = lengths[rows]

        splits = []
        for length in lengths:
            new_split = RelativePeriod(length=length).to_slice(
                len(index),
                index=index,
                freq=freq
            )
            if split is not None:
                model = SplitPeriod(
                    period=new_split,
                    index=index,
                    allow_zero_len=allow_zero_len,
                    range_format=range_format,
                    freq=freq
                )
                new_split = model.split(split, backwards=backwards)

            splits.append(new_split)

        super().__init__(
            index,
            splits,
            fix_ranges=fix_ranges,
            backwards=backwards,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )


