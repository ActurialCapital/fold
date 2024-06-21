import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, RelativePeriod, SplitPeriod
from fold.utils.datetime import prepare_dt_index, to_freq, infer_index_freq


class RollingNumberSplit(BaseModel):
    """
    Create a `BaseModel` instance from a number of rolling ranges of the same 
    length.

    Notes
    -----
    If `length` is None, splits the index evenly into `n` non-overlapping ranges. 
    Otherwise, picks `n` evenly-spaced, potentially overlapping ranges  of a 
    fixed length.

    Parameters
    ----------
    index : range
        The index to split.
    n : int
        The number of splits.
    length : str, int, float, pd.Timedelta, optional
        The length of each split. If None, the index is split into `n` 
        non-overlapping ranges.
    split : int, float, slice, BaseTool, BasePeriod, optional
        The split point within each range.
    allow_zero_len : bool, optional
        Whether to allow zero length splits.
    range_format : str, optional
        The format of the range.
    freq : str, int, float, Offset, pd.Timedelta, optional
        The frequency of the index. Default is "auto".
    constraints : BaseTool, optional
        Constraints for the split.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the splits.
    sample_labels : range, optional
        Labels for the samples.

    Examples
    --------
    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = RollingNumberSplit(
    ...     index,
    ...     n=7,
    ...     length=360,
    ...     split=0.5,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> model.get_bounds(index_bounds=True)
    bound          start        end
    split set                      
    0     IS  2020-01-01 2020-06-29
          OOS 2020-06-29 2020-12-26
    1     IS  2020-01-02 2020-06-30
          OOS 2020-06-30 2020-12-27
    2     IS  2020-01-03 2020-07-01
          OOS 2020-07-01 2020-12-28
    3     IS  2020-01-05 2020-07-03
          OOS 2020-07-03 2020-12-30
    4     IS  2020-01-06 2020-07-04
          OOS 2020-07-04 2020-12-31
    5     IS  2020-01-07 2020-07-05
          OOS 2020-07-05 2021-01-01
    6     IS  2020-01-08 2020-07-06
          OOS 2020-07-06 2021-01-02
    ```
    """
    def __init__(
        self,
        index: range,
        n: int,
        length: str | int | float | pd.Timedelta,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None
    ):

        index = prepare_dt_index(index)
        
        try:
            freq = infer_index_freq(index, freq, allow_numeric=False)
        except Exception:
            freq = None

        if isinstance(length, (float, np.floating)):
            if 0 <= abs(length) <= 1:
                length = len(index) * length

            elif not length.is_integer():
                raise TypeError(
                    "Floating number for length must be between 0 and 1"
                )

            length = int(length)

        if (
            isinstance(length, (int, np.integer)) and not
            isinstance(length, np.timedelta64)
        ):
            if length < 1 or length > len(index):
                raise TypeError(
                    f"Length must be within [{1}, {len(index)}]"
                )

            offsets = np.arange(len(index))
            offsets = offsets[offsets + length <= len(index)]

        else:
            length = to_freq(length)
            if freq is None:
                raise ValueError("Must provide freq")

            if length < freq or length > index[-1] + freq - index[0]:
                raise TypeError(
                    "Length must be within "
                    f"[{freq}, {index[-1] + freq - index[0]}]"
                )

            offsets = index[index + length <= index[-1] + freq] - index[0]

        if n > len(offsets):
            n = len(offsets)

        rows = np.round(np.linspace(0, len(offsets) - 1, n)).astype(int)
        offsets = offsets[rows]

        splits = []
        for offset in offsets:
            new_split = RelativePeriod(
                offset=offset,
                length=length,
            ).to_slice(
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
            backwards=backwards,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )

