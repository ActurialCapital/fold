import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, RelativePeriod, SplitPeriod
from fold.utils.datetime import prepare_dt_index, infer_index_freq


class RollingSplit(BaseModel):
    """
    Initialize a RollingSplit instance to create rolling ranges of fixed length.

    Parameters
    ----------
    index : range
        The index to be used for the rolling range.
    length : str, int, float, pd.Timedelta
        The fixed length of each rolling range segment.
    offset : str, int, float, pd.Timedelta, optional
        The offset value for each range, by default 0.
    offset_anchor : str, optional
        The anchor point for the offset, by default "prev_end".
    offset_anchor_set : int, optional
        The set from the previous range to be used as an offset anchor, 
        by default 0.
    offset_space : str, optional
        The spacing for the offset, by default "prev".
    split : int, float, slice, BaseTool, BasePeriod, optional
        Ranges to split the range into, by default None.
    allow_zero_len : bool, optional
        Whether to allow zero-length ranges, by default False.
    range_format : str, optional
        The format of the range, by default None.
    freq : str, int, float, Offset, pd.Timedelta, optional
        The index frequency in case it cannot be parsed from `index`, 
        by default "auto".
    index_bounds : bool, optional
        Whether to use index bounds or not, by default False.
    right_inclusive : bool, optional
        Whether the right bound is inclusive or not, by default False.
    constraints : BaseTool, optional
        Additional constraints, by default None.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the split, by default None.
    sample_labels : range, optional
        Labels for the set, by default None.

    Examples
    --------
    Example 1: Divide a range into a set of non-overlapping ranges:

    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = RollingSplit(index, length=30)
    >>> model.get_bounds(index_bounds=True)
    bound      start        end
    split                      
    0     2020-01-01 2020-01-31
    1     2020-01-31 2020-03-01
    2     2020-03-01 2020-03-31
    3     2020-03-31 2020-04-30
    4     2020-04-30 2020-05-30
    5     2020-05-30 2020-06-29
    6     2020-06-29 2020-07-29
    7     2020-07-29 2020-08-28
    8     2020-08-28 2020-09-27
    9     2020-09-27 2020-10-27
    10    2020-10-27 2020-11-26
    11    2020-11-26 2020-12-26
    ```

    Example 2: Divide a range into ranges, each split into 1/2:

    ```pycon
    >>> model = RollingSplit(
    ...     index,
    ...     60,
    ...     split=1/2,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> model.get_bounds(index_bounds=True)
    bound             start        end
    split sample                      
    0     IS     2020-01-01 2020-01-31
          OOS    2020-01-31 2020-03-01
    1     IS     2020-01-31 2020-03-01
          OOS    2020-03-01 2020-03-31
    2     IS     2020-03-01 2020-03-31
          OOS    2020-03-31 2020-04-30
    3     IS     2020-03-31 2020-04-30
          OOS    2020-04-30 2020-05-30
    4     IS     2020-04-30 2020-05-30
          OOS    2020-05-30 2020-06-29
    5     IS     2020-05-30 2020-06-29
          OOS    2020-06-29 2020-07-29
    6     IS     2020-06-29 2020-07-29
          OOS    2020-07-29 2020-08-28
    7     IS     2020-07-29 2020-08-28
          OOS    2020-08-28 2020-09-27
    8     IS     2020-08-28 2020-09-27
          OOS    2020-09-27 2020-10-27
    9     IS     2020-09-27 2020-10-27
          OOS    2020-10-27 2020-11-26
    10    IS     2020-10-27 2020-11-26
          OOS    2020-11-26 2020-12-26
    ```

    Example 3: Make the ranges above non-overlapping by using the right 
    bound of the last set as an offset anchor:

    ```pycon
    >>> model = RollingSplit(
    ...     index,
    ...     60,
    ...     offset_anchor_set=-1,
    ...     split=1/2,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> model.get_bounds(index_bounds=True)
    bound             start        end
    split sample                      
    0     IS     2020-01-01 2020-01-31
          OOS    2020-01-31 2020-03-01
    1     IS     2020-03-01 2020-03-31
          OOS    2020-03-31 2020-04-30
    2     IS     2020-04-30 2020-05-30
          OOS    2020-05-30 2020-06-29
    3     IS     2020-06-29 2020-07-29
          OOS    2020-07-29 2020-08-28
    4     IS     2020-08-28 2020-09-27
          OOS    2020-09-27 2020-10-27
    5     IS     2020-10-27 2020-11-26
          OOS    2020-11-26 2020-12-26
    ```
    """

    def __init__(
        self,
        index: range,
        length: str | int | float | pd.Timedelta,
        offset: tp.Optional[str | int | float | pd.Timedelta] = 0,
        offset_anchor: tp.Optional[str] = "prev_end",
        offset_anchor_set: tp.Optional[int] = 0,
        offset_space: tp.Optional[str] = "prev",
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
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

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                model = RelativePeriod(
                    length=length,
                    offset_anchor="start",
                    out_of_bounds="keep",
                )
                new_split = model.to_slice(
                    total_len=len(index),
                    index=index,
                    freq=freq
                )

            else:
                if offset_anchor_set is None:
                    prev_start = bounds[-1][0][0]
                    prev_end = bounds[-1][-1][1]

                else:
                    prev_start, prev_end = bounds[-1][offset_anchor_set]

                model = RelativePeriod(
                    offset=offset,
                    offset_anchor=offset_anchor,
                    offset_space=offset_space,
                    length=length,
                    length_space="all",
                    out_of_bounds="keep",
                )
                new_split = model.to_slice(
                    total_len=len(index),
                    prev_start=prev_start,
                    prev_end=prev_end,
                    index=index,
                    freq=freq
                )

                if new_split.start <= bounds[-1][0][0]:
                    raise ValueError(
                        "Infinite loop detected. Provide a positive offset."
                    )

            if new_split.start < 0:
                raise ValueError(
                    "Range start cannot be negative"
                )

            if new_split.stop > len(index):
                break

            if split is not None:
                model = SplitPeriod(
                    period=new_split,
                    index=index,
                    allow_zero_len=allow_zero_len,
                    range_format=range_format,
                    freq=freq
                )
                new_split = model.split(split, backwards=backwards)

                data = tuple(map(
                    lambda x: SplitPeriod.get_period_bounds(
                        x,
                        index=index,
                        index_bounds=index_bounds,
                        right_inclusive=right_inclusive,
                        freq=freq
                    ),
                    new_split,
                ))
                bounds.append(data)

            else:
                bounds.append(((new_split.start, new_split.stop),))

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

