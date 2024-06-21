import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, RelativePeriod, SplitPeriod
from fold.utils.datetime import prepare_dt_index, infer_index_freq


class ExpandingSplit(BaseModel):
    """
    Split index with an expanding range.

    Parameters
    ----------
    index : range
        The range of the index to be split.
    min_length : str, int, float, pd.Timedelta
        The minimum length of the expanding range. It can be a float between
        0 and 1 to make it relative to the length of the index.
    offset : str, int, float, pd.Timedelta
        The offset after the right bound of the previous range from which the 
        next range should start. It can also be a float relative to the index 
        length.
    split : int, float, slice, BaseTool, BasePeriod, optional
        The specific split to apply, by default None.
    allow_zero_len : bool, optional
        Whether to allow zero-length splits, by default False.
    range_format : str, optional
        Format for the range, by default None.
    freq : str, int, float, Offset, pd.Timedelta, optional
        Frequency of the index, by default "auto".
    index_bounds : bool, optional
        Whether to use index bounds, by default False.
    right_inclusive : bool, optional
        Whether the right bound is inclusive, by default False.
    constraints : BaseTool, optional
        Constraints for the splits, by default None.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the splits, by default None.
    sample_labels : range, optional
        Labels for the samples, by default None.

    Examples
    --------
    ```pycon
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = ExpandingSplit(
    ...     data.index,
    ...     min_length=1000,
    ...     offset=600,
    ...     split=-400,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2010-10-03 2011-04-01
          OOS    2011-04-01 2011-09-28
    1     IS     2010-10-03 2011-09-28
          OOS    2011-09-28 2012-03-26
    2     IS     2010-10-03 2012-03-26
          OOS    2012-03-26 2012-09-22
    3     IS     2010-10-03 2012-09-22
          OOS    2012-09-22 2013-03-21
    4     IS     2010-10-03 2013-03-21
          OOS    2013-03-21 2013-09-17
    5     IS     2010-10-03 2013-09-17
          OOS    2013-09-17 2014-03-16
    6     IS     2010-10-03 2014-03-16
          OOS    2014-03-16 2014-09-12
    7     IS     2010-10-03 2014-09-12
          OOS    2014-09-12 2015-03-11
    8     IS     2010-10-03 2015-03-11
          OOS    2015-03-11 2015-09-07
    9     IS     2010-10-03 2015-09-07
          OOS    2015-09-07 2016-03-05
    10    IS     2010-10-03 2016-03-05
          OOS    2016-03-05 2016-09-01
    11    IS     2010-10-03 2016-09-01
          OOS    2016-09-01 2017-02-28
    12    IS     2010-10-03 2017-02-28
          OOS    2017-02-28 2017-08-27
    13    IS     2010-10-03 2017-08-27
          OOS    2017-08-27 2018-02-23
    14    IS     2010-10-03 2018-02-23
          OOS    2018-02-23 2018-08-22
    15    IS     2010-10-03 2018-08-22
          OOS    2018-08-22 2019-02-18
    16    IS     2010-10-03 2019-02-18
          OOS    2019-02-18 2019-08-17
    17    IS     2010-10-03 2019-08-17
          OOS    2019-08-17 2020-02-13
    18    IS     2010-10-03 2020-02-13
          OOS    2020-02-13 2020-08-11
    19    IS     2010-10-03 2020-08-11
          OOS    2020-08-11 2021-02-07
    20    IS     2010-10-03 2021-02-07
          OOS    2021-02-07 2021-08-06
    21    IS     2010-10-03 2021-08-06
          OOS    2021-08-06 2022-02-02
    22    IS     2010-10-03 2022-02-02
          OOS    2022-02-02 2022-08-01
    23    IS     2010-10-03 2022-08-01
          OOS    2022-08-01 2023-01-28
    24    IS     2010-10-03 2023-01-28
          OOS    2023-01-28 2023-07-27
    25    IS     2010-10-03 2023-07-27
          OOS    2023-07-27 2024-01-23
    ```
    """

    def __init__(
        self,
        index: range,
        min_length: str | int | float | pd.Timedelta,
        offset: str | int | float | pd.Timedelta,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
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

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = RelativePeriod(
                    length=min_length,
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), index=index, freq=freq)
            else:
                prev_end = bounds[-1][-1][-1]
                new_split = RelativePeriod(
                    offset=offset,
                    offset_anchor="prev_end",
                    offset_space="all",
                    length=-1.0,
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), prev_end=prev_end, index=index, freq=freq)
                if new_split.stop <= prev_end:
                    raise ValueError(
                        "Infinite loop detected. Provide a positive offset.")
            if new_split.start < 0:
                raise ValueError("Range start cannot be negative")
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

                def _lambda_func(x): return SplitPeriod.get_period_bounds(
                    x,
                    index=index,
                    index_bounds=index_bounds,
                    right_inclusive=right_inclusive,
                    freq=freq
                )
                bounds.append(tuple(map(_lambda_func, new_split)))

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

