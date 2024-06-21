from datetime import time
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection import IntervalSplit
from fold.tools import BaseTool
from fold.utils.datetime import prepare_dt_index

    
class CalendarSplit(IntervalSplit):
    """
    Split index by beginning of calendar years.

    Parameters
    ----------
    index : range
        The range of the index to be split.
    n_train : int
        Number of training years. Example `n_train=3` is 3 years training set.
    n_test : int
        Number of testing years. Example `n_test=1` is 1 year testing set.
    index_freq : str, int, float, Offset, or pd.Timedelta, optional
        Frequency of the index, e.g., 'D' for daily, 'M' for monthly.
    every : str, optional
        Calendar frequency to split the index with.
    normalize_every : bool, optional
        Normalize start/end dates to midnight before generating date range.
    split_every : bool, optional
        Whether to split the sequence generated using `every` into `start` and
        `end` arrays. After creation, and if `split_every` is True, an index 
        range is created from each pair of elements in the generated sequence. 
        Otherwise, the entire sequence is assigned to `start` and `end`, and 
        only time and delta instructions can be used to further differentiate 
        between them. Forced to False if `every`, `start_time`, and `end_time` 
        are not None and `fixed_start` is False.
    start_time : str or datetime.time, optional
        Start time of the day either as a (human-readable) string or 
        `datetime.time`. Every datetime in `start` gets floored to the daily 
        frequency, while `start_time` gets converted into a timedelta using 
        `.datetime_.time_to_timedelta` and added to `add_start_delta`. Index 
        must be datetime-like.
    end_time : str or datetime.time, optional
        End time of the day either as a (human-readable) string or 
        `datetime.time`. Every datetime in `end` gets floored to the daily 
        frequency, while `end_time` gets converted into a timedelta using 
        `.datetime_.time_to_timedelta` and added to `add_end_delta`. Index must 
        be datetime-like.
    start : str, int, float, Offset, or pd.Timestamp, optional
        Start index/label or a sequence of such. Gets converted into datetime 
        format whenever possible. Gets broadcasted together with `end`.
    end : str, int, float, Offset, or pd.Timestamp, optional
        End index/label or a sequence of such. Gets converted into datetime 
        format whenever possible. Gets broadcasted together with `start`.
    exact_start : bool, optional
        Whether the first index in the `start` array should be exactly `start`.
        Depending on `every`, the first index picked by `pd.date_range` may 
        happen after `start`. In such a case, `start` gets injected before the 
        first index generated by `pd.date_range`. Cannot be used together with 
        `lookback_period`.
    closed_start : bool, optional
        Whether `start` should be inclusive.
    closed_end : bool, optional
        Whether `end` should be inclusive.
    add_start_delta :  str, int, float, Offset, or pd.Timedelta, optional
        Offset to be added to each in `start`. If string, gets converted into 
        an offset using pandas.
    add_end_delta : str, int, float, Offset, or pd.Timedelta, optional
        Offset to be added to each in `end`. If string, gets converted into an
        offset using pandas.
    kind : str, optional
        Kind of data in `on`: indices, labels, or bounds.
        If None, gets assigned to `indices` if `start` and `end` contain 
        integer data, to `bounds` if `start`, `end`, and index are 
        datetime-like, otherwise to `labels`. If `kind` is 'labels', `start` 
        and `end` get converted into indices using `pd.Index.get_indexer`.
        Prior to this, get their timezone aligned to the timezone of the index. 
        If `kind` is 'indices', `start` and `end` get wrapped with NumPy. If 
        kind` is 'bounds', `.Resampler.map_bounds_to_source_ranges` is used.
    allow_zero_len : bool, optional
        Allow zero length. Default is False.
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

    """

    def __init__(
        self,
        index: range,
        n_train: int,
        n_test: int,
        *,
        index_freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        every: tp.Optional[str] = "YS",
        normalize_every: tp.Optional[bool] = False,
        split_every: tp.Optional[bool] = True,
        start_time: tp.Optional[str | time] = None,
        end_time: tp.Optional[str | time] = None,
        start: tp.Optional[str | int | float | Offset | pd.Timestamp] = None,
        end: tp.Optional[str | int | float | Offset | pd.Timestamp] = None,
        exact_start: tp.Optional[bool] = False,
        closed_start: tp.Optional[bool] = True,
        closed_end: tp.Optional[bool] = False,
        add_start_delta: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        add_end_delta: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        kind: tp.Optional[str] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,

    ):
        index = prepare_dt_index(index)
        total_length = n_train + n_test
        fixed_start = False
        
        if every.lower().startswith("y"):
            lookback_period = f"{total_length}{every}"
            split = (
                lambda index: index.year != index.year[-1],
                lambda index: index.year == index.year[-1]
            )
        elif every.lower().startswith("q"):
            lookback_period=f"{n_train}{every}"
            split=(
                lambda index: ( 
                    index < index[0] + pd.offsets.QuarterBegin(
                        n_train, 
                        startingMonth=1
                    )
                ),
                lambda index: ( 
                    index >= index[0] + pd.offsets.QuarterBegin(
                        n_test, 
                        startingMonth=1
                    )
                )
            )
        elif every.lower().startswith("m"):
            lookback_period=f"{n_train}{every}"
            split=(
                lambda index: ( 
                    index < index[0] + pd.offsets.MonthBegin(n_train)
                ),
                lambda index: ( 
                    index >= index[0] + pd.offsets.MonthBegin(n_test)
                )
            )
        else:
            raise NotImplementedError(
                f'Parameter "every={every}" is not implemented.'
            )

        super().__init__(
            index=index,
            index_freq=index_freq,
            every=every,
            normalize_every=normalize_every,
            split_every=split_every,
            start_time=start_time,
            end_time=end_time,
            lookback_period=lookback_period,
            start=start,
            end=end,
            exact_start=exact_start,
            fixed_start=fixed_start,
            closed_start=closed_start,
            closed_end=closed_end,
            add_start_delta=add_start_delta,
            add_end_delta=add_end_delta,
            kind=kind,
            split=split,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            backwards=backwards,
            split_labels=split_labels,
            sample_labels=sample_labels,
            
        )

