from datetime import time
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, SplitPeriod
from fold.utils.indexing import repeat_index
from fold.utils.resampler import CustomResampler
from fold.utils.datetime import (
    prepare_dt_index,
    infer_index_freq,
    to_freq,
    try_align_to_dt_index,
    date_range,
    time_to_timedelta
)
from fold.utils import checks


def get_index_ranges(
    index: pd.Index,
    index_freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
    every: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
    normalize_every: tp.Optional[bool] = False,
    split_every: tp.Optional[bool] = True,
    start_time: tp.Optional[str | time] = None,
    end_time: tp.Optional[str | time] = None,
    lookback_period: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
    start: tp.Optional[str | int | float | Offset | pd.Timestamp] = None,
    end: tp.Optional[str | int | float | Offset | pd.Timestamp] = None,
    exact_start: tp.Optional[bool] = False,
    fixed_start: tp.Optional[bool] = False,
    closed_start: tp.Optional[bool] = True,
    closed_end: tp.Optional[bool] = False,
    add_start_delta: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
    add_end_delta: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
    kind: tp.Optional[str] = None,
    skip_not_found: bool = True
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Translate indices, labels, or bounds into index ranges.

    Parameters
    ----------
    index : pd.Index
        Input index.
    index_freq : str, int, float, Offset, or pd.Timedelta, optional
        Frequency of the index, e.g., 'D' for daily, 'M' for monthly.
    every : str, int, float, Offset, or pd.Timedelta, optional
        Frequency either as an integer or timedelta. If integer, an index 
        sequence from `start` to `end` (exclusive) is created with 'indices' 
        as `kind`. If timedelta-like, a date sequence from `start` to `end` 
        (inclusive) is created with 'bounds' as `kind`.
    normalize_every : bool, optional
        Normalize start/end dates to midnight before generating date range.
    split_every : bool, optional
        Whether to split the sequence generated using `every` into `start` and 
        `end` arrays.
    start_time : str or datetime.time, optional
        Start time of the day.
    end_time : str or datetime.time, optional
        End time of the day.
    lookback_period : str, int, float, Offset, or pd.Timedelta, optional
        Lookback period either as an integer or offset.
    start : str, int, float, Offset, or pd.Timestamp, optional
        Start index/label or a sequence of such.
    end : str, int, float, Offset, or pd.Timestamp, optional
        End index/label or a sequence of such.
    exact_start : bool, optional
        Whether the first index in the `start` array should be exactly `start`.
    fixed_start : bool, optional
        Whether all indices in the `start` array should be exactly `start`.
    closed_start : bool, optional
        Whether `start` should be inclusive.
    closed_end : bool, optional
        Whether `end` should be inclusive.
    add_start_delta : str, int, float, Offset, or pd.Timedelta, optional
        Offset to be added to each in `start`.
    add_end_delta : str, int, float, Offset, or pd.Timedelta, optional
        Offset to be added to each in `end`.
    kind : str, optional
        Kind of data in `on`: indices, labels, or bounds.
    skip_not_found : bool, optional
        Whether to drop indices that are -1 (not found).

    Returns
    -------
    tuple of np.ndarray
        Computed start and end indices for the ranges.

    Raises
    ------
    ValueError
        If `fixed_start` and `lookback_period` are used together, or if 
        `exact_start` and `lookback_period` are used together, or if `start` 
        and `lookback_period` are used together.
    AssertionError
        If `kind` is not 'indices', 'labels' or 'bounds', or if lengths of 
        `start` and `end` do not match, or if some start indices are equal to 
        or higher than end indices.
    """
    index = prepare_dt_index(index)
    if isinstance(index, pd.DatetimeIndex):
        if start is not None:
            start = try_align_to_dt_index(start, index)
            if isinstance(start, pd.DatetimeIndex):
                start = start.tz_localize(None)
        if end is not None:
            end = try_align_to_dt_index(end, index)
            if isinstance(end, pd.DatetimeIndex):
                end = end.tz_localize(None)
        naive_index = index.tz_localize(None)
    else:
        if start is not None:
            if not isinstance(start, pd.Index):
                try:
                    start = pd.Index(start)
                except Exception:
                    start = pd.Index([start])
        if end is not None:
            if not isinstance(end, pd.Index):
                try:
                    end = pd.Index(end)
                except Exception:
                    end = pd.Index([end])
        naive_index = index

    if lookback_period is not None and not checks.is_int(lookback_period):
        try:
            lookback_period = to_offset(lookback_period)
        except Exception:
            lookback_period = to_offset(pd.Timedelta(lookback_period))

    if fixed_start and lookback_period is not None:
        raise ValueError(
            "Cannot use fixed_start and lookback_period together"
        )

    if exact_start and lookback_period is not None:
        raise ValueError(
            "Cannot use exact_start and lookback_period together"
        )

    if start_time is not None or end_time is not None:
        if every is None and start is None and end is None:
            every = "D"

    if every is not None:
        if not fixed_start:
            if start_time is None and end_time is not None:
                start_time = time(0, 0, 0, 0)
                closed_start = True
            if start_time is not None and end_time is None:
                end_time = time(0, 0, 0, 0)
                closed_end = False

        if start_time is not None and end_time is not None and not fixed_start:
            split_every = False

        if checks.is_int(every):
            if start is None:
                start = 0
            else:
                start = start[0]

            if end is None:
                end = len(naive_index)
            else:
                end = end[-1]

            if closed_end:
                end -= 1

            if lookback_period is None:
                new_index = np.arange(start, end + 1, every)
                if not split_every:
                    start = end = new_index
                else:
                    if fixed_start:
                        start = np.full(len(new_index) - 1, new_index[0])
                    else:
                        start = new_index[:-1]
                    end = new_index[1:]

            else:
                end = np.arange(start + lookback_period, end + 1, every)
                start = end - lookback_period

            kind = "indices"
            lookback_period = None

        else:
            if start is None:
                start = 0
            else:
                start = start[0]

            if checks.is_int(start):
                start_date = naive_index[start]
            else:
                start_date = start

            if end is None:
                end = len(naive_index) - 1
            else:
                end = end[-1]

            if checks.is_int(end):
                end_date = naive_index[end]
            else:
                end_date = end

            if lookback_period is None:
                new_index = date_range(
                    start_date,
                    end_date,
                    freq=every,
                    normalize=normalize_every,
                    inclusive="both",
                )

                if exact_start and new_index[0] > start_date:
                    new_index = new_index.insert(0, start_date)
                if not split_every:
                    start = end = new_index
                else:
                    if fixed_start:
                        start = repeat_index(
                            new_index[[0]], len(new_index) - 1)
                    else:
                        start = new_index[:-1]
                    end = new_index[1:]

            else:
                if checks.is_int(lookback_period):
                    lookback_period *= infer_index_freq(
                        naive_index, freq=index_freq)

                end = date_range(
                    start_date + lookback_period,
                    end_date,
                    freq=every,
                    normalize=normalize_every,
                    inclusive="both",
                )
                start = end - lookback_period

            kind = "bounds"
            lookback_period = None

    if kind is None:

        if start is None and end is None:
            kind = "indices"
        else:
            if start is not None:
                ref_index = start

            if end is not None:
                ref_index = end

            if pd.api.types.is_integer_dtype(ref_index):
                kind = "indices"

            elif (
                isinstance(ref_index, pd.DatetimeIndex) and
                isinstance(naive_index, pd.DatetimeIndex)
            ):
                kind = "bounds"

            else:
                kind = "labels"

    if kind not in ("indices", "labels", "bounds"):
        raise AssertionError(
            f"{kind} is not 'indices', 'labels' or 'bounds'"
        )

    if end is None:

        if kind.lower() in ("labels", "bounds"):
            end = pd.Index([naive_index[-1]])

        else:
            end = pd.Index([len(naive_index)])

    if start is not None and lookback_period is not None:
        raise ValueError(
            "Cannot use start and lookback_period together"
        )

    if start is None:

        if lookback_period is None:
            if kind.lower() in ("labels", "bounds"):
                start = pd.Index([naive_index[0]])
            else:
                start = pd.Index([0])

        else:
            if (
                checks.is_int(lookback_period) and not
                pd.api.types.is_integer_dtype(end)
            ):
                lookback_period *= infer_index_freq(
                    naive_index, freq=index_freq)

            start = end - lookback_period

    if len(start) == 1 and len(end) > 1:
        start = repeat_index(start, len(end))

    elif len(start) > 1 and len(end) == 1:
        end = repeat_index(end, len(start))

    if len(start) != len(end):
        raise AssertionError("Lengths of 'start' and 'end' do not match")

    if start_time is not None:
        checks.assert_instance_of(start, pd.DatetimeIndex)
        start = start.floor("D")
        add_start_time_delta = time_to_timedelta(start_time)

        if add_start_delta is None:
            add_start_delta = add_start_time_delta
        else:
            add_start_delta += add_start_time_delta

    else:
        add_start_time_delta = None

    if end_time is not None:
        checks.assert_instance_of(end, pd.DatetimeIndex)
        end = end.floor("D")
        add_end_time_delta = time_to_timedelta(end_time)
        if add_start_time_delta is not None:
            if add_end_time_delta < add_start_delta:
                add_end_time_delta += pd.Timedelta(days=1)

        if add_end_delta is None:
            add_end_delta = add_end_time_delta

        else:
            add_end_delta += add_end_time_delta

    if add_start_delta is not None:
        start += to_freq(add_start_delta)

    if add_end_delta is not None:
        end += to_freq(add_end_delta)

    if kind.lower() == "bounds":
        range_starts, range_ends = CustomResampler.map_bounds_to_source_ranges(
            source_index=naive_index.values,
            target_lbound_index=start.values,
            target_rbound_index=end.values,
            closed_lbound=closed_start,
            closed_rbound=closed_end,
            skip_not_found=skip_not_found,
        )

    else:
        if kind.lower() == "labels":

            range_starts = np.empty(len(start), dtype=np.int_)
            range_ends = np.empty(len(end), dtype=np.int_)
            range_index = pd.Series(
                np.arange(len(naive_index)), index=naive_index)

            for i in range(len(range_starts)):
                selected_range = range_index[start[i]: end[i]]
                if len(selected_range) > 0 and not closed_start and selected_range.index[0] == start[i]:
                    selected_range = selected_range.iloc[1:]

                if len(selected_range) > 0 and not closed_end and selected_range.index[-1] == end[i]:
                    selected_range = selected_range.iloc[:-1]

                if len(selected_range) > 0:
                    range_starts[i] = selected_range.iloc[0]
                    range_ends[i] = selected_range.iloc[-1]

                else:
                    range_starts[i] = -1
                    range_ends[i] = -1

        else:
            if not closed_start:
                start = start + 1

            if closed_end:
                end = end + 1

            range_starts = np.asarray(start)
            range_ends = np.asarray(end)

        if skip_not_found:
            valid_mask = (range_starts != -1) & (range_ends != -1)
            range_starts = range_starts[valid_mask]
            range_ends = range_ends[valid_mask]

    if np.any(range_starts >= range_ends):
        raise ValueError(
            "Some start indices are equal to or higher than end indices"
        )

    return range_starts, range_ends


class IntervalSplit(BaseModel):
    """
    Split index by intervals.

    Parameters
    ----------
    index : range
        The range of the index to be split.
    index_freq : str, int, float, Offset, or pd.Timedelta, optional
        Frequency of the index, e.g., 'D' for daily, 'M' for monthly.
    every : str, int, float, Offset, or pd.Timedelta, optional
        Frequency either as an integer or timedelta.
        Gets translated into `start` and `end` arrays by creating a range. If
        integer, an index sequence from `start` to `end` (exclusive) is created 
        and 'indices' as `kind` is used. If timedelta-like, a date sequence 
        from `start` to `end` (inclusive) is created and 'bounds' as `kind` is 
        used. If `start_time` and `end_time` are not None and `every`, `start`, 
        and `end` are None, `every` defaults to one day.
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
    lookback_period : str, int, float, Offset, or pd.Timedelta, optional
        Lookback period either as an integer or offset. If `lookback_period` is 
        set, `start` becomes `end-lookback_period`. If `every` is not None, the
        sequence is generated from `start+lookback_period` to `end` and then 
        assigned to `end`. If string, gets converted into an offset using 
        pandas. If integer, gets multiplied by the frequency of the index if 
        the index is not integer.
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
    fixed_start : bool, optional
        Whether all indices in the `start` array should be exactly `start`. 
        Works only together with `every`. Cannot be used together with 
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
    split : int, float, slice, BaseTool, or BasePeriod, optional
        Split criteria. Default is None.
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

    Example
    -------
    Example 1: Split every half a year
    ```pycon
    >>> model = IntervalSplit(
    ...     index,
    ...     every="YS",
    ...     closed_end=True,
    ...     split=0.5,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2011-01-01 2011-07-03
          OOS    2011-07-03 2012-01-02
    1     IS     2012-01-01 2012-07-02
          OOS    2012-07-02 2013-01-02
    2     IS     2013-01-01 2013-07-03
          OOS    2013-07-03 2014-01-02
    3     IS     2014-01-01 2014-07-03
          OOS    2014-07-03 2015-01-02
    4     IS     2015-01-01 2015-07-03
          OOS    2015-07-03 2016-01-02
    5     IS     2016-01-01 2016-07-02
          OOS    2016-07-02 2017-01-02
    6     IS     2017-01-01 2017-07-03
          OOS    2017-07-03 2018-01-02
    7     IS     2018-01-01 2018-07-03
          OOS    2018-07-03 2019-01-02
    8     IS     2019-01-01 2019-07-03
          OOS    2019-07-03 2020-01-02
    9     IS     2020-01-01 2020-07-02
          OOS    2020-07-02 2021-01-02
    10    IS     2021-01-01 2021-07-03
          OOS    2021-07-03 2022-01-02
    11    IS     2022-01-01 2022-07-03
          OOS    2022-07-03 2023-01-02
    12    IS     2023-01-01 2023-07-03
          OOS    2023-07-03 2024-01-02
    ```

    Example 2: Split by Calendar years
    ```pycon
    from fold import Lambda
    >>> n_train, n_test = 3, 1
    >>> total = n_train + n_test
    >>> model = IntervalSplit(
    ...     index,
    ...     fixed_start=False,
    ...     ndex_freq='M',
    ...     every="YS",
    ...     normalize_every=True,
    ...     lookback_period=f"{total}AS",
    ...     split=(
    ...         Lambda(f"index.year != index.year[{-n_test}]"),
    ...         Lambda(f"index.year = index.year[{-n_test}]"),
    ...     ),
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2010-10-06 2013-01-01
          OOS    2013-01-01 2014-01-01
    1     IS     2011-01-01 2014-01-01
          OOS    2014-01-01 2015-01-01
    2     IS     2012-01-01 2015-01-01
          OOS    2015-01-01 2016-01-01
    3     IS     2013-01-01 2016-01-01
          OOS    2016-01-01 2017-01-01
    4     IS     2014-01-01 2017-01-01
          OOS    2017-01-01 2018-01-01
    5     IS     2015-01-01 2018-01-01
          OOS    2018-01-01 2019-01-01
    6     IS     2016-01-01 2019-01-01
          OOS    2019-01-01 2020-01-01
    7     IS     2017-01-01 2020-01-01
          OOS    2020-01-01 2021-01-01
    8     IS     2018-01-01 2021-01-01
          OOS    2021-01-01 2022-01-01
    9     IS     2019-01-01 2022-01-01
          OOS    2022-01-01 2023-01-01
    10    IS     2020-01-01 2023-01-01
          OOS    2023-01-01 2024-01-01
    ```

    Example 3: Calendar Quarters
    ```pycon
    >>> model = IntervalSplit(
    ...     index,
    ...     every="QS",
    ...     lookback_period=f"{n_train + 1}QS",  # + 1 testset quarter
    ...     split=(
    ...         Lambda(
                    f"index < index[0] + pd.offsets.QuarterBegin({n_train}, startingMonth=1)"
    ...     ),
    ...         Lambda(
    ...             f"index >= index[0] + pd.offsets.QuarterBegin({n_train}, startingMonth=1)"
    ...         ),
    ...     ),
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2010-10-06 2011-07-01
          OOS    2011-07-01 2011-10-01
    1     IS     2011-01-01 2011-10-01
          OOS    2011-10-01 2012-01-01
    2     IS     2011-04-01 2012-01-01
                    ...        ...
    48    OOS    2023-07-01 2023-10-01
    49    IS     2023-01-01 2023-10-01
          OOS    2023-10-01 2024-01-01
    50    IS     2023-04-01 2024-01-01
          OOS    2024-01-01 2024-04-01
    ```
    """

    def __init__(
        self,
        index: range,
        index_freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        every: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        normalize_every: tp.Optional[bool] = False,
        split_every: tp.Optional[bool] = True,
        start_time: tp.Optional[str | time] = None,
        end_time: tp.Optional[str | time] = None,
        lookback_period: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        start: tp.Optional[str | int | float | Offset | pd.Timestamp] = None,
        end: tp.Optional[str | int | float | Offset | pd.Timestamp] = None,
        exact_start: tp.Optional[bool] = False,
        fixed_start: tp.Optional[bool] = False,
        closed_start: tp.Optional[bool] = True,
        closed_end: tp.Optional[bool] = False,
        add_start_delta: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        add_end_delta: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        kind: tp.Optional[str] = None,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
    ):

        index = prepare_dt_index(index)
        
        start_idxs, stop_idxs = get_index_ranges(
            index,
            skip_not_found=True,
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
        )
        splits = []
        for start, stop in zip(start_idxs, stop_idxs):
            new_split = slice(start, stop)
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

