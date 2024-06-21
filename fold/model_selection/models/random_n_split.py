import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, RelativePeriod, SplitPeriod
from fold.utils.datetime import (
    prepare_dt_index,
    try_align_dt_to_index,
    to_freq,
    infer_index_freq
)


class RandomNumberSplit(BaseModel):
    """
    Split time-series data at random intervals.

    This class randomly selects the length and start position of ranges within 
    a specified time-series index, and splits these ranges according to given 
    parameters. The length and start position are chosen using specified 
    functions, which can be customized.

    Parameters
    ----------
    index : range
        The time-series index to be split.
    n : int
        Number of random ranges to generate.
    min_length : str, int, float, or pd.Timedelta
        Minimum length of each range.
    max_length : str, int, float, or pd.Timedelta, optional
        Maximum length of each range. If None, defaults to `min_length`.
    min_start : str, int, float, or pd.Timedelta, optional
        Minimum start position of each range. If None, defaults to 0.
    max_end : str, int, float, or pd.Timedelta, optional
        Maximum end position of each range. If None, defaults to the length of 
        the index.
    length_choice_func : callable, optional
        Function to choose the length of each range. Defaults to 
        `numpy.random.Generator.choice`.
    start_choice_func : callable, optional
        Function to choose the start position of each range. Defaults to 
        `numpy.random.Generator.choice`.
    length_p_func : callable, optional
        Function to provide probabilities for `length_choice_func`. Must return 
        either None or an array of probabilities.
    start_p_func : callable, optional
        Function to provide probabilities for `start_choice_func`. Must return 
        either None or an array of probabilities.
    seed : int, optional
        Random seed for reproducibility.
    split : int, float, slice, BaseTool, or BasePeriod, optional
        Split parameter for dividing the ranges.
    allow_zero_len : bool, optional
        Whether to allow zero-length splits.
    range_format : str, optional
        Format of the ranges.
    freq : str, int, float, Offset, or pd.Timedelta, optional
        Frequency of the time-series. Defaults to "auto".
    constraints : BaseTool, optional
        Constraints for the splits.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the splits.
    sample_labels : range, optional
        Labels for the samples.

    Notes
    -----
    Each of the functions: 
    * `length_choice_func`
    * `start_choice_func`
    * `length_p_func`
    * `start_p_func`

    must accept two arguments: the iteration index and an array of possible 
    values.

    Examples
    --------
    Generate 20 random ranges with a length of 360 days and split each into 
    halves:

    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> model = RandomNumberSplit(
    ...     index,
    ...     50,
    ...     min_length=360,
    ...     split=0.5,
    ...     sample_labels=["train", "test"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2011-11-21 2012-05-19
          OOS    2012-05-19 2012-11-15
    1     IS     2020-08-02 2021-01-29
          OOS    2021-01-29 2021-07-28
    2     IS     2019-01-26 2019-07-25
                    ...        ...
    47    OOS    2022-08-05 2023-02-01
    48    IS     2019-05-15 2019-11-11
          OOS    2019-11-11 2020-05-09
    49    IS     2020-08-23 2021-02-19
          OOS    2021-02-19 2021-08-18
    """

    def __init__(
        self,
        index: range,
        n: int,
        min_length: str | int | float | pd.Timedelta,
        max_length: tp.Optional[str | int | float | pd.Timedelta] = None,
        min_start: tp.Optional[str | int | float | pd.Timedelta] = None,
        max_end: tp.Optional[str | int | float | pd.Timedelta] = None,
        length_choice_func: tp.Optional[tp.Callable] = None,
        start_choice_func: tp.Optional[tp.Callable] = None,
        length_p_func: tp.Optional[tp.Callable] = None,
        start_p_func: tp.Optional[tp.Callable] = None,
        seed: tp.Optional[int] = None,
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

        if min_start is None:
            min_start = 0

        if min_start is not None:
            if isinstance(min_start, (float, np.floating)):
                if 0 <= abs(min_start) <= 1:
                    min_start = len(index) * min_start
                elif not min_start.is_integer():
                    raise TypeError(
                        "Floating number for minimum start must be between 0 and 1"
                    )

            if isinstance(min_start, (float, np.floating)):
                min_start = int(min_start)

            if isinstance(min_start, (int, np.integer)) and not isinstance(min_start, np.timedelta64):
                if min_start < 0 or min_start > len(index) - 1:
                    raise TypeError(
                        f"Minimum start must be within [{0}, {len(index) - 1}]"
                    )

            else:
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(
                        f"Index must be of type pandas.DatetimeIndex, not {index.dtype}"
                    )

                min_start = try_align_dt_to_index(min_start, index)

                if not isinstance(min_start, pd.Timestamp):
                    raise ValueError(
                        f"Minimum start ({min_start}) could not be parsed"
                    )

                if min_start < index[0] or min_start > index[-1]:
                    raise TypeError(
                        f"Minimum start must be within [{index[0]}, {index[-1]}]"
                    )
                min_start = index.get_indexer([min_start], method="bfill")[0]

        if max_end is None:
            max_end = len(index)

        if isinstance(max_end, (float, np.floating)):
            if 0 <= abs(max_end) <= 1:
                max_end = len(index) * max_end
            elif not max_end.is_integer():
                raise TypeError(
                    "Floating number for maximum end must be between 0 and 1"
                )

        if isinstance(max_end, (float, np.floating)):
            max_end = int(max_end)

        if isinstance(max_end, (int, np.integer)) and not isinstance(max_end, np.timedelta64):
            if max_end < 1 or max_end > len(index):
                raise TypeError(
                    f"Maximum end must be within [{1}, {len(index)}]"
                )

        else:
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError(
                    f"Index must be of type pandas.DatetimeIndex, not {index.dtype}"
                )

            max_end = try_align_dt_to_index(max_end, index)
            if not isinstance(max_end, pd.Timestamp):
                raise ValueError(
                    f"Maximum end ({max_end}) could not be parsed"
                )

            if freq is None:
                raise ValueError("Must provide freq")
            if max_end < index[0] + freq or max_end > index[-1] + freq:
                raise TypeError(
                    f"Maximum end must be within [{index[0] + freq}, {index[-1] + freq}]"
                )

            if max_end > index[-1]:
                max_end = len(index)
            else:
                max_end = index.get_indexer([max_end], method="bfill")[0]

        space_len = max_end - min_start
        if (
                not isinstance(min_length, (float, np.floating))
                or isinstance(min_length, (int, np.integer))
                and not isinstance(min_length, np.timedelta64)
        ):
            index_min_start = index[min_start]
            if max_end < len(index):
                index_max_end = index[max_end]

            else:
                if freq is None:
                    raise ValueError("Must provide freq")
                index_max_end = index[-1] + freq
            index_space_len = index_max_end - index_min_start
        else:
            index_min_start = None
            index_max_end = None
            index_space_len = None

        if isinstance(min_length, (float, np.floating)):
            if 0 <= abs(min_length) <= 1:
                min_length = space_len * min_length
            elif not min_length.is_integer():
                raise TypeError(
                    "Floating number for minimum length must be between 0 and 1"
                )
            min_length = int(min_length)

        if isinstance(min_length, (int, np.integer)) and not isinstance(min_length, np.timedelta64):
            if min_length < 1 or min_length > space_len:
                raise TypeError(
                    f"Minimum length must be within [{1}, {space_len}]")
        else:
            min_length = to_freq(min_length)
            if freq is None:
                raise ValueError(
                    "Must provide freq"
                )
            if min_length < freq or min_length > index_space_len:
                raise TypeError(
                    f"Minimum length must be within [{freq}, {index_space_len}]")

        if max_length is not None:
            if isinstance(max_length, (float, np.floating)):
                if 0 <= abs(max_length) <= 1:
                    max_length = space_len * max_length
                elif not max_length.is_integer():
                    raise TypeError(
                        "Floating number for maximum length must be between 0 and 1")
                max_length = int(max_length)

            if isinstance(max_length, (int, np.integer)) and not isinstance(max_length, np.timedelta64):
                if max_length < min_length or max_length > space_len:
                    raise TypeError(
                        f"Maximum length must be within [{min_length}, {space_len}]")
            else:
                max_length = to_freq(max_length)
                if freq is None:
                    raise ValueError("Must provide freq")
                if max_length < min_length or max_length > index_space_len:
                    raise TypeError(
                        f"Maximum length must be within [{min_length}, {index_space_len}]")
        else:
            max_length = min_length

        rng = np.random.default_rng(seed=seed)
        if length_p_func is None:
            def length_p_func(i, x):
                return None

        if start_p_func is None:
            def start_p_func(i, x):
                return None

        if length_choice_func is None:
            def length_choice_func(i, x):
                return rng.choice(x, p=length_p_func(i, x))

        else:
            if seed is not None:
                np.random.seed(seed)

        if start_choice_func is None:
            def start_choice_func(i, x):
                return rng.choice(x, p=start_p_func(i, x))

        else:
            if seed is not None:
                np.random.seed(seed)

        if isinstance(min_length, (int, np.integer)) and not isinstance(min_length, np.timedelta64):
            length_space = np.arange(min_length, max_length + 1)
        else:
            if freq is None:
                raise ValueError("Must provide freq")

            length_space = np.arange(
                min_length // freq, max_length // freq + 1) * freq

        index_space = np.arange(len(index))

        splits = []
        for i in range(n):

            length = length_choice_func(i, length_space)

            if isinstance(length, (int, np.integer)) and not isinstance(length, np.timedelta64):
                start = start_choice_func(
                    i, index_space[min_start: max_end - length + 1])

            else:
                from_dt = index_min_start.to_datetime64()
                to_dt = index_max_end.to_datetime64() - length
                start = start_choice_func(
                    i, index_space[(index.values >= from_dt) & (index.values <= to_dt)])

            model = RelativePeriod(offset=start, length=length)
            new_split = model.to_slice(len(index), index=index, freq=freq)
            if split is not None:
                model = SplitPeriod(
                    period=new_split,
                    index=index,
                    allow_zero_len=allow_zero_len,
                    range_format=range_format,
                    freq=freq
                )
                new_split = model.split(split)

            splits.append(new_split)

        super().__init__(
            index,
            splits,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            backwards=backwards,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )
