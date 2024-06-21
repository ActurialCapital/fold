import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.models.function_split import FunctionSplit
from fold.tools import Key, BaseTool, BasePeriod
from fold.utils.datetime import prepare_dt_index


def get_freq_offset(params: tuple):
    """
    Determine the offset based on parameters.

    Parameters
    ----------
    params : tuple
        A tuple containing the period and frequency.

    Returns
    -------
    pd.DateOffset
        A DateOffset based on the specified parameters.

    Raises
    ------
    ValueError
        If the frequency is invalid.
    """
    period, freq = params

    if freq.startswith(('week', 'W', 'w')):
        func = pd.offsets.Week(period)

    elif freq.startswith(('month', 'M', 'm')):
        func = pd.offsets.MonthBegin(period)

    elif freq.startswith(('quarter', 'Q', 'q')):
        func = pd.offsets.QuarterBegin(period, startingMonth=1)

    elif freq.startswith(('year', 'Y', 'y')):
        func = pd.DateOffset(years=period)

    else:
        raise NotImplementedError(
            'Frequency `{freq}` has not been implemented. Please check parameters `train` or `test`.'
        )

    return func


class PeriodSplit(FunctionSplit):
    """
    Split index with a custom split function.
    
    This class is useful for creating training and testing splits for 
    time series data, where each split encompasses a certain number of 
    periods with a specific frequency. Each training and testing period can
    be injected with a different frequency.

    Notes
    -----
    This method is a specific instance of the custom FunctionSplit method

    Parameters
    ----------
    index : range
        The range of the index to be split.
    n_train : Tuple[int, str]
        Number of training periods. Example `n_train=(3, 'Y')` is 3 years 
        training set.
    n_test : Tuple[int, str]
        Number of testing periods. Example `n_test=(1, 'Y')` is 1 year testing
        set.
    split_func : Callable
        The custom function to perform the split.
    split_args : Tuple[Any, ...], optional
        Arguments to pass to the split function, by default None.
    split_kwargs : Dict[str, Any], optional
        Keyword arguments to pass to the split function, by default None.
    fix_ranges : bool, optional
        Whether to convert the split into a fixed split, by default True.
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

    Raises
    ------
    ValueError
        If the split function returns more than one range when `split` is 
        provided.
    ValueError
        If all splits do not have the same number of sets.

    Examples
    --------
    >>> model = PeriodSplit(
    ...     data.index,
    ...     n_train = (3, 'Y'),
    ...     n_test = (2, 'Q'),
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    # bound             start        end
    # split sample                      
    # 0     IS     2011-04-01 2014-04-01
    #       OOS    2014-04-01 2014-10-01
    # 1     IS     2011-10-01 2014-10-01
    #       OOS    2014-10-01 2015-04-01
    # 2     IS     2012-04-01 2015-04-01
    #       OOS    2015-04-01 2015-10-01
    # 3     IS     2012-10-01 2015-10-01
    #       OOS    2015-10-01 2016-04-01
    # 4     IS     2013-04-01 2016-04-01
    #       OOS    2016-04-01 2016-10-01
    # 5     IS     2013-10-01 2016-10-01
    #       OOS    2016-10-01 2017-04-01
    # 6     IS     2014-04-01 2017-04-01
    #       OOS    2017-04-01 2017-10-01
    # 7     IS     2014-10-01 2017-10-01
    #       OOS    2017-10-01 2018-04-01
    # 8     IS     2015-04-01 2018-04-01
    #       OOS    2018-04-01 2018-10-01
    # 9     IS     2015-10-01 2018-10-01
    #       OOS    2018-10-01 2019-04-01
    # 10    IS     2016-04-01 2019-04-01
    #       OOS    2019-04-01 2019-10-01
    # 11    IS     2016-10-01 2019-10-01
    #       OOS    2019-10-01 2020-04-01
    # 12    IS     2017-04-01 2020-04-01
    #       OOS    2020-04-01 2020-10-01
    # 13    IS     2017-10-01 2020-10-01
    #       OOS    2020-10-01 2021-04-01
    # 14    IS     2018-04-01 2021-04-01
    #       OOS    2021-04-01 2021-10-01
    # 15    IS     2018-10-01 2021-10-01
    #       OOS    2021-10-01 2022-04-01
    # 16    IS     2019-04-01 2022-04-01
    #       OOS    2022-04-01 2022-10-01
    # 17    IS     2019-10-01 2022-10-01
    #       OOS    2022-10-01 2023-04-01
    # 18    IS     2020-04-01 2023-04-01
    #       OOS    2023-04-01 2023-10-01
    # 19    IS     2020-10-01 2023-10-01
    #       OOS    2023-10-01 2024-04-01
    ```
    """

    def __init__(
        self,
        index: range,
        n_train: int,
        n_test: int,
        *,
        split_kwargs: tp.Dict[str, tp.Any] = None,
        fix_ranges: tp.Optional[bool] = True,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        right_inclusive: tp.Optional[bool] = False,
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
    ):
        index = prepare_dt_index(index)

        def split_func(index: pd.DatetimeIndex, prev_start: pd.Timestamp):
            """Define the split function for creating training and testing periods."""
            # If this is the first split, prev_start (i.e., the start index of the
            # previous split) will be None
            if prev_start is None:
                prev_start = index[0]
            # The start date of this window is the beginning of the next quarter
            # (inclusive) - For next month, use MonthBegin() instead.
            # Testset increment
            # pd.offsets.QuarterBegin(n_test)
            new_start = prev_start + get_freq_offset(n_test)
            # Total number of years to split
            # The end date of this window is the same date but in the n_train + n_test
            # year (exclusive)
            # pd.DateOffset(years=n_train + n_test)
            new_end = new_start + get_freq_offset(n_train) + get_freq_offset(n_test)
            # If the split is incomplete (i.e., the end date comes after the next
            # possible end date), abort!
            if new_end > index[-1] + index.freq:
                return None
            # Trainset increment
            # pd.DateOffset(years=n_train)
            offset_period = new_start + get_freq_offset(n_train)
            return [
                # Trainset increment
                # Allocate n_train freq for the IS period and n_test freq for
                # the OOS period
                slice(new_start, offset_period),
                slice(offset_period, new_end)
            ]

        super().__init__(
            index=index,
            split_func=split_func,
            split_args=(Key("index"), Key("prev_start")),
            index_bounds=True,
            split_kwargs=split_kwargs,
            fix_ranges=fix_ranges,
            split=split,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            right_inclusive=right_inclusive,
            constraints=constraints,
            backwards=backwards,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )
