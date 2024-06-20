from collections.abc import Hashable
from inspect import getmro
import datetime
import numpy as np
import pandas as pd
import typing as tp

if tp.TYPE_CHECKING:
    from fold.tools import (
        BaseTool,
        FixedPeriod,
        RelativePeriod,
        SelectPosition,
        SelectLabel
    )
else:
    BaseTool = "BaseTool"
    FixedPeriod = "FixedPeriod"
    RelativePeriod = "RelativePeriod"
    SelectPosition = "SelectPosition"
    SelectLabel = "SelectLabel"


def is_label_position(labels: pd.Index, selection: tp.Any, kind: str):
    """
    Determine if the selection is a label position based on the kind and the 
    labels provided.

    This function checks if the selection corresponds to a label position by 
    considering the type of selection and the nature of the labels. It helps 
    differentiate between position-based and label-based selections.

    Parameters
    ----------
    labels : pd.Index
        The index labels of the data. This can be any pandas Index object.
    selection : tp.Any
        The selection to be checked. This can be an integer, a label, or any 
        other type that may represent a position or a label.
    kind : str
        The kind of selection, indicating whether it is based on positions or 
        labels. Supported values are:
        - 'positions': Indicates that the selection is position-based.
        - None: Indicates that the selection type is to be inferred.

    Returns
    -------
    bool
        True if the selection is considered a label position, False otherwise.

    """
    return (
        kind in "positions" or kind in None and is_int(selection) and
        not pd.api.types.is_integer_dtype(labels)
    )


def is_range_relative(arg: tp.Any) -> bool:
    """
    Check if a period is relative.

    Parameters
    ----------
    arg : RangeLike
        The period to check.

    Returns
    -------
    bool
        True if the period is relative, False otherwise.

    """
    from fold.tools import RelativePeriod
    return is_number(arg) or is_td_like(arg) or isinstance(arg, RelativePeriod)


def is_range(arr: np.ndarray) -> bool:
    """
    Check if an array is a period.

    Parameters
    ----------
    arr : np.array
        The array to check.

    Returns
    -------
    bool
        True if the array is a period, False otherwise.

    """
    return np.all(np.diff(arr) == 1)


def is_int(arg: tp.Any) -> bool:
    """
    Check if an argument is an integer (excluding timedelta).

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is an integer, False otherwise.

    """
    return (
        isinstance(arg, (int, np.integer)) and not
        isinstance(arg, np.timedelta64)
    )


def is_float(arg: tp.Any) -> bool:
    """
    Check if an argument is a float.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a float, False otherwise.

    """
    return isinstance(arg, (float, np.floating))


def is_number(arg: tp.Any) -> bool:
    """
    Check if an argument is a number.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a number, False otherwise.

    """
    return is_int(arg) or is_float(arg)


def is_td(arg: tp.Any) -> bool:
    """
    Check if an argument is a timedelta.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a timedelta, False otherwise.

    """
    return isinstance(arg, (pd.Timedelta, datetime.timedelta, np.timedelta64))


def is_td_like(arg: tp.Any) -> bool:
    """
    Check if an argument is timedelta-like.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is timedelta-like, False otherwise.

    """
    return is_td(arg) or is_number(arg) or isinstance(arg, str)


def is_np_array(arg: tp.Any) -> bool:
    """
    Check if an argument is a NumPy array.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a NumPy array, False otherwise.

    """
    return isinstance(arg, np.ndarray)


def is_series(arg: tp.Any) -> bool:
    """
    Check if an argument is a Pandas Series.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a Pandas Series, False otherwise.

    """
    return isinstance(arg, pd.Series)


def is_index(arg: tp.Any) -> bool:
    """
    Check if an argument is a Pandas Index.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a Pandas Index, False otherwise.

    """
    return isinstance(arg, pd.Index)


def is_frame(arg: tp.Any) -> bool:
    """
    Check if an argument is a Pandas DataFrame.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a Pandas DataFrame, False otherwise.

    """
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg: tp.Any) -> bool:
    """
    Check if an argument is a Pandas object (Series, Index, or DataFrame).

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a Pandas object, False otherwise.

    """
    return is_series(arg) or is_index(arg) or is_frame(arg)


def is_any_array(arg: tp.Any) -> bool:
    """
    Check if an argument is a NumPy array or a Pandas object.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a NumPy array or a Pandas object, False 
        otherwise.

    """
    return is_pandas(arg) or isinstance(arg, np.ndarray)


def is_sequence(arg: tp.Any) -> bool:
    """
    Check if an argument is a sequence.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a sequence, False otherwise.

    """
    try:
        len(arg)
        arg[0:0]
        return True

    except (TypeError, KeyError):
        return False


def is_complex_sequence(arg: tp.Any) -> bool:
    """
    Check if an argument is a complex sequence (excluding string and bytes).

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is a complex sequence, False otherwise.

    """
    if isinstance(arg, (str, bytes, bytearray)):
        return False

    return is_sequence(arg)


def is_iterable(arg: tp.Any) -> bool:
    """
    Check if an argument is iterable.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is iterable, False otherwise.

    """
    try:
        _ = iter(arg)
        return True

    except TypeError:
        return False


def is_hashable(arg: tp.Any) -> bool:
    """
    Check if an argument is hashable.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.

    Returns
    -------
    bool
        True if the argument is hashable, False otherwise.

    """
    if not isinstance(arg, Hashable):
        return False

    try:
        hash(arg)
    except TypeError:
        return False

    return True


def is_index_equal(arg1: tp.Any, arg2: tp.Any, check_names: bool = True) -> bool:
    """
    Check if two indexes are equal.

    Parameters
    ----------
    arg1 : tp.Any
        The first index.
    arg2 : tp.Any
        The second index.
    check_names : bool, optional
        Whether to check names (default is True).

    Returns
    -------
    bool
        True if the indexes are equal, False otherwise.

    """
    if not check_names:
        return pd.Index.equals(arg1, arg2)

    if isinstance(arg1, pd.MultiIndex) and isinstance(arg2, pd.MultiIndex):
        if arg1.names != arg2.names:
            return False

    elif isinstance(arg1, pd.MultiIndex) or isinstance(arg2, pd.MultiIndex):
        return False

    else:
        if arg1.name != arg2.name:
            return False

    return pd.Index.equals(arg1, arg2)


def is_default_index(arg: tp.Any, check_names: bool = True) -> bool:
    """
    Check if an index is a default period index.

    Parameters
    ----------
    arg : tp.Any
        The index to check.
    check_names : bool, optional
        Whether to check names (default is True).

    Returns
    -------
    bool
        True if the index is a default period index, False otherwise.

    """
    return is_index_equal(
        arg,
        pd.RangeIndex(start=0, stop=len(arg), step=1),
        check_names=check_names
    )


def is_namedtuple(arg: tp.Any) -> bool:
    """
    Check if an object is a namedtuple.

    Parameters
    ----------
    arg : tp.Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a namedtuple, False otherwise.

    """
    if not isinstance(arg, type):
        arg = type(arg)
    bases = arg.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(arg, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(field) == str for field in fields)


def is_subclass_of(arg: tp.Any, types: tp.Any) -> bool:
    """
    Check if an argument is a subclass of specified types.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.
    types : tp.Any
        The types to check against.

    Returns
    -------
    bool
        True if the argument is a subclass of the specified types, False 
        otherwise.

    """
    if isinstance(types, type):
        return issubclass(arg, types)

    if isinstance(types, str):
        for base_t in getmro(arg):
            if str(base_t) == types or base_t.__name__ == types:
                return True

    if isinstance(types, tuple):
        for t in types:
            if is_subclass_of(arg, t):
                return True

    return False


def is_instance_of(arg: tp.Any, types: tp.Any) -> bool:
    """
    Check if an argument is an instance of specified types.

    Parameters
    ----------
    arg : tp.Any
        The argument to check.
    types : tp.Any
        The types to check against.

    Returns
    -------
    bool
        True if the argument is an instance of the specified types, False 
        otherwise.

    """
    return is_subclass_of(type(arg), types)


def assert_bounds_is_valid(start, stop, index):
    """
    Assert that the start and stop bounds are valid within the given index.

    Parameters
    ----------
    start : int
        The start bound.
    stop : int
        The stop bound.
    index : pandas.Index
        The index to check bounds against.

    Raises
    ------
    ValueError
        If the start is greater than the stop.
        If the start or stop is out of bounds of the index.
    """
    if start != stop:
        assert_bound_is_valid(start, stop)
        assert_start_not_out_of_bounds(start, index)
        assert_stop_not_out_of_bounds(stop, index)


def assert_range_format_is_valid(range_format):
    """
    Assert that the range format is valid.

    Parameters
    ----------
    range_format : str
        The range format to check.

    Raises
    ------
    ValueError
        If the range format is not one of the accepted options.
    """
    if range_format.lower() not in (
        "any",
        "indices",
        "mask",
        "slice",
        "slice_or_indices",
        "slice_or_mask",
        "slice_or_any",
    ):
        raise ValueError(f"Invalid option range_format='{range_format}'")


def assert_slice_range_not_constant(range_format):
    """
    Assert that the slice range is not constant.

    Parameters
    ----------
    range_format : str
        The range format to check.

    Raises
    ------
    ValueError
        If the range format is 'slice'.
    """
    if range_format.lower() == "slice":
        raise ValueError(
            "Cannot convert to slice: range is not constant")


def assert_range_values_is_valid(period):
    """
    Assert that the period values are valid.

    Parameters
    ----------
    period : array-like
        The period values to check.

    Raises
    ------
    ValueError
        If the period contains a value of -1.
    """
    if -1 in period:
        raise ValueError(
            "Range array has values that cannot be found in index"
        )


def assert_bound_is_valid(start, stop):
    """
    Assert that the start and stop bounds are valid.

    Parameters
    ----------
    start : int
        The start bound.
    stop : int
        The stop bound.

    Raises
    ------
    ValueError
        If the start is greater than the stop.
    """
    if start > stop:
        raise ValueError(
            f"Range start ({start}) is higher than range stop ({stop})")


def assert_start_not_out_of_bounds(start, index):
    """
    Assert that the start bound is not out of bounds of the index.

    Parameters
    ----------
    start : int
        The start bound.
    index : pandas.Index
        The index to check against.

    Raises
    ------
    ValueError
        If the start is out of bounds of the index.
    """
    if start < 0 or start >= len(index):
        raise ValueError(f"Range start ({start}) is out of bounds")


def assert_stop_not_out_of_bounds(stop, index):
    """
    Assert that the stop bound is not out of bounds of the index.

    Parameters
    ----------
    stop : int
        The stop bound.
    index : pandas.Index
        The index to check against.

    Raises
    ------
    ValueError
        If the stop is out of bounds of the index.
    """
    if stop < 0 or stop > len(index):
        raise ValueError(f"Range stop ({stop}) is out of bounds")


def assert_same_length(period, index):
    """
    Assert that the period and index have the same length.

    Parameters
    ----------
    period : array-like
        The period values to check.
    index : pandas.Index
        The index to check against.

    Raises
    ------
    ValueError
        If the period and index do not have the same length.
    """
    if len(period) != len(index):
        raise ValueError("Mask must have the same length as index")


def assert_zero_len(length, allow_zero_len):
    """
    Assert that the length is zero if allowed.

    Parameters
    ----------
    length : int
        The length to check.
    allow_zero_len : bool
        Whether zero length is allowed.

    Raises
    ------
    ValueError
        If the length is zero and zero length is not allowed.
    """
    if not allow_zero_len and length == 0:
        raise ValueError("Range has zero length")


def assert_slice_is_valid(start: int, stop: int):
    """
    Assert that the slice defined by start and stop is valid.

    Parameters
    ----------
    start : int
        The start bound of the slice.
    stop : int
        The stop bound of the slice.

    Raises
    ------
    ValueError
        If the slice is not strictly negative or positive.
    """
    if start is not None and is_int(start) and start < 0:
        if stop is not None and is_int(stop) and stop > 0:
            raise ValueError(
                "Slices must be either strictly negative or positive"
            )


def assert_step_is_valid(step: int):
    """
    Assert that the step is valid.

    Parameters
    ----------
    step : int
        The step value to check.

    Raises
    ------
    ValueError
        If the step is greater than 1.
    """
    if step is not None and step > 1:
        raise ValueError("Step must be either None or 1")


def assert_dt_index(index: pd.DatetimeIndex):
    """
    Assert that the index is a pandas DatetimeIndex.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        The index to check.

    Raises
    ------
    TypeError
        If the index is not a pandas DatetimeIndex.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            "Index must be of type pandas.DatetimeIndex, not "
            f"{index.dtype}"
        )


def assert_can_be_parsed(bound: pd.Timestamp, name: str):
    """
    Assert that the bound can be parsed as a pandas Timestamp.

    Parameters
    ----------
    bound : pandas.Timestamp
        The bound to check.
    name : str
        The name of the bound.

    Raises
    ------
    ValueError
        If the bound cannot be parsed as a pandas Timestamp.
    """
    if not isinstance(bound, pd.Timestamp):
        raise ValueError(
            f"Range {name} ({bound}) could not be parsed"
        )


def assert_not_out_of_bounds(bound: int, name: str):
    """
    Assert that the bound is not out of bounds.

    Parameters
    ----------
    bound : int
        The bound to check.
    name : str
        The name of the bound.

    Raises
    ------
    ValueError
        If the bound is out of bounds.
    """
    if bound == -1:
        raise ValueError(
            f"Range {name} ({bound}) is out of bounds"
        )
