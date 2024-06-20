import numpy as np
import pandas as pd
from numba import jit
import typing as tp

from fold.utils.datetime import to_freq, prepare_dt_index
from fold.utils import checks


def to_any_index(index_like: range) -> pd.Index:
    """
    Convert any index-like object to an index.

    Index objects are kept as-is.

    Parameters
    ----------
    index_like : range
        Index-like object to convert.

    Returns
    -------
    pd.Index
        Converted index.
    """
    if checks.is_np_array(index_like) and index_like.ndim == 0:
        index_like = index_like[None]

    if not checks.is_index(index_like):
        return pd.Index(index_like)

    return index_like


def repeat_index(
    index: range,
    n: int,
    ignore_ranges: tp.Optional[bool] = True,
) -> pd.Index:
    """
    Repeat each element in `index` `n` times.

    Parameters
    ----------
    index : range
        Index to repeat.
    n : int
        Number of times to repeat each element.
    ignore_ranges : tp.Optional[bool], optional
        If True, ignore indexes of type `pd.RangeIndex`, by default True.

    Returns
    -------
    pd.Index
        Repeated index.
    """
    index = to_any_index(index)
    if n == 1:
        return index

    if checks.is_default_index(index) and ignore_ranges:
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)

    return index.repeat(n)


def tile_index(
    index: range,
    n: int,
    ignore_ranges: tp.Optional[bool] = True
) -> pd.Index:
    """
    Tile the whole `index` `n` times.

    Parameters
    ----------
    index : range
        Index to tile.
    n : int
        Number of times to tile the index.

    Returns
    -------
    pd.Index
        Tiled index.
    """
    index = to_any_index(index)
    if n == 1:
        return index

    if checks.is_default_index(index) and ignore_ranges:
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)

    return pd.Index(np.tile(index, n), name=index.name)


def stack_index(*indexes: tp.Union[range, tp.Tuple[range, ...]]) -> pd.Index:
    """
    Stack each index in `indexes` on top of each other, from top to bottom.

    Parameters
    ----------
    *indexes : tp.Union[range, tp.Tuple[range, ...]]
        Indexes to stack.

    Returns
    -------
    pd.Index
        Stacked index.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    levels = []
    for i in range(len(indexes)):
        index = indexes[i]
        if not isinstance(index, pd.MultiIndex):
            levels.append(to_any_index(index))
        else:
            for j in range(index.nlevels):
                levels.append(index.get_level_values(j))

    max_len = max(map(len, levels))
    for i in range(len(levels)):
        if len(levels[i]) < max_len:
            if len(levels[i]) != 1:
                raise ValueError(
                    f"Index at level {i} could not be broadcast to "
                    f"shape ({max_len},)."
                )
            levels[i] = repeat_index(levels[i], max_len, ignore_ranges=False)
    new_index = pd.MultiIndex.from_arrays(levels)
    return to_any_index(new_index)


def combine_index(*indexes: tp.Union[range, tp.Tuple[range, ...]]) -> pd.Index:
    """
    Combine each index in `indexes` using Cartesian product.

    Parameters
    ----------
    *indexes : tp.Union[range, tp.Tuple[range, ...]]
        Indexes to combine.

    Returns
    -------
    pd.Index
        Combined index.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    new_index = to_any_index(indexes[0])
    for i in range(1, len(indexes)):
        index1, index2 = new_index, to_any_index(indexes[i])
        new_index1 = repeat_index(index1, len(index2), ignore_ranges=False)
        new_index2 = tile_index(index2, len(index1))
        new_index = stack_index([new_index1, new_index2])
    return new_index


def to_context_period(index: range, period: tp.Any, split: tp.Any):
    """
    Substitutes values in `split` based on the given `index` and `period`.

    Parameters
    ----------
    index : range
        The range of indices to be used.
    period : Any
        The period within the index to consider.
    split : Any
        The value to be substituted within the context of the specified period.

    Returns
    -------
    Any
        The result of substituting `split` with the context.

    Notes
    -----
    This function utilizes the `substitute` function from `fold.tools`.

    """
    from fold.tools import substitute
    context = dict(index=index[period])
    return substitute(split, context, eval_id="new_split")


@jit(cache=True)
def handle_gaps(period: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Splits a range with gaps into start and end indices.

    Parameters
    ----------
    period : np.ndarray
        Array of periods with possible gaps.

    Returns
    -------
    tuple of np.ndarray
        Two arrays: start indices and end indices of contiguous ranges.

    Raises
    ------
    ValueError
        If the input `period` array is empty.

    """
    if len(period) == 0:
        raise ValueError("Range is empty")

    start_idxs_out = np.empty(len(period), dtype=np.int_)
    stop_idxs_out = np.empty(len(period), dtype=np.int_)
    start_idxs_out[0] = 0
    k = 0
    for i in range(1, len(period)):
        if period[i] - period[i - 1] != 1:
            stop_idxs_out[k] = i
            k += 1
            start_idxs_out[k] = i
    stop_idxs_out[k] = len(period)
    return start_idxs_out[:k + 1], stop_idxs_out[:k + 1]


def to_gap_period(index: range, period: tp.Any) -> tp.List[tp.Any]:
    """
    Converts a period with gaps into a list of slices.

    Parameters
    ----------
    index : range
        The range of indices.
    period : Any
        The period within the index to consider.

    Returns
    -------
    list of Any
        List of slices corresponding to contiguous ranges in the period.

    """
    range_arr = (
        period
        if isinstance(period, np.ndarray)
        and np.issubdtype(period.dtype, np.integer)
        else np.arange(len(index))[period]
    )
    start_idxs, stop_idxs = handle_gaps(range_arr)
    return list(
        map(lambda x: slice(x[0], x[1]), zip(start_idxs, stop_idxs))
    )


def to_number_period(split: tp.Any, backwards: bool):
    """
    Converts a split value into a numerical period tuple and determines 
    direction.

    Parameters
    ----------
    split : Any
        The split value to convert.
    backwards : bool
        Flag indicating the direction.

    Returns
    -------
    tuple of (Any, bool)
        A tuple containing the converted split and the direction flag.

    """
    if split < 0:
        backwards = not backwards
        split = abs(split)
    if not backwards:
        split = (split, 1.0)
    else:
        split = (1.0, split)

    return split, backwards


def to_datetime_period(split: tp.Any, backwards: bool):
    """
    Converts a split value into a datetime period tuple and determines 
    direction.

    Parameters
    ----------
    split : Any
        The split value to convert.
    backwards : bool
        Flag indicating the direction.

    Returns
    -------
    tuple of (Any, bool)
        A tuple containing the converted split and the direction flag.

    """
    split = to_freq(split)
    if split < pd.Timedelta(0):
        backwards = not backwards
        split = abs(split)
    if not backwards:
        split = (split, 1.0)
    else:
        split = (1.0, split)

    return split, backwards


def select_index(
    labels: pd.Index, 
    selection: tp.Optional[tp.Any] = None
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Get selected indices.

    Parameters
    ----------
    labels : pd.Index
        Index labels.
    selection : tp.Any, optional
        Selection to apply (default is None).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the group indices.

    """
    from fold.tools import SelectPosition, SelectLabel
    if isinstance(selection, SelectPosition):
        selection = selection.value
        kind = "positions"

    elif isinstance(selection, SelectLabel):
        selection = selection.value
        kind = "labels"

    else:
        kind = None

    if checks.is_hashable(selection):
        selection = [selection]

    # Create labels for selection
    containter = []
    for s in selection:
        if isinstance(s, SelectPosition):
            s = s.value
            kind = "positions"
        elif isinstance(s, SelectLabel):
            s = s.value
            kind = "labels"

        if checks.is_label_position(labels, s, kind):
            i = s
        else:
            i = labels.get_indexer([s])[0]
            if i == -1:
                raise ValueError(f"Split '{s}' not found")
        containter.append(i)
    return np.asarray(containter)


def create_split_labels(
    splits_arr: np.ndarray,
    removed_indices: tp.List[tp.Any],
    labels: tp.Optional[tp.Union[range, tp.Sequence[tp.Any]]] = None
) -> tp.Union[range, tp.Sequence[tp.Any]]:
    """
    Create split labels.

    Parameters
    ----------
    splits_arr : np.ndarray
        Array of splits.
    removed_indices : tp.List[tp.Any]
        List of removed indices.
    labels : tp.Optional[tp.Union[range, tp.Sequence[tp.Any]]], optional
        Labels for splits (default is None).

    Returns
    -------
    tp.Union[range, tp.Sequence[tp.Any]]
        The split labels.

    """
    if labels is None:
        labels = pd.RangeIndex(stop=splits_arr.shape[0], name="split")
    else:
        if not isinstance(labels, pd.Index):
            labels = pd.Index(labels, name="split")
        if len(removed_indices) > 0:
            labels = labels.delete(removed_indices)
    return prepare_dt_index(labels, parse_index=True)


def create_sample_labels(
    splits_arr: tp.Optional[np.ndarray] = None,
    labels: tp.Optional[tp.Union[range, tp.Sequence[tp.Any]]] = None
) -> tp.Union[range, tp.Sequence[tp.Any]]:
    """
    Create sample labels.

    Parameters
    ----------
    splits_arr : tp.Optional[np.ndarray]
        Array of splits.
    labels : tp.Optional[tp.Union[range, tp.Sequence[tp.Any]]], optional
        Labels for samples (default is None).

    Returns
    -------
    tp.Union[range, tp.Sequence[tp.Any]]
        The sample labels.

    """
    if labels is None:
        return pd.Index(
            ["sample_%d" % i for i in range(splits_arr.shape[1])],
            name="sample"
        )
    if not isinstance(labels, pd.Index):
        return pd.Index(labels, name="sample")
    else:
        return labels
