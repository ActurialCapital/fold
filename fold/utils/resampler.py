import warnings
from functools import cached_property
from numba import jit
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset
import typing as tp

from fold.utils.datetime import prepare_dt_index, infer_index_freq, to_timedelta64
from fold.utils.indexing import repeat_index
from fold.utils import checks


class CustomResampler:
    """
    Class that exposes methods to resample index.

    Parameters:
    -----------
    source_index : range
        Index being resampled.
    target_index : range
        Index resulted from resampling.
    source_freq : tp.Union[bool, BaseOffset, pd.Timedelta], default=None
        Frequency or date offset of the source index.
        Set to False to force-set the frequency to None.
    target_freq : tp.Union[bool, BaseOffset, pd.Timedelta], default=None
        Frequency or date offset of the target index.
        Set to False to force-set the frequency to None.
    silence_warnings : Optional[bool], default=None
        Whether to silence all warnings.

    Attributes:
    -----------
    source_index : pd.Index
        Index being resampled.
    target_index : pd.Index
        Index resulted from resampling.
    source_freq : tp.Union[int, BaseOffset, pd.Timedelta]
        Frequency or date offset of the source index.
    target_freq : tp.Union[int, BaseOffset, pd.Timedelta]
        Frequency or date offset of the target index.
    """

    def __init__(
        self,
        source_index: range,
        target_index: range,
        source_freq: tp.Union[bool, BaseOffset, pd.Timedelta] = None,
        target_freq: tp.Union[bool, BaseOffset, pd.Timedelta] = None,
        silence_warnings: tp.Optional[bool] = True,
    ):
        source_index = prepare_dt_index(source_index)
        target_index = prepare_dt_index(target_index)
        
        infer_source_freq = True
        if isinstance(source_freq, bool):
            if not source_freq:
                infer_source_freq = False
            source_freq = None
        infer_target_freq = True

        if isinstance(target_freq, bool):
            if not target_freq:
                infer_target_freq = False
            target_freq = None

        if infer_source_freq:
            source_freq = infer_index_freq(source_index, freq=source_freq)

        if infer_target_freq:
            target_freq = infer_index_freq(target_index, freq=target_freq)

        self._source_index = source_index
        self._target_index = target_index
        self._source_freq = source_freq
        self._target_freq = target_freq
        self._silence_warnings = silence_warnings

    @property
    def source_index(self) -> pd.Index:
        """Index being resampled."""
        return self._source_index

    @property
    def target_index(self) -> pd.Index:
        """Index resulted from resampling."""
        return self._target_index

    @property
    def source_freq(self) -> tp.Union[int, BaseOffset, pd.Timedelta]:
        """Frequency or date offset of the source index."""
        return self._source_freq

    @property
    def target_freq(self) -> tp.Union[int, BaseOffset, pd.Timedelta]:
        """Frequency or date offset of the target index."""
        return self._target_freq

    def get_np_source_freq(
        self,
        silence_warnings: tp.Optional[bool] = None
    ) -> tp.Union[int, BaseOffset, pd.Timedelta]:
        """Frequency or date offset of the source index in NumPy format."""
        silence_warnings = silence_warnings or self._silence_warnings

        warned = False
        source_freq = self.source_freq
        if source_freq is not None:
            if not isinstance(source_freq, (int, float)):
                try:
                    source_freq = to_timedelta64(source_freq)
                except ValueError:
                    if not silence_warnings:
                        warnings.warn(
                            f"Cannot convert {source_freq} to np.timedelta64. "
                            "Setting to None.",
                            stacklevel=2
                        )
                        warned = True
                    source_freq = None
        if source_freq is None:
            if not warned and not silence_warnings:
                warnings.warn(
                    "Using right bound of source index without frequency. "
                    "Set source frequency.",
                    stacklevel=2
                )
        return source_freq

    def get_np_target_freq(
        self, 
        silence_warnings: tp.Optional[bool] = None
    ) -> tp.Union[int, BaseOffset, pd.Timedelta]:
        """Frequency or date offset of the target index in NumPy format."""
        silence_warnings = silence_warnings or self._silence_warnings

        warned = False
        target_freq = self.target_freq
        if target_freq is not None:
            if not isinstance(target_freq, (int, float)):
                try:
                    target_freq = to_timedelta64(target_freq)
                except ValueError:
                    if not silence_warnings:
                        warnings.warn(
                            f"Cannot convert {target_freq} to np.timedelta64. "
                            "Setting to None.",
                            stacklevel=2
                        )
                        warned = True
                    target_freq = None
        if target_freq is None:
            if not warned and not silence_warnings:
                warnings.warn(
                    "Using right bound of target index without frequency. "
                    "Set target frequency.",
                    stacklevel=2
                )
        return target_freq

    @classmethod
    def get_lbound_index(
        cls,
        index: pd.Index,
        freq: tp.Union[int, BaseOffset, pd.Timedelta] = None
    ) -> pd.Index:
        """
        Get the left bound of a datetime index.
        If `freq` is None, calculates the leftmost bound.
        """
        index = prepare_dt_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if freq is not None:
            return index.shift(-1, freq=freq) + pd.Timedelta(1, "ns")
        min_ts = pd.DatetimeIndex([pd.Timestamp.min.tz_localize(index.tz)])
        return (index[:-1] + pd.Timedelta(1, "ns")).append(min_ts)

    @classmethod
    def get_rbound_index(
        cls,
        index: pd.Index,
        freq: tp.Union[int, BaseOffset, pd.Timedelta] = None
    ) -> pd.Index:
        """
        Get the right bound of a datetime index.
        If `freq` is None, calculates the rightmost bound.
        """
        index = prepare_dt_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if freq is not None:
            return index.shift(1, freq=freq) - pd.Timedelta(1, "ns")
        max_ts = pd.DatetimeIndex([pd.Timestamp.max.tz_localize(index.tz)])
        return (index[1:] - pd.Timedelta(1, "ns")).append(max_ts)

    @cached_property
    def target_lbound_index(self) -> pd.Index:
        """Get the left bound of the target datetime index."""
        return self.get_lbound_index(self.target_index, freq=self.target_freq)

    @cached_property
    def target_rbound_index(self) -> pd.Index:
        """Get the right bound of the target datetime index."""
        return self.get_rbound_index(self.target_index, freq=self.target_freq)

    @staticmethod
    @jit(cache=True)
    def _map_bounds_to_source_ranges_nb(
        source_index: np.ndarray,
        target_lbound_index: np.ndarray,
        target_rbound_index: np.ndarray,
        closed_lbound: bool = True,
        closed_rbound: bool = False,
        skip_not_found: bool = False,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Get the source bounds that correspond to the target bounds.

        Returns a 2-dimensional array where the first column is the absolute start 
        index (inclusive) and the second column is the absolute end index 
        (exclusive).

        If an element cannot be mapped, the start and end of the range becomes -1.

        Parameters:
        -----------
        source_index : np.ndarray
            The source index to map from.
        target_lbound_index : np.ndarray
            The target left-bound index to map to.
        target_rbound_index : np.ndarray
            The target right-bound index to map to.
        closed_lbound : bool, default=True
            Whether the left-bound of the target is inclusive.
        closed_rbound : bool, default=False
            Whether the right-bound of the target is inclusive.
        skip_not_found : bool, default=False
            Whether to skip ranges that cannot be mapped.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays representing the mapped start and end indices of source 
            ranges.
        """

        range_starts_out = np.empty(len(target_lbound_index), dtype=np.int_)
        range_ends_out = np.empty(len(target_lbound_index), dtype=np.int_)
        k = 0

        to_j = 0
        for i in range(len(target_lbound_index)):
            if i > 0 and target_lbound_index[i] < target_lbound_index[i - 1]:
                raise ValueError("Target left-bound index must be increasing")
            if i > 0 and target_rbound_index[i] < target_rbound_index[i - 1]:
                raise ValueError("Target right-bound index must be increasing")

            from_j = -1
            for j in range(len(source_index)):
                if j > 0 and source_index[j] < source_index[j - 1]:
                    raise ValueError("Array index must be increasing")
                found = False
                if closed_lbound and closed_rbound:
                    if target_lbound_index[i] <= source_index[j] <= target_rbound_index[i]:
                        found = True
                    elif source_index[j] > target_rbound_index[i]:
                        break
                elif closed_lbound:
                    if target_lbound_index[i] <= source_index[j] < target_rbound_index[i]:
                        found = True
                    elif source_index[j] >= target_rbound_index[i]:
                        break
                elif closed_rbound:
                    if target_lbound_index[i] < source_index[j] <= target_rbound_index[i]:
                        found = True
                    elif source_index[j] > target_rbound_index[i]:
                        break
                else:
                    if target_lbound_index[i] < source_index[j] < target_rbound_index[i]:
                        found = True
                    elif source_index[j] >= target_rbound_index[i]:
                        break
                if found:
                    if from_j == -1:
                        from_j = j
                    to_j = j + 1

            if skip_not_found:
                if from_j != -1:
                    range_starts_out[k] = from_j
                    range_ends_out[k] = to_j
                    k += 1
            else:
                if from_j == -1:
                    range_starts_out[i] = -1
                    range_ends_out[i] = -1
                else:
                    range_starts_out[i] = from_j
                    range_ends_out[i] = to_j

        if skip_not_found:
            return range_starts_out[:k], range_ends_out[:k]
        return range_starts_out, range_ends_out

    @classmethod
    def map_bounds_to_source_ranges(
        cls,
        source_index: tp.Optional[range] = None,
        target_lbound_index: tp.Optional[range] = None,
        target_rbound_index: tp.Optional[range] = None,
        closed_lbound: bool = True,
        closed_rbound: bool = False,
        skip_not_found: bool = False,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Map target bounds to corresponding source ranges.

        Either `target_lbound_index` or `target_rbound_index` must be set.
        Set `target_lbound_index` and `target_rbound_index` to 'pandas' to use
        `Resampler.get_lbound_index` and `Resampler.get_rbound_index` 
        respectively.

        Also, both allow providing a single datetime string and will 
        automatically broadcast to the `Resampler.target_index`.

        Parameters:
        -----------
        source_index : Optional[IndexLike], default=None
            The source index for mapping.
        target_lbound_index : Optional[IndexLike], default=None
            The target left-bound index for mapping.
        target_rbound_index : Optional[IndexLike], default=None
            The target right-bound index for mapping.
        closed_lbound : bool, default=True
            Whether the left-bound of the target is inclusive.
        closed_rbound : bool, default=False
            Whether the right-bound of the target is inclusive.
        skip_not_found : bool, default=False
            Whether to skip ranges that cannot be mapped.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays representing the mapped start and end indices of source 
            ranges.
        """
        if not isinstance(cls, type):
            if target_lbound_index is None and target_rbound_index is None:
                raise ValueError(
                    "Either target_lbound_index or target_rbound_index "
                    "must be set"
                )
            if target_lbound_index is not None:
                if (
                    isinstance(target_lbound_index, str) and
                    target_lbound_index.lower() == "pandas"
                ):
                    target_lbound_index = cls.target_lbound_index
                else:
                    target_lbound_index = prepare_dt_index(target_lbound_index)
                target_rbound_index = cls.target_index

            if target_rbound_index is not None:
                target_lbound_index = cls.target_index
                if (
                    isinstance(target_rbound_index, str) and
                    target_rbound_index.lower() == "pandas"
                ):
                    target_rbound_index = cls.target_rbound_index
                else:
                    target_rbound_index = prepare_dt_index(target_rbound_index)

            if len(target_lbound_index) == 1 and len(target_rbound_index) > 1:
                target_lbound_index = repeat_index(
                    target_lbound_index,
                    len(target_rbound_index)
                )
            elif len(target_lbound_index) > 1 and len(target_rbound_index) == 1:
                target_rbound_index = repeat_index(
                    target_rbound_index,
                    len(target_lbound_index)
                )
        else:
            source_index = prepare_dt_index(source_index)
            target_lbound_index = prepare_dt_index(target_lbound_index)
            target_rbound_index = prepare_dt_index(target_rbound_index)

        if len(target_rbound_index) != len(target_lbound_index):
            raise AssertionError(
                "Lengths of 'target_rbound_index' and 'target_lbound_index' "
                "do not match"
            )

        func = cls._map_bounds_to_source_ranges_nb.py_func
        return func(
            source_index.values,
            target_lbound_index.values,
            target_rbound_index.values,
            closed_lbound=closed_lbound,
            closed_rbound=closed_rbound,
            skip_not_found=skip_not_found,
        )

    @staticmethod
    def to_series_array(arg: tp.Any) -> tp.Union[np.ndarray, pd.Index, pd.Series]:
        """
        Reshape argument to one dimension.

        Parameters
        ----------
        arg : tp.Any
            The array-like object to reshape.

        Returns
        -------
        Anynp.ndarray
            The reshaped 1-dimensional array-like object.

        Raises
        ------
        ValueError
            If unable to reshape the array-like object to 1 dimension.
        """
        arg = np.asarray(arg)
        if arg.ndim == 2:
            if arg.shape[1] == 1:
                if checks.is_frame(arg):
                    return arg.iloc[:, 0]
                return arg[:, 0]

        if arg.ndim == 1:
            return arg

        elif arg.ndim == 0:
            return arg.reshape((1,))

        raise ValueError(
            f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension"
        )

    @staticmethod
    def to_frame_array(arg: tp.Any, expand_axis: int = 1) -> tp.Union[np.ndarray, pd.DataFrame]:
        """
        Reshape argument to two dimensions.

        Parameters
        ----------
        arg : ArrayLike
            The array-like object to reshape.   
        expand_axis : int, optional
            The axis to expand if the input is 1-dimensional `0` for rows, `1` 
            for columns.

        Returns
        -------
        AnyArray2d
            The reshaped 2-dimensional array-like object.

        Raises
        ------
        ValueError
            If unable to reshape the array-like object to 2 dimensions.
        """
        arg = np.asarray(arg)
        if arg.ndim == 2:
            return arg

        elif arg.ndim == 1:
            if checks.is_series(arg):
                if expand_axis == 0:
                    return pd.DataFrame(arg.values[None, :], columns=arg.index)

                elif expand_axis == 1:
                    return arg.to_frame()
            return np.expand_dims(arg, expand_axis)

        elif arg.ndim == 0:
            return arg.reshape((1, 1))

        raise ValueError(
            f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions"
        )

    @staticmethod
    @jit(cache=True)
    def _resample_source_mask_nb(
        source_mask: np.ndarray,
        source_index: np.ndarray,
        target_index: np.ndarray,
        source_freq: tp.Optional[tp.Any] = None,
        target_freq: tp.Optional[tp.Any] = None,
    ) -> np.ndarray:
        """
        Resample a source mask to the target index.

        The resampled mask becomes True only if the target bar is fully contained 
        in the source bar. The source bar is represented by a non-interrupting 
        sequence of True values in the source mask.

        Parameters:
        -----------
        source_mask : np.ndarray
            The mask associated with the source index.
        source_index : np.ndarray
            The source index for resampling.
        target_index : np.ndarray
            The target index to resample the mask onto.
        source_freq : Optional[tp.Any], default=None
            The frequency or date offset of the source index.
        target_freq : Optional[tp.Any], default=None
            The frequency or date offset of the target index.

        Returns:
        --------
        np.ndarray
            The resampled mask aligned with the target index.
        """
        out = np.full(len(target_index), False, dtype=np.bool_)

        from_j = 0
        for i in range(len(target_index)):
            if i > 0 and target_index[i] < target_index[i - 1]:
                raise ValueError("Target index must be increasing")
            target_lbound = target_index[i]
            if target_freq is None:
                if i + 1 < len(target_index):
                    target_rbound = target_index[i + 1]
                else:
                    target_rbound = None
            else:
                target_rbound = target_index[i] + target_freq

            found_start = False
            for j in range(from_j, len(source_index)):
                if j > 0 and source_index[j] < source_index[j - 1]:
                    raise ValueError("Source index must be increasing")
                source_lbound = source_index[j]
                if source_freq is None:
                    if j + 1 < len(source_index):
                        source_rbound = source_index[j + 1]
                    else:
                        source_rbound = None
                else:
                    source_rbound = source_index[j] + source_freq

                if target_rbound is not None and target_rbound <= source_lbound:
                    break
                if found_start or (
                    target_lbound >= source_lbound and (
                        source_rbound is None or target_lbound < source_rbound)
                ):
                    if not found_start:
                        from_j = j
                        found_start = True
                    if not source_mask[j]:
                        break
                    if (
                        source_rbound is None or
                        target_rbound is not None and
                        target_rbound <= source_rbound
                    ):
                        out[i] = True
                        break

        return out

    def resample_source_mask(
        self,
        source_mask: tp.Any,
        silence_warnings: tp.Optional[bool] = None,
    ) -> np.ndarray:
        """
        Resample a source mask to the target index.

        Parameters:
        -----------
        source_mask : tp.Any
            The mask associated with the source index.
        silence_warnings : Optional[bool], default=None
            Whether to silence all warnings.

        Returns:
        --------
        np.ndarray
            The resampled mask aligned with the target index.
        """
        silence_warnings = silence_warnings or self._silence_warnings

        target_shape = (
            (int(len(self.source_index)),)
            if checks.is_int(len(self.source_index))
            else tuple(len(self.source_index))
        )
        new_arr = (
            self.to_frame_array(source_mask)
            if len(target_shape) == 2
            else self.to_series_array(source_mask)
        )
        source_mask = np.broadcast_to(new_arr, target_shape)
        source_freq = self.get_np_source_freq(
            silence_warnings=silence_warnings
        )
        target_freq = self.get_np_target_freq(
            silence_warnings=silence_warnings
        )
        func = self._resample_source_mask_nb.py_func
        return func(
            source_mask,
            self.source_index.values,
            self.target_index.values,
            source_freq,
            target_freq,
        )
