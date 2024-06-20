import warnings
from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import numpy as np
import typing as tp

from fold.tools.base import BasePeriod, BaseTool, BaseMetadata, BaseTemplate
from fold.tools.utils import Key
from fold.utils.indexing import (
    to_context_period,
    to_gap_period,
    to_number_period,
    to_datetime_period,
)
from fold.utils.datetime import (
    prepare_dt_index,
    to_freq,
    infer_index_freq,
    try_align_dt_to_index,
    try_align_to_dt_index
)
from fold.utils import checks


@dataclass
class RelativeSpace(BasePeriod, BaseMetadata):
    """
    Class representing a relative space based on a period and index.

    Parameters
    ----------
    period : tp.Any
        The period object representing the time period.
    index : range
        The index or range of the period.
    allow_relative : bool, optional, default=False
        Whether relative periods are allowed.
    allow_zero_len : bool, optional, default=False
        Whether a period of zero length is allowed.
    range_format : str, optional, default="slice_or_any"
        The format of the range, specifying slice or other formats.

    """
    period: tp.Any
    index: range
    allow_relative: bool = False
    allow_zero_len: bool = False
    range_format: str = "slice_or_any"

    def bounds(self):
        pass

    def span(self):
        pass

    def coverage(self) -> tp.Any:
        return self.period

    def __post_init__(self):
        """
        Perform post-initialization checks and operations.

        Raises
        ------
        TypeError
            If relative periods are not allowed.
        """
        if not self.allow_relative:
            raise TypeError("Relative periods must be converted to fixed")
        period = self.coverage()
        BaseMetadata.__init__(self, period, index=self.index)


@dataclass
class SliceSpace(BasePeriod, BaseMetadata):
    """
    Class representing a slice-based space using period and index.

    Parameters
    ----------
    period : tp.Any
        The period object representing the time period.
    index : range
        The index or range of the period.
    allow_relative : bool, optional, default=False
        Whether relative periods are allowed.
    allow_zero_len : bool, optional, default=False
        Whether a period of zero length is allowed.
    range_format : str, optional, default="slice_or_any"
        The format of the range, specifying slice or other formats.

    """
    period: tp.Any
    index: range
    allow_relative: bool = False
    allow_zero_len: bool = False
    range_format: str = "slice_or_any"

    def _handle_start_bound(self, start: int) -> int:
        """Handle start"""
        start = 0 if start is None else start
        if not checks.is_int(start):
            checks.assert_dt_index(self.index)
            start = try_align_dt_to_index(start, self.index)
            checks.assert_can_be_parsed(start, name='start')
        if checks.is_int(start):
            start = 0 if start < 0 else start
        else:
            if start < self.index[0]:
                start = 0
            else:
                start = self.index.get_indexer([start], method="bfill")[0]
                checks.assert_not_out_of_bounds(start, name='start')

        return start

    def _handle_stop_bound(self, stop: int) -> int:
        """Handle stop"""
        stop = len(self.index) if stop is None else stop
        if not checks.is_int(stop):
            checks.assert_dt_index(self.index)
            stop = try_align_dt_to_index(stop, self.index)
            checks.assert_can_be_parsed(stop, name='stop')
        if checks.is_int(stop):
            stop = len(self.index) if stop > len(self.index) else stop
        else:
            if stop > self.index[-1]:
                stop = len(self.index)
            else:
                stop = self.index.get_indexer([stop], method="bfill")[0]
                checks.assert_not_out_of_bounds(stop, name='stop')

        return stop

    def bounds(self, period: tp.Any) -> tp.Tuple[int, int]:
        """
        Calculate the bounds of the slice-based space.

        Parameters
        ----------
        period : tp.Any
            The period object representing the time period.

        Returns
        -------
        tuple
            Start and stop bounds of the period.

        Raises
        ------
        ValueError
            If start cannot be converted to slice or is out of bounds.
        """
        start = period.start
        stop = period.stop
        checks.assert_step_is_valid(period.step)
        checks.assert_slice_is_valid(start, stop)
        if start is not None and checks.is_int(start) and start < 0:
            start = len(self.index) + start
            stop = (
                len(self.index) + stop
                if stop is not None and checks.is_int(stop)
                else stop
            )

        start = self._handle_start_bound(start)
        stop = self._handle_stop_bound(stop)
        return start, stop

    def span(self, start: int, stop: int) -> int:
        """
        Calculate the span of the slice-based space.

        Parameters
        ----------
        start : int
            Start index of the slice.
        stop : int
            Stop index of the slice.

        Returns
        -------
        int
            Span of the period.
        """
        return stop - start

    def coverage(self, start: int, stop: int) -> tp.Any:
        """
        Determine the coverage of the slice-based space.

        Parameters
        ----------
        start : int
            Start index of the slice.
        stop : int
            Stop index of the slice.

        Returns
        -------
        tp.Any
            The period representing coverage.
        """
        period = slice(start, stop)
        if self.range_format.lower() == "indices":
            period = np.arange(*period.indices(len(self.index)))

        elif self.range_format.lower() == "mask":
            mask = np.full(len(self.index), False)
            mask[period] = True
            period = mask

        return period

    def __post_init__(self):
        """
        Perform post-initialization checks and operations.

        Raises
        ------
        ValueError
            If slice bounds are invalid.
        """
        start, stop = self.bounds(self.period)
        length = self.span(start, stop)
        period = self.coverage(start, stop)
        checks.assert_zero_len(length, self.allow_zero_len)
        checks.assert_bounds_is_valid(start, stop, self.index)
        BaseMetadata.__init__(
            self,
            period,
            index=self.index,
            start=start,
            stop=stop,
            length=length
        )


@dataclass
class MaskSpace(BasePeriod, BaseMetadata):
    """
    Class representing a mask-based space using period and index.

    Parameters
    ----------
    period : tp.Any
        The period object representing the time period.
    index : range
        The index or range of the period.
    allow_relative : bool, optional, default=False
        Whether relative periods are allowed.
    allow_zero_len : bool, optional, default=False
        Whether a period of zero length is allowed.
    range_format : str, optional, default="slice_or_any"
        The format of the range, specifying slice or other formats.

    """
    period: tp.Any
    index: range
    allow_relative: bool = False
    allow_zero_len: bool = False
    range_format: str = "slice_or_any"

    def bounds(self, indices) -> tp.Tuple[int, int]:
        """
        Calculate the bounds of the mask-based space.

        Parameters
        ----------
        indices : list or np.ndarray
            Indices representing the mask.

        Returns
        -------
        tuple
            Start and stop bounds of the period.
        """
        start, stop = (
            (0, 0)
            if len(indices) == 0
            else (indices[0], indices[-1] + 1)
        )
        return start, stop

    def span(self, indices: tp.List | np.ndarray) -> int:
        """
        Calculate the span of the mask-based space.

        Parameters
        ----------
        indices : list or np.ndarray
            Indices representing the mask.

        Returns
        -------
        int
            Span of the period.
        """
        return len(indices)

    def coverage(
        self,
        indices: tp.List | np.ndarray,
        start: int,
        stop: int
    ) -> tp.Any:
        """
        Determine the coverage of the mask-based space.

        Parameters
        ----------
        indices : list or np.ndarray
            Indices representing the mask.
        start : int
            Start index of the slice.
        stop : int
            Stop index of the slice.

        Returns
        -------
        tp.Any
            The period representing coverage.

        Raises
        ------
        ValueError
            If range format cannot be converted to slice.
        """
        if self.range_format.lower() == "indices":
            return indices

        elif self.range_format.lower().startswith("slice"):
            is_valid_range = len(indices) == 0 or checks.is_range(indices)
            if not is_valid_range:
                if self.range_format.lower() == "slice":
                    raise ValueError(
                        "Cannot convert to slice: Period is not continuous"
                    )
                if self.range_format.lower() == "slice_or_indices":
                    return indices
            else:
                return slice(start, stop)

        return self.period

    def __post_init__(self):
        """
        Perform post-initialization checks and operations.

        Raises
        ------
        ValueError
            If indices are not valid or out of bounds.
        """
        checks.assert_same_length(self.period, self.index)
        indices = np.flatnonzero(self.period)
        checks.assert_zero_len(len(indices), self.allow_zero_len)

        start, stop = self.bounds(indices)
        length = self.span(indices)
        period = self.coverage(indices, start, stop)

        checks.assert_bounds_is_valid(start, stop, self.index)
        BaseMetadata.__init__(
            self,
            period,
            index=self.index,
            start=start,
            stop=stop,
            length=length,
            range_format="slice_or_mask"
        )


@dataclass
class IndexSpace(BasePeriod, BaseMetadata):
    """
    Class representing an index-based space using period and index.

    Parameters
    ----------
    period : tp.Any
        The period object representing the time period.
    index : range
        The index or range of the period.
    allow_relative : bool, optional, default=False
        Whether relative periods are allowed.
    allow_zero_len : bool, optional, default=False
        Whether a period of zero length is allowed.
    range_format : str, optional, default="slice_or_any"
        The format of the range, specifying slice or other formats.

    """
    period: tp.Any
    index: range
    allow_relative: bool = False
    allow_zero_len: bool = False
    range_format: str = "slice_or_any"

    def _to_index(self, period: tp.Any) -> tp.Any:
        """
        Convert period to index-based representation.

        Parameters
        ----------
        period : tp.Any
            The period object representing the time period.

        Returns
        -------
        tp.Any
            The period converted to index-based representation.
        """
        if not np.issubdtype(period.dtype, np.integer):
            period = try_align_to_dt_index(period, self.index)
            checks.assert_dt_index(period)
            period = self.index.get_indexer(period, method=None)
            checks.assert_range_values_is_valid(period)
        return period

    def bounds(self, period: tp.Any) -> tp.Tuple[int, int]:
        """
        Calculate the bounds of the index-based space.

        Parameters
        ----------
        period : tp.Any
            The period object representing the time period.

        Returns
        -------
        tuple
            Start and stop bounds of the period.

        Raises
        ------
        TypeError
            If period data type is invalid.
        """
        start, stop = (
            (0, 0)
            if len(period) == 0
            else (period[0], period[-1] + 1)
            if checks.is_range(period)
            else (np.min(period), np.max(period) + 1)
        )
        return start, stop

    def span(self, period: tp.Any) -> int:
        """
        Calculate the span of the index-based space.

        Parameters
        ----------
        period : tp.Any
            The period object representing the time period.

        Returns
        -------
        int
            Span of the period.
        """
        return len(period)

    def coverage(self, start: int, stop: int, period: tp.Any) -> tp.Any:
        """
        Determine the coverage of the index-based space.

        Parameters
        ----------
        start : int
            Start index of the slice.
        stop : int
            Stop index of the slice.
        period : tp.Any
            The period object representing the time period.

        Returns
        -------
        tp.Any
            The period representing coverage.

        Raises
        ------
        TypeError
            If period data type is invalid.
        """
        if self.range_format.lower() == "mask":
            mask = np.full(len(self.index), False)
            mask[period] = True
            period = mask
        elif self.range_format.lower().startswith("slice"):
            is_constant = len(period) == 0 or checks.is_range(period)
            if not is_constant:
                checks.assert_slice_range_not_constant(self.range_format)
                if self.range_format.lower() == "slice_or_mask":
                    mask = np.full(len(self.index), False)
                    mask[period] = True
                    period = mask
            else:
                period = slice(start, stop)

        return period

    def __post_init__(self):
        """
        Perform post-initialization checks and operations.

        Raises
        ------
        TypeError
            If period data type is invalid.
        """
        period = self._to_index(self.period)
        if np.issubdtype(period.dtype, np.integer):
            start, stop = self.bounds(period)
            length = self.span(period)
            period = self.coverage(start, stop, period)

        else:
            raise TypeError(
                f"Range array has invalid data type ({period.dtype})"
            )
        checks.assert_bounds_is_valid(start, stop, self.index)
        BaseMetadata.__init__(
            self,
            period,
            index=self.index,
            start=start,
            stop=stop,
            length=length,
            range_format="slice_or_indices"
        )


@dataclass
class FixedPeriod(BaseTemplate):
    """
    Class that represents a fixed period.

    Parameters
    ----------
    period : Range.

    """
    period: tp.Any


@dataclass
class RelativePeriod(BaseTemplate):
    """
    Class that represents a relative period.

    Parameters
    ----------
    offset : int | pd.Timedelta, optional
        Offset. Floating values between 0 and 1 are considered relative. It can 
        be negative. Default is 0.
    offset_anchor : str, optional
        Offset anchor. Supported anchors are:
        - 'start': Start of the range
        - 'end': End of the range
        - 'prev_start': Start of the previous range
        - 'prev_end': End of the previous range
        Default is 'prev_end'.
    offset_space : str, optional
        Offset space.
        Supported spaces are:
        - 'all': All space
        - 'free': Remaining space after the offset anchor
        - 'prev': Length of the previous range
        Applied only when `offset` is a relative number.
        Default is 'free'.
    length : int | pd.Timedelta, optional
        Length.
        Floating values between 0 and 1 are considered relative.
        Can be negative.
        Default is 1.0.
    length_space : str, optional
        Length space.
        Supported spaces are:
        - 'all': All space
        - 'free': Remaining space after the offset
        - 'free_or_prev': Remaining space after the offset or the start/end of
            the previous range,depending on the direction of `length`
        Applied only when `length` is a relative number.
        Default is 'free'.
    out_of_bounds : str, optional
        Check if start and stop are within bounds.
        Supported actions are:
        - 'keep': Keep out-of-bounds values
        - 'ignore': Ignore if out-of-bounds
        - 'warn': Emit a warning if out-of-bounds
        - 'raise': Raise an error if out-of-bounds
        Default is 'warn'.
    is_gap : bool, optional
        Whether the range acts as a gap.
        Default is False.

    Methods
    -------
    to_slice
        Convert the relative range into a slice.

    """
    offset: int | pd.Timedelta = 0
    offset_anchor: str = "prev_end"
    offset_space: str = "free"
    length: int | pd.Timedelta = 1.0
    length_space: str = "free"
    out_of_bounds: str = "warn"
    is_gap: bool = False

    def __post_init__(self):
        """
        Validate and normalize the initialization parameters of the 
        RelativePeriod object.

        Converts string parameters to lowercase and checks if they are within 
        the supported options.

        Raises
        ------
        ValueError
            If `offset_anchor` is not one of 'start', 'end', 'prev_start', 
            'prev_end', 'next_start', 'next_end'.
            If `offset_space` is not one of 'all', 'free', 'prev'.
            If `length_space` is not one of 'all', 'free', 'free_or_prev'.
            If `out_of_bounds` is not one of 'keep', 'ignore', 'warn', 'raise'.
        """
        object.__setattr__(self, "offset_anchor", self.offset_anchor.lower())
        if self.offset_anchor not in (
            "start",
            "end",
            "prev_start",
            "prev_end",
            "next_start",
            "next_end"
        ):
            raise ValueError(
                f"Invalid option offset_anchor='{self.offset_anchor}'")
        object.__setattr__(self, "offset_space", self.offset_space.lower())
        if self.offset_space not in (
            "all",
            "free",
            "prev"
        ):
            raise ValueError(
                f"Invalid option offset_space='{self.offset_space}'")
        object.__setattr__(self, "length_space", self.length_space.lower())
        if self.length_space not in (
            "all",
            "free",
            "free_or_prev"
        ):
            raise ValueError(
                f"Invalid option length_space='{self.length_space}'")
        object.__setattr__(self, "out_of_bounds", self.out_of_bounds.lower())
        if self.out_of_bounds not in (
            "keep",
            "ignore",
            "warn",
            "raise"
        ):
            raise ValueError(
                f"Invalid option out_of_bounds='{self.out_of_bounds}'")

    def to_slice(
        self,
        total_len: int,
        prev_start: tp.Optional[int] = 0,
        prev_end: tp.Optional[int] = 0,
        index: tp.Optional[range] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
    ) -> slice:
        """
        Convert the relative range into a slice.

        Parameters
        ----------
        total_len : int
            Total length of the range.
        prev_start : int, optional
            Start of the previous range, by default 0.
        prev_end : int, optional
            End of the previous range, by default 0.
        index : tp.Optional[range], optional
            Index to align with the range, by default None.
        freq : tp.Optional[str | int | float | Offset | pd.Timedelta], optional
            Frequency of the index, by default "auto".

        Returns
        -------
        slice
            The resulting slice object.

        Raises
        ------
        TypeError
            If index is not of type pandas.DatetimeIndex.
        ValueError
            If frequency is not provided when required.
            If range start or stop is out of bounds.
            If range length is negative or zero.
        """
        if index is not None:
            index = prepare_dt_index(index)
            try:
                freq = infer_index_freq(index, freq, allow_numeric=False)
            except Exception:
                freq = None

        offset_anchor = self.offset_anchor
        offset = self.offset
        length = self.length
        if not checks.is_number(offset) or not checks.is_number(length):
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError(
                    "Index must be of type pandas.DatetimeIndex, not "
                    f"{index.dtype}"
                )

        if offset_anchor == "start":
            if checks.is_number(offset):
                offset_anchor = 0
            else:
                offset_anchor = index[0]
        elif offset_anchor == "end":
            if checks.is_number(offset):
                offset_anchor = total_len
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                offset_anchor = index[-1] + freq
        elif offset_anchor == "prev_start":
            if checks.is_number(offset):
                offset_anchor = prev_start
            else:
                offset_anchor = index[prev_start]
        else:
            if checks.is_number(offset):
                offset_anchor = prev_end
            else:
                if prev_end < total_len:
                    offset_anchor = index[prev_end]
                else:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset_anchor = index[-1] + freq

        if checks.is_float(offset) and 0 <= abs(offset) <= 1:
            if self.offset_space == "all":
                offset = offset_anchor + int(offset * total_len)
            elif self.offset_space == "free":
                if offset < 0:
                    offset = int((1 + offset) * offset_anchor)
                else:
                    offset = offset_anchor + \
                        int(offset * (total_len - offset_anchor))
            else:
                offset = offset_anchor + int(offset * (prev_end - prev_start))
        else:
            if checks.is_float(offset):
                if not offset.is_integer():
                    raise TypeError(
                        f"Floating number for offset ({offset}) "
                        "must be between 0 and 1"
                    )
                offset = offset_anchor + int(offset)
            elif not checks.is_int(offset):
                offset = offset_anchor + to_freq(offset)
                if index[0] <= offset <= index[-1]:
                    offset = index.get_indexer([offset], method="ffill")[0]
                elif offset < index[0]:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset = -int((index[0] - offset) / freq)
                else:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset = total_len - 1 + int((offset - index[-1]) / freq)
            else:
                offset = offset_anchor + offset

        if checks.is_float(length) and 0 <= abs(length) <= 1:
            if self.length_space == "all":
                length = int(length * total_len)
            elif self.length_space == "free":
                if length < 0:
                    length = int(length * offset)
                else:
                    length = int(length * (total_len - offset))
            else:
                if length < 0:
                    if offset > prev_end:
                        length = int(length * (offset - prev_end))
                    else:
                        length = int(length * offset)
                else:
                    if offset < prev_start:
                        length = int(length * (prev_start - offset))
                    else:
                        length = int(length * (total_len - offset))
        else:
            if checks.is_float(length):
                if not length.is_integer():
                    raise TypeError(
                        f"Floating number for length ({length}) "
                        "must be between 0 and 1"
                    )
                length = int(length)
            elif not checks.is_int(length):
                length = to_freq(length)

        start = offset
        if checks.is_int(length):
            stop = start + length
        else:
            if 0 <= start < total_len:
                stop = index[start] + length
            elif start < 0:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = index[0] + start * freq + length
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = index[-1] + (start - total_len + 1) * freq + length
            if stop <= index[-1]:
                stop = index.get_indexer([stop], method="bfill")[0]
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = total_len - 1 + int((stop - index[-1]) / freq)
        if checks.is_int(length):
            if length < 0:
                start, stop = stop, start
        else:
            if length < pd.Timedelta(0):
                start, stop = stop, start
        if start < 0:
            if self.out_of_bounds == "ignore":
                start = 0
            elif self.out_of_bounds == "warn":
                warnings.warn(
                    f"Range start ({start}) is out of bounds",
                    stacklevel=2
                )
                start = 0
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range start ({start}) is out of bounds")
        if stop > total_len:
            if self.out_of_bounds == "ignore":
                stop = total_len
            elif self.out_of_bounds == "warn":
                warnings.warn(
                    f"Range stop ({stop}) is out of bounds",
                    stacklevel=2
                )
                stop = total_len
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range stop ({stop}) is out of bounds")
        if stop - start <= 0:
            raise ValueError("Range length is negative or zero")
        return slice(start, stop)


class PeriodTransformer:
    """
    A transformer class to handle various period transformations.

    Parameters
    ----------
    period : tp.Any
        The period to transform.
    index : range
        The index to align with the period.
    allow_relative : tp.Optional[bool], optional
        Whether to allow relative periods, by default False.
    allow_zero_len : tp.Optional[bool], optional
        Whether to allow periods with zero length, by default False.
    range_format : tp.Optional[str], optional
        The desired format of the returned range, by default "slice_or_any".
    template_context : tp.Optional[tp.Dict[str, tp.Any]], optional
        Contextual information for template substitution, by default None.
    """

    def __init__(
        self,
        period: tp.Any,
        index: range,
        allow_relative: tp.Optional[bool] = False,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = "slice_or_any",
        template_context: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        checks.assert_range_format_is_valid(range_format)
        index = prepare_dt_index(index)

        if isinstance(period, FixedPeriod):
            period = period.period

        if isinstance(period, BaseTool):
            template_context = template_context or {}
            if "index" not in template_context:
                template_context["index"] = index
            period = period.substitute(
                context=template_context,
                eval_id="range"
            )

        if callable(period):
            period = period(index)

        if not checks.is_range_relative(period) and not isinstance(period, slice):
            period = np.asarray(period)

        model_cls = (
            RelativeSpace
            if checks.is_range_relative(period)
            else SliceSpace
            if isinstance(period, slice)
            else MaskSpace
            if np.issubdtype(period.dtype, np.bool_)
            else IndexSpace
        )
        self._model = model_cls(
            period,
            index,
            allow_relative,
            allow_zero_len,
            range_format,
        )

    @property
    def model(self) -> tp.Type[BasePeriod]:
        """
        Get the model instance.

        Returns
        -------
        tp.Type[BasePeriod]
            The model instance.
        """
        return self._model

    @property
    def model_output(self) -> tp.Dict[str, tp.Any]:
        """
        Get the model output.

        Returns
        -------
        tp.Dict[str, tp.Any]
            The model output.
        """
        return self.model.output

    @property
    def period(self) -> tp.Any:
        """
        Get the period.

        Returns
        -------
        tp.Any
            The period.
        """
        return self.model.period

    @property
    def index(self) -> range:
        """
        Get the index.

        Returns
        -------
        range
            The index.
        """
        return self.model.output['index']

    @property
    def start(self) -> int:
        """
        Get the start index.

        Returns
        -------
        int
            The start index.
        """
        return self.model.output['start']

    @property
    def stop(self) -> int:
        """
        Get the stop index.

        Returns
        -------
        int
            The stop index.
        """
        return self.model.output['stop']

    @property
    def length(self) -> int:
        """
        Get the length of the period.

        Returns
        -------
        int
            The length of the period.
        """
        return self.model.output['length']

    @property
    def layout(self) -> str:
        """
        Get the range format.

        Returns
        -------
        str
            The range format.
        """
        return self.model.output['range_format']


class SplitPeriod(PeriodTransformer):
    """
    Split a fixed range into multiple fixed periods based on the given split
    criteria.

    Parameters
    ----------
    period : FixedRangeLike
        The range to split.
    new_split : SplitLike
        The criteria to use for splitting the range.
    backwards : bool, optional
        Whether to split the range in reverse order, by default False.
    allow_zero_len : bool, optional
        Whether to allow periods with zero length, by default False.
    range_format : Optional[str], optional
        The desired format of the returned range, by default None.
    wrap_with_template : bool, optional
        Whether to wrap the resulting periods with a template, by default False.
    wrap_with_fixrange : Optional[bool], optional
        Whether to wrap the periods with FixedRange, by default False.
    template_context : tp.Dict[str, tp.Any], optional
        Contextual information for template substitution, by default None.
    index : Optional[IndexLike], optional
        The index to align with the range, by default None.
    freq : Optional[FrequencyLike], optional
        Frequency information for datetime-like periods, by default None.

    Returns
    -------
    FixSplit
        A tuple of fixed periods resulting from the split.

    Notes
    -----
    The `new_split` parameter can be:
    - An iterable of fixed or relative periods.
    - An integer or float indicating the length to split.
    - A string indicating a specific split method ('by_gap' is supported).

    Each sub-range in `new_split` can be a FixedRange, RelativeRange, or a
    number that represents the length for creating a RelativeRange.

    If `wrap_with_template` is enabled, resulting periods will be wrapped with a
    template.
    If `wrap_with_fixrange` is None, the function will determine whether to
    wrap with FixedRange.

    """

    def __init__(
        self,
        period: tp.Any,
        index: range,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        template_context: tp.Optional[tp.Dict[str, tp.Any]] = None,
        freq: tp.Optional[str | int | Offset | pd.Timedelta] = None,
        wrap_with_template: tp.Optional[bool] = False,
        wrap_with_fixrange: tp.Optional[bool] = False
    ):
        super().__init__(
            period,
            index,
            allow_zero_len=allow_zero_len,
            range_format="slice_or_indices",
            template_context=template_context,
        )
        self.template_context = template_context
        self.freq = freq
        self.wrap_with_template = wrap_with_template
        self.wrap_with_fixrange = wrap_with_fixrange
        self.allow_zero_len = allow_zero_len
        self.template_context = template_context
        self.range_format = (
            self.model_output['range_format'] or "slice_or_any"
            if range_format is None
            else range_format
        )

    def handle_data(self, split: tp.Any, backwards: bool):
        """
        Handle and process the split data.

        Parameters
        ----------
        split : tp.Any
            The split criteria.
        backwards : bool
            Whether to split in reverse order.

        Returns
        -------
        split : tp.Any
            Processed split criteria.
        backwards : bool
            Whether to split in reverse order.
        """
        if isinstance(split, BaseTool):
            split = to_context_period(self.index[self.period], split)

        if isinstance(split, str) and split.lower() == "by_gap":
            split = to_gap_period(self.index, self.period)

        if checks.is_number(split):
            split, backwards = to_number_period(split, backwards)

        elif checks.is_td_like(split):
            split, backwards = to_datetime_period(split, backwards)

        elif not checks.is_iterable(split):
            split = (split,)

        return split, backwards

    def update(
        self,
        new_split: tp.Any,
        prev_start: int,
        prev_end: int,
        backwards: bool
    ):
        """
        Update the periods based on the new split criteria.

        Parameters
        ----------
        new_split : tp.Any
            The new split criteria.
        prev_start : int
            The previous start index.
        prev_end : int
            The previous end index.
        backwards : bool
            Whether to split in reverse order.

        Returns
        -------
        updated_periods : tuple
            A tuple of updated periods.
        """
        updated_periods = []
        for updated_period in new_split:
            model = PeriodTransformer(
                period=updated_period,
                index=self.index[self.period],
                allow_relative=True,
                allow_zero_len=self.allow_zero_len,
                range_format="slice_or_any",
                template_context=self.template_context,
            )
            updated_period = model.period

            if checks.is_number(updated_period) or checks.is_td_like(updated_period):
                updated_period = RelativePeriod(length=updated_period)

            if isinstance(updated_period, RelativePeriod):
                updated_period = updated_period.to_slice(
                    total_len=self.length,
                    prev_start=self.length - prev_end if backwards else prev_start,
                    prev_end=self.length - prev_start if backwards else prev_end,
                    index=self.index,
                    freq=self.freq,
                )
                if backwards:
                    updated_period = slice(
                        self.length - updated_period.stop,
                        self.length - updated_period.start
                    )

            # Update previous bounds
            if isinstance(updated_period, slice):
                prev_start = updated_period.start
                prev_end = updated_period.stop
            else:
                prev_start = model.start
                prev_end = model.stop

            if isinstance(self.period, slice) and isinstance(updated_period, slice):
                updated_period = slice(
                    self.period.start + updated_period.start,
                    self.period.start + updated_period.stop,
                )
            else:
                if isinstance(self.period, slice):
                    updated_period = np.arange(
                        self.period.start,
                        self.period.stop
                    )[updated_period]
                else:
                    updated_period = self.period[updated_period]

            model_cls = PeriodTransformer(
                period=updated_period,
                index=self.index,
                allow_zero_len=self.allow_zero_len,
                range_format=self.range_format,
            )
            updated_period = model_cls.period

            if self.wrap_with_template:
                updated_period = Key(
                    "period", context=dict(period=updated_period))

            if self.wrap_with_fixrange and checks.is_sequence(updated_period):
                updated_period = FixedPeriod(updated_period)

            updated_periods.append(updated_period)

        return tuple(updated_periods)

    def split(self, new_split: tp.Any, backwards: tp.Optional[bool] = False):
        """
        Split the range based on the new split criteria.

        Parameters
        ----------
        new_split : tp.Any
            The new split criteria.
        backwards : bool, optional
            Whether to split in reverse order, by default False.

        Returns
        -------
        updated : tuple
            A tuple of updated periods.
        """
        new_split, backwards = self.handle_data(new_split, backwards)
        if backwards:
            new_split = new_split[::-1]
            prev_start = prev_end = self.length
        else:
            prev_start, prev_end = (0, 0)

        updated = self.update(
            new_split,
            prev_start=prev_start,
            prev_end=prev_end,
            backwards=backwards
        )
        return updated[::-1] if backwards else updated

    @classmethod
    def get_period_mask(
        cls,
        period: tp.Any,
        index: tp.Optional[range] = None,
    ) -> np.ndarray:
        """
        Get the mask of a range.

        Parameters
        ----------
        period : tp.Any
            The period to mask.
        index : tp.Optional[range], optional
            The index to align with the period, by default None.

        Returns
        -------
        np.ndarray
            A boolean mask array.
        """
        index = prepare_dt_index(index)
        model = cls(period, index, allow_zero_len=True)

        if isinstance(model.period, np.ndarray) and model.period.dtype == np.bool_:
            return model.period

        mask = np.full(len(index), False)
        mask[model.period] = True
        return mask

    @classmethod
    def get_period_bounds(
        cls,
        period: tp.Any,
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
        index: tp.Optional[range] = None,
        freq: tp.Optional[str | int | Offset | pd.Timedelta] = "auto",
    ) -> tp.Tuple[tp.Any, tp.Any]:
        """
        Get the left (inclusive) and right (exclusive) bound of a range.

        Parameters
        ----------
        period : np.ndarray
            Array representing the range.
        index_bounds : bool, optional
            Flag to indicate if index bounds are used (default is False).
        right_inclusive : bool, optional
            Flag to indicate if the right bound is inclusive (default is False).
        index : pd.Index, optional
            Index for range bounds (default is None).
        freq : tp.Optional[str | int | Offset | pd.Timedelta], optional
            Frequency of the index (default is None).

        Returns
        -------
        Tuple[Any, Any]
            Tuple containing the left (inclusive) and right (exclusive) bounds of
            the range.
        """
        model = cls(period, index)
        index, start, stop = model.index, model.start, model.stop
        if index_bounds:
            if right_inclusive:
                start, stop = (index[start], index[stop - 1])
            else:
                if stop == len(index):
                    try:
                        freq = infer_index_freq(index, freq)
                    except Exception:
                        raise ValueError(
                            "Please provide a frequency. freq={freq}."
                        )
                    start, stop = index[start], index[stop - 1] + freq
                else:
                    start, stop = index[start], index[stop]

        else:
            if right_inclusive:
                stop = stop - 1
        return start, stop


def select_period(index: range, periods: list) -> tp.Any:
    """
    Select period.

    Parameters
    ----------
    index : range
        Index associated with the model.
    periods : list
        Periods range.

    Returns
    -------
    tp.Any
        The selected range.

    """
    # Dealing with a single period
    if len(periods) == 1:
        return periods[0]

    # Dealing with multiple periods
    all_masks = True
    updates_periods = []
    for period in periods:
        transformer = PeriodTransformer(
            period,
            index,
            allow_zero_len=True,
            range_format="any",
        )
        if not transformer.layout == 'slice_or_mask':
            all_masks = False

        updates_periods.append(transformer.period)

    # Create a binary array
    updated_period = np.full(len(index), False)
    for period in updates_periods:
        updated_period[period] = True

    transformer = PeriodTransformer(
        updated_period,
        index,
        range_format="slice_or_mask" if all_masks else "slice_or_indices"
    )
    return transformer.period
