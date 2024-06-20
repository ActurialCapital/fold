import abc
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset
import typing as tp

from fold.tools import SelectPosition, SplitPeriod, select_period
from fold.utils.indexing import combine_index, select_index


__all__ = ["Duration"]


class BaseDuration(abc.ABC):
    @abc.abstractmethod
    def get_bounds_arr(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_bounds(self, *args, **kwargs) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_duration(self, *args, **kwargs) -> pd.Series:
        pass


class Duration(BaseDuration):
    """
    Class for computing durations and bounds from splits and samples.

    Parameters
    ----------
    index : range
        The index used to define the ranges.
    n_splits : int
        Number of splits.
    n_samples : int
        Number of samples.
    split_labels : pd.Index
        Labels for splits.
    sample_labels : pd.Index
        Labels for samples.

    Attributes
    ----------
    index : pd.Index
        Index.
    n_splits : int
        Number of splits.
    n_samples : int
        Number of samples.
    split_labels : pd.Index
        Split labels.
    sample_labels : pd.Index
        Sample labels.
    """

    def __init__(
        self,
        index: range,
        splits_arr: np.ndarray,
        n_splits: int,
        n_samples: int,
        split_labels: pd.Index,
        sample_labels: pd.Index,
    ):
        self._index = index
        self._splits_arr = splits_arr
        self._n_splits = n_splits
        self._n_samples = n_samples
        self._split_labels = split_labels
        self._sample_labels = sample_labels

    @property
    def index(self) -> pd.Index:
        """
        Index.

        Returns
        -------
        pd.Index
            Index.
        """
        return self._index

    @property
    def splits_arr(self) -> np.ndarray:
        """
        Get two-dimensional splits array.

        Returns
        -------
        SplitsArray
            The splits array.

        Note
        ----
        First axis represents splits. Second axis represents samples. Elements
        represent periods. periods must be either a slice, a sequence of
        indices, a mask, or a callable that returns such.

        """
        return self._splits_arr

    @property
    def n_splits(self) -> int:
        """
        Number of splits.

        Returns
        -------
        int
            Number of splits
        """
        return self._n_splits

    @property
    def n_samples(self) -> int:
        """
        Number of samples.

        Returns
        -------
        int
            Number of samples
        """
        return self._n_samples

    @property
    def split_labels(self) -> pd.Index:
        """
        Split labels.

        Returns
        -------
        pd.Index
            Split labels index.

        """
        return self._split_labels

    @property
    def sample_labels(self) -> pd.Index:
        """
        sample labels.

        Returns
        -------
        pd.Index
            sample labels index.

        """
        return self._sample_labels

    @property
    def duration(self) -> pd.Series:
        """Get duration with default arguments."""
        return self.get_duration()

    @property
    def index_bounds(self) -> pd.Series:
        """Get duration with index_bounds=True."""
        return self.get_bounds(index_bounds=True)

    @property
    def index_duration(self) -> pd.Series:
        """Get duration with index_bounds=True."""
        return self.get_duration(index_bounds=True)

    def iter_bounds_arr(
        self,
        n_splits: int,
        n_samples: int,
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
        freq: tp.Optional[str | int | BaseOffset | pd.Timedelta] = None,
    ) -> np.ndarray:
        """
        Iterate over bounds array to get range bounds.

        Parameters
        ----------
        n_splits : int
            Number of splits.
        n_samples : int
            Number of samples.
        index_bounds : bool, default=False
            Flag to specify whether to use index bounds.
        right_inclusive : bool, default=False
            Flag to specify whether ranges are right inclusive.
        freq : tp.Optional[str | int | BaseOffset | pd.Timedelta], default=None
            Frequency of the index.

        Returns
        -------
        np.ndarray
            Three-dimensional integer array with bounds.
        """
        dtype = self.index.dtype if index_bounds else np.int_
        try:
            bounds = np.empty((n_splits, n_samples, 2), dtype=dtype)
        except TypeError:
            bounds = np.empty((n_splits, n_samples, 2), dtype=object)

        for split_idx in range(n_splits):
            for sample_idx in range(n_samples):
                period = select_period(
                    self.index,
                    self.get_periods(split_idx, sample_idx)
                )
                bounds[split_idx, sample_idx, :] = SplitPeriod.get_period_bounds(
                    period,
                    index_bounds=index_bounds,
                    right_inclusive=right_inclusive,
                    index=self.index,
                    freq=freq,
                )

        return bounds

    @staticmethod
    def stack_bounds(
        bounds_arr: np.ndarray,
        split_labels: pd.Index,
        sample_labels: pd.Index,
        squeeze_one_split: tp.Optional[bool] = True,
        squeeze_one_sample: tp.Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Stack bounds.

        Parameters
        ----------
        bounds_arr : np.ndarray
            Bounds array.
        split_labels : pd.Index
            Split labels.
        sample_labels : pd.Index
            Sample labels.
        squeeze_one_split : tp.Optional[bool], default=True
            Flag to specify whether to squeeze one split.
        squeeze_one_sample : tp.Optional[bool], default=True
            Flag to specify whether to squeeze one sample.

        Returns
        -------
        tp.Union[pd.Series, pd.DataFrame]
            Series or DataFrame with bounds stacked.
        """
        out = bounds_arr.reshape((-1, 2))
        labels = pd.Index(["start", "end"], name="bound")
        if (
            len(split_labels) == 1 and squeeze_one_split and
            len(sample_labels) == 1 and squeeze_one_sample
        ):
            return pd.Series(out[0], index=labels)

        if len(split_labels) == 1 and squeeze_one_split:
            return pd.DataFrame(out, index=sample_labels, columns=labels)

        if len(sample_labels) == 1 and squeeze_one_sample:
            return pd.DataFrame(out, index=split_labels, columns=labels)

        new_index = combine_index((split_labels, sample_labels))

        return pd.DataFrame(out, index=new_index, columns=labels)

    def split_index(
        self,
        selection: tp.Optional[tp.Any] = None
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Select split index object.

        Note
        ----
        Selection can be either integers and labels. Multiple values are 
        accepted; in such a case, the corresponding periods are merged.

        If selection is an integer data type, this method treats the provided 
        values as labels rather than index, unless the selection index is not
        of an integer data type or the values are wrapped with `SelectPosition`.

        If selection is not provided, this method selects the entire index.

        Returns two arrays: 
            * selected group indices
            * selected indices

        Parameters
        ----------
        selection : tp.Optional[tp.Any], optional
            Selection (default is None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing the split and sample indices.

        """
        if selection is None:
            return np.arange(self.n_splits)

        else:
            return select_index(self.split_labels, selection)

    def sample_index(
        self,
        selection: tp.Optional[tp.Any] = None
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Select sample index object.

        Note
        ----
        Selection can be either integers and labels. Multiple values are 
        accepted; in such a case, the corresponding periods are merged.

        If selection is an integer data type, this method treats the provided 
        values as labels rather than index, unless the selection index is not
        of an integer data type or the values are wrapped with `SelectPosition`.

        If selection is not provided, this method selects the entire index.

        Returns two arrays: 
            * selected group indices
            * selected indices

        Parameters
        ----------
        selection : tp.Optional[tp.Any], optional
            Selection (default is None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing the split and sample indices.

        """
        if selection is None:
            return np.arange(self.n_samples)

        else:
            return select_index(self.sample_labels, selection)

    def get_periods(
        self,
        split: tp.Optional[tp.Any] = None,
        sample: tp.Optional[tp.Any] = None
    ) -> list:
        """
        Get periods.

        Parameters
        ----------
        split : tp.Optional[tp.Any], optional
            Selection for splits (default is None).
        sample : tp.Optional[tp.Any], optional
            Selection for samples (default is None).

        Returns
        -------
        list
            Periods range.
        """
        return [
            self.splits_arr[i, j]
            for i in self.split_index(SelectPosition(split))
            for j in self.sample_index(SelectPosition(sample))
        ]

    def get_bounds_arr(
        self,
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
        freq: tp.Optional[str | int | BaseOffset | pd.Timedelta] = None,
    ) -> np.ndarray:
        """
        Get bounds array.

        Parameters
        ----------
        index_bounds : bool, default=False
            Flag to specify whether to use index bounds.
        right_inclusive : bool, default=False
            Flag to specify whether ranges are right inclusive.
        freq : tp.Optional[str | int | BaseOffset | pd.Timedelta], default=None
            Frequency of the index.

        Returns
        -------
        np.ndarray
            Three-dimensional integer array with bounds.
            * First axis represents splits. 
            * Second axis represents samples. 
            * Third axis represents bounds.
        """
        return self.iter_bounds_arr(
            self.n_splits,
            self.n_samples,
            index_bounds,
            right_inclusive,
            freq,
        )

    def get_bounds(
        self,
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
        squeeze_one_split: tp.Optional[bool] = True,
        squeeze_one_sample: tp.Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Get bounds.

        Parameters
        ----------
        index_bounds : bool, default=False
            Flag to specify whether to use index bounds.
        right_inclusive : bool, default=False
            Flag to specify whether ranges are right inclusive.
        squeeze_one_split : bool, default=True
            Flag to specify whether to squeeze one split.
        squeeze_one_sample : bool, default=True
            Flag to specify whether to squeeze one sample.

        Returns
        -------
        tp.Union[pd.Series, pd.DataFrame]
            Boolean Series or DataFrame where index are bounds and columns are 
            splits stacked together.
        """
        bounds_arr = self.get_bounds_arr(
            index_bounds=index_bounds,
            right_inclusive=right_inclusive,
        )
        return self.stack_bounds(
            bounds_arr,
            self.split_labels,
            self.sample_labels,
            squeeze_one_split,
            squeeze_one_sample
        )

    def get_duration(
        self,
        index_bounds: tp.Optional[bool] = False,
        squeeze_one_split: tp.Optional[bool] = True,
        squeeze_one_sample: tp.Optional[bool] = True,
    ) -> pd.Series:
        """
        Get duration.

        Parameters
        ----------
        index_bounds : bool, default=False
            Flag to specify whether to use index bounds.
        squeeze_one_split : bool, default=True
            Flag to specify whether to squeeze one split.
        squeeze_one_sample : bool, default=True
            Flag to specify whether to squeeze one sample.

        Returns
        -------
        pd.Series
            Duration calculated from bounds.
        """
        bounds = self.get_bounds(
            right_inclusive=False,
            index_bounds=index_bounds,
            squeeze_one_split=squeeze_one_split,
            squeeze_one_sample=squeeze_one_sample,
        )
        return (bounds["end"] - bounds["start"]).rename("duration")
