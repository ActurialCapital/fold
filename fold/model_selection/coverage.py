import abc
import numpy as np
import pandas as pd
from functools import cache
from numba import jit, prange
import typing as tp

from fold.tools import SelectPosition, SplitPeriod, select_period
from fold.utils.indexing import combine_index, select_index


__all__ = ["Coverage"]


class BaseCoverage(abc.ABC):
    @abc.abstractmethod
    def get_mask(self, *args, **kwargs) -> tp.Union[pd.Series, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_mask_arr(self) -> np.ndarray:
        pass


class Overlap:
    @staticmethod
    @jit(cache=True)
    def by_splits(mask_arr: np.ndarray) -> np.ndarray:
        """
        Compute the cross matrix for splits.

        Parameters
        ----------
        mask_arr : numpy.ndarray
            3D array representing split masks.

        Returns
        -------
        numpy.ndarray
            cross matrix for splits.

        """
        out = np.empty((mask_arr.shape[0], mask_arr.shape[0]), dtype=np.int_)
        temp_mask = np.empty(
            (mask_arr.shape[0], mask_arr.shape[2]), dtype=np.bool_)
        for i in range(mask_arr.shape[0]):
            for m in range(mask_arr.shape[2]):
                if mask_arr[i, :, m].any():
                    temp_mask[i, m] = True
                else:
                    temp_mask[i, m] = False
        for i1 in prange(mask_arr.shape[0]):
            for i2 in range(mask_arr.shape[0]):
                intersection = (temp_mask[i1] & temp_mask[i2]).sum()
                out[i1, i2] = intersection
        return out

    @staticmethod
    @jit(cache=True)
    def by_norm_samples(mask_arr: np.ndarray) -> np.ndarray:
        """
        Compute the normalized cross matrix for samples.

        Parameters
        ----------
        mask_arr : numpy.ndarray
            3D array representing split masks.

        Returns
        -------
        numpy.ndarray
            Normalized cross matrix for samples.

        """
        out = np.empty((mask_arr.shape[1], mask_arr.shape[1]), dtype=np.float_)
        temp_mask = np.empty(
            (mask_arr.shape[1], mask_arr.shape[2]), dtype=np.bool_)
        for j in range(mask_arr.shape[1]):
            for m in range(mask_arr.shape[2]):
                if mask_arr[:, j, m].any():
                    temp_mask[j, m] = True
                else:
                    temp_mask[j, m] = False
        for j1 in prange(mask_arr.shape[1]):
            for j2 in range(mask_arr.shape[1]):
                intersection = (temp_mask[j1] & temp_mask[j2]).sum()
                union = (temp_mask[j1] | temp_mask[j2]).sum()
                out[j1, j2] = intersection / union
        return out

    @staticmethod
    @jit(cache=True)
    def by_norm_period(mask_arr: np.ndarray) -> np.ndarray:
        """
        Compute the normalized cross matrix for ranges.

        Parameters
        ----------
        mask_arr : numpy.ndarray
            3D array representing split masks.

        Returns
        -------
        numpy.ndarray
            Normalized cross matrix for ranges.

        """
        out = np.empty((mask_arr.shape[0] * mask_arr.shape[1],
                       mask_arr.shape[0] * mask_arr.shape[1]), dtype=np.float_)
        for k in prange(mask_arr.shape[0] * mask_arr.shape[1]):
            i1 = k // mask_arr.shape[1]
            j1 = k % mask_arr.shape[1]
            for l in range(mask_arr.shape[0] * mask_arr.shape[1]):
                i2 = l // mask_arr.shape[1]
                j2 = l % mask_arr.shape[1]
                intersection = (mask_arr[i1, j1] & mask_arr[i2, j2]).sum()
                union = (mask_arr[i1, j1] | mask_arr[i2, j2]).sum()
                out[k, l] = intersection / union
        return out

    @staticmethod
    @jit(cache=True)
    def by_period(mask_arr: np.ndarray) -> np.ndarray:
        """
        Compute the cross matrix for ranges.

        Parameters
        ----------
        mask_arr : numpy.ndarray
            3D array representing split masks.

        Returns
        -------
        numpy.ndarray
            cross matrix for ranges.

        """
        out = np.empty((mask_arr.shape[0] * mask_arr.shape[1],
                       mask_arr.shape[0] * mask_arr.shape[1]), dtype=np.int_)
        for k in prange(mask_arr.shape[0] * mask_arr.shape[1]):
            i1 = k // mask_arr.shape[1]
            j1 = k % mask_arr.shape[1]
            for l in range(mask_arr.shape[0] * mask_arr.shape[1]):
                i2 = l // mask_arr.shape[1]
                j2 = l % mask_arr.shape[1]
                intersection = (mask_arr[i1, j1] & mask_arr[i2, j2]).sum()
                out[k, l] = intersection
        return out

    @staticmethod
    @jit(cache=True)
    def by_samples(mask_arr: np.ndarray) -> np.ndarray:
        """
        Compute the cross matrix for samples.

        Parameters
        ----------
        mask_arr : numpy.ndarray
            3D array representing split masks.

        Returns
        -------
        numpy.ndarray
            cross matrix for samples.

        """
        out = np.empty((mask_arr.shape[1], mask_arr.shape[1]), dtype=np.int_)
        temp_mask = np.empty(
            (mask_arr.shape[1], mask_arr.shape[2]), dtype=np.bool_)
        for j in range(mask_arr.shape[1]):
            for m in range(mask_arr.shape[2]):
                if mask_arr[:, j, m].any():
                    temp_mask[j, m] = True
                else:
                    temp_mask[j, m] = False
        for j1 in prange(mask_arr.shape[1]):
            for j2 in range(mask_arr.shape[1]):
                intersection = (temp_mask[j1] & temp_mask[j2]).sum()
                out[j1, j2] = intersection
        return out

    @staticmethod
    @jit(cache=True)
    def by_norm_splits(mask_arr: np.ndarray) -> np.ndarray:
        """
        Compute the normalized cross matrix for splits.

        Parameters
        ----------
        mask_arr : numpy.ndarray
            3D array representing split masks.

        Returns
        -------
        numpy.ndarray
            Normalized cross matrix for splits.

        """
        out = np.empty((mask_arr.shape[0], mask_arr.shape[0]), dtype=np.float_)
        temp_mask = np.empty(
            (mask_arr.shape[0], mask_arr.shape[2]), dtype=np.bool_)
        for i in range(mask_arr.shape[0]):
            for m in range(mask_arr.shape[2]):
                if mask_arr[i, :, m].any():
                    temp_mask[i, m] = True
                else:
                    temp_mask[i, m] = False
        for i1 in prange(mask_arr.shape[0]):
            for i2 in range(mask_arr.shape[0]):
                intersection = (temp_mask[i1] & temp_mask[i2]).sum()
                union = (temp_mask[i1] | temp_mask[i2]).sum()
                out[i1, i2] = intersection / union
        return out


class Coverage:
    """
    Class representing coverage calculations for mask arrays.

    Attributes
    ----------
    index : pd.Index
        Index associated with the coverage.
    split_arr : np.ndarray
        Two-dimensional splits array.
        First axis represents splits. Second axis represents samples. Elements 
        represent periods. periods must be either a slice, a sequence of 
        indices, a mask, or a callable that returns such.
    n_splits : int
        Number of splits.
    n_samples : int
        Number of samples.

    Methods
    -------
    get_iter_split_mask_arrs
        Generate two-dimensional boolean arrays, one per split.
    get_iter_split_masks
        Generate boolean DataFrames, one per split.
    get_iter_sample_mask_arrs
        Generate two-dimensional boolean arrays, one per sample.
    stack_mask
        Stack mask arrays into a Series or DataFrame.
    get_mask
        Get a boolean Series or DataFrame representing the mask.

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
    def split_labels(self) -> pd.Index:
        """Split labels."""
        return self._split_labels

    @property
    def sample_labels(self) -> pd.Index:
        """Sample labels."""
        return self._sample_labels

    @property
    def index(self) -> pd.Index:
        """
        Get the index.

        Returns
        -------
        Index
            The index.

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
        Get the number of splits.

        Returns
        -------
        int
            The number of splits.

        """
        return self._n_splits

    @property
    def n_samples(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            The number of samples.

        """
        return self._n_samples

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

    def get_iter_split_mask_arrs(self) -> tp.Generator[np.ndarray, None, None]:
        """
        Generate two-dimensional boolean arrays, one per split.

        Returns
        -------
        generator of np.ndarray
            Each array represents a split's mask where the first axis is samples 
            and the second axis is the index.

        """
        n_splits = self.n_splits
        n_samples = self.n_samples
        for split_idx in range(n_splits):
            out = np.full((n_samples, len(self.index)), False)
            for sample_idx in range(n_samples):
                period = select_period(
                    self.index,
                    self.get_periods(split_idx, sample_idx)
                )
                out[sample_idx, :] = SplitPeriod.get_period_mask(
                    period, index=self.index)
            yield out

    def get_iter_split_masks(self) -> tp.Generator[pd.DataFrame, None, None]:
        """
        Generate boolean DataFrames, one per split.

        Returns
        -------
        generator of pd.DataFrame
            Each DataFrame represents a split's mask with index
            and columns as sample_labels.

        """
        for mask in self.get_iter_split_mask_arrs():
            yield pd.DataFrame(
                np.moveaxis(mask, -1, 0),
                index=self.index,
                columns=self.sample_labels
            )

    def get_iter_sample_mask_arrs(self) -> tp.Generator[np.ndarray, None, None]:
        """
        Generate two-dimensional boolean arrays, one per sample.

        Parameters
        ----------
        index : tp.Union[range, tp.Sequence[tp.Union[str, float, int, complex, bool, object, np.generic]]], optional
            Index for which to generate masks (default is None).

        Returns
        -------
        generator of np.ndarray
            Each array represents a sample's mask where the first axis is splits 
            and the second axis is the index.

        """
        for sample_idx in range(self.n_samples):
            out = np.full((self.n_splits, len(self.index)), False)
            for split_idx in range(self.n_splits):
                period = select_period(
                    self.index,
                    self.get_periods(split_idx, sample_idx)
                )
                out[split_idx, :] = SplitPeriod.get_period_mask(
                    period, self.index)
            yield out

    def get_iter_sample_masks(self) -> tp.Generator[pd.DataFrame, None, None]:
        """
        Generate boolean DataFrames, one per sample.

        Returns
        -------
        generator of pd.DataFrame
            Each DataFrame represents a sample's mask with index
            and columns as split_labels.

        """
        for mask in self.get_iter_sample_mask_arrs():
            yield pd.DataFrame(
                np.moveaxis(mask, -1, 0),
                index=self.index,
                columns=self.split_labels
            )

    def stack_mask(
        self,
        mask_arr: np.ndarray,
        split_labels: pd.Index,
        sample_labels: pd.Index,
        squeeze_one_split: bool = True,
        squeeze_one_sample: bool = True,
    ) -> tp.Union[pd.Series, pd.DataFrame]:
        """
        Stack mask arrays into a Series or DataFrame.

        Parameters
        ----------
        mask_arr : np.ndarray
            Mask array to stack.
        split_labels : pd.Index
            Labels for splits.
        sample_labels : pd.Index
            Labels for samples.
        squeeze_one_split : bool, optional
            Whether to squeeze if there's only one split label. It defaults to 
            True.
        squeeze_one_sample : bool, optional
            Whether to squeeze if there's only one sample label. It defaults to 
            True.

        Returns
        -------
        SeriesFrame
            Series or DataFrame representing the stacked mask.

        """
        out = np.moveaxis(mask_arr, -1, 0).reshape((len(self.index), -1))
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_sample = len(sample_labels) == 1 and squeeze_one_sample
        if one_split and one_sample:
            return pd.Series(out[:, 0], index=self.index)
        if one_split:
            return pd.DataFrame(out, index=self.index, columns=sample_labels)
        if one_sample:
            return pd.DataFrame(out, index=self.index, columns=split_labels)

        labels = combine_index((split_labels, sample_labels))
        return pd.DataFrame(out, index=self.index, columns=labels)

    def get_mask(
        self,
        squeeze_one_split: bool = True,
        squeeze_one_sample: bool = True,
    ) -> tp.Union[pd.Series, pd.DataFrame]:
        """
        Get a boolean Series or DataFrame representing the mask.

        Parameters
        ----------
        squeeze_one_split : bool, optional
            Whether to squeeze if there's only one split label. It defaults to 
            True.
        squeeze_one_sample : bool, optional
            Whether to squeeze if there's only one sample label. It defaults to 
            True.

        Returns
        -------
        SeriesFrame
            Boolean Series or DataFrame where index is `.index` and columns are 
            splits stacked together.

        """
        return self.stack_mask(
            self.get_mask_arr(),
            self.split_labels,
            self.sample_labels,
            squeeze_one_split,
            squeeze_one_sample
        )

    @cache
    def get_mask_arr(self) -> np.ndarray:
        """
        Get a three-dimensional boolean array with splits.

        Returns
        -------
        SplitsMask
            Boolean array where the first axis represents splits, the second 
            axis represents samples, and the third axis represents the index.

        """
        arr = self.get_iter_split_mask_arrs()
        return np.array(list(arr))

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
    ) -> float:
        """
        Get the coverage of the entire mask.

        Parameters
        ----------
        overlapping : bool, optional
            If True, return the number of overlapping True values.
        normalize : bool, optional
            If True, return the number of True values relative to the length of 
            the index.

        Note
        ----
        If `overlapping` and `normalize` are True, returns the number of 
        overlapping True values relative to the total number of True values.

        Returns
        -------
        float
            Coverage value based on the specified parameters.
        """
        mask = self.get_mask_arr()
        cover_any = mask.any(axis=(0, 1))
        if overlapping:
            cover_over = (mask.sum(axis=(0, 1)) > 1)
            if normalize:
                return cover_over.sum() / cover_any.sum()
            return cover_over.sum()

        if normalize:
            return cover_any.mean()
        return cover_any.sum()

    def get_split_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        squeeze_one_split: bool = True,
    ) -> tp.Union[tp.Any, pd.Series]:
        """
        Get the coverage of each split mask.

        Parameters
        ----------
        overlapping : bool, optional
            If True, return the number of overlapping True values between samples 
            in each split.
        normalize : bool, optional
            If True, return the number of True values in each split relative to 
            the length of the index.
        relative : bool, optional
            If True, return the number of True values in each split relative to 
            the total number of True values across all splits.
        squeeze_one_split : bool, optional
            If True and there is only one split, return a scalar instead of a 
            Series.

        Returns
        -------
        tp.Union[tp.Any, pd.Series]
            Coverage values for each split.
        """
        mask = self.get_mask_arr()
        arr_sum = mask.sum(axis=1)
        arr_any = mask.any(axis=1)
        if overlapping:
            if normalize:
                coverage = (arr_sum > 1).sum(axis=1) / arr_any.sum(axis=1)
            else:
                coverage = (arr_sum > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = (
                        arr_any.sum(axis=1) / mask.any(axis=(0, 1)).sum()
                    )
                else:
                    coverage = arr_any.mean(axis=1)
            else:
                coverage = arr_any.sum(axis=1)

        one_split = len(self.split_labels) == 1 and squeeze_one_split

        if one_split:
            return coverage[0]

        return pd.Series(coverage, index=self.split_labels, name="split_coverage")

    def get_sample_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        squeeze_one_sample: bool = True,
    ) -> tp.Union[tp.Any, pd.Series]:
        """
        Get the coverage of each sample mask.

        Parameters
        ----------
        overlapping : bool, optional
            If True, return the number of overlapping True values between 
            splits in each sample.
        normalize : bool, optional
            If True, return the number of True values in each sample relative to 
            the length of the index.
        relative : bool, optional
            If True, return the number of True values in each sample relative to 
            the total number of True values across all samples.
        squeeze_one_sample : bool, optional
            If True and there is only one sample, return a scalar instead of a 
            Series.

        Returns
        -------
        tp.Union[tp.Any, pd.Series]
            Coverage values for each sample.
        """
        mask = self.get_mask_arr()
        arr_sum = mask.sum(axis=0)
        arr_any = mask.any(axis=0)
        if overlapping:
            if normalize:
                coverage = (arr_sum > 1).sum(axis=1) / arr_any.sum(axis=1)
            else:
                coverage = (arr_sum > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = (
                        arr_any.sum(axis=1) / mask.any(axis=(0, 1)).sum()
                    )

                else:
                    coverage = arr_any.mean(axis=1)
            else:
                coverage = arr_any.sum(axis=1)

        if len(self.sample_labels) == 1 and squeeze_one_sample:
            return coverage[0]

        return pd.Series(coverage, index=self.sample_labels, name="sample_coverage")

    def get_period_coverage(
        self,
        normalize: bool = True,
        relative: bool = False,
        squeeze_one_split: bool = True,
        squeeze_one_sample: bool = True,
    ) -> tp.Union[tp.Any, pd.Series]:
        """
        Get the coverage of each period.

        Parameters
        ----------
        normalize : bool, optional
            If True, return the number of True values in each range relative to 
            the length of the index.
        relative : bool, optional
            If True, return the number of True values in each range relative to 
            the total number of True values in its split.
        squeeze_one_split : bool, optional
            If True and there is only one split, return a scalar instead of a 
            Series.
        squeeze_one_sample : bool, optional
            If True and there is only one sample, return a scalar instead of a 
            Series.

        Returns
        -------
        tp.Union[tp.Any, pd.Series]
            Coverage values for each period.
        """
        mask = self.get_mask_arr()
        if normalize:
            if relative:
                coverage = (
                    mask.sum(axis=2) /
                    mask.any(axis=1).sum(axis=1)[:, None]
                )
            else:
                coverage = mask.sum(axis=2) / mask.shape[2]
        else:
            coverage = mask.sum(axis=2)

        one_split = len(self.split_labels) == 1 and squeeze_one_split
        one_sample = len(self.sample_labels) == 1 and squeeze_one_sample

        if one_split and one_sample:
            return coverage[0]

        if one_split:
            return pd.Series(
                coverage.flatten(),
                index=self.sample_labels,
                name="period_coverage"
            )

        if one_sample:
            return pd.Series(
                coverage.flatten(),
                index=self.split_labels,
                name="period_coverage"
            )
        return pd.Series(
            coverage.flatten(),
            index=combine_index((self.split_labels, self.sample_labels)),
            name="period_coverage"
        )

    def get_overlap_matrix(
        self,
        by: str = "split",
        normalize: bool = True,
        squeeze_one_split: bool = True,
        squeeze_one_sample: bool = True,
    ) -> pd.DataFrame:
        """
        Get the cross between each pair of ranges.

        Parameters
        ----------
        by : str, optional
            Specify the method of cross calculation 'split', 'sample', or 
            'range'.
        normalize : bool, optional
            If True, return the cross values normalized.
        squeeze_one_split : bool, optional
            If True and there is only one split, return a scalar instead of a 
            DataFrame.
        squeeze_one_sample : bool, optional
            If True and there is only one sample, return a scalar instead of a 
            DataFrame.

        Returns
        -------
        pd.DataFrame
            Matrix of cross values between pairs based on the specified 
            method.
        """
        mask = self.get_mask_arr()
        one_split = len(self.split_labels) == 1 and squeeze_one_split
        one_sample = len(self.sample_labels) == 1 and squeeze_one_sample
        if by.lower() == "split":
            if normalize:
                func = Overlap.by_norm_splits.py_func

            else:
                func = Overlap.by_splits.py_func

            cross_matrix = func(mask)
            if one_split:
                return cross_matrix[0, 0]
            index = self.split_labels

        elif by.lower() == "sample":
            if normalize:
                func = Overlap.by_norm_samples.py_func

            else:
                func = Overlap.by_samples.py_func

            cross_matrix = func(mask)
            if one_sample:
                return cross_matrix[0, 0]
            index = self.sample_labels

        elif by.lower() == "period":
            if normalize:
                func = Overlap.by_norm_period.py_func

            else:
                func = Overlap.by_period.py_func

            cross_matrix = func(mask)
            if one_split and one_sample:
                return cross_matrix[0, 0]

            if one_split:
                index = self.sample_labels
            elif one_sample:
                index = self.split_labels
            else:
                index = combine_index((self.split_labels, self.sample_labels))
        else:
            raise ValueError(f"Invalid option by='{self.by}'")
        return pd.DataFrame(cross_matrix, index=index, columns=index)
