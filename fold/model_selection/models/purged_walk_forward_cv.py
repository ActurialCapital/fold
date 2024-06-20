import numpy as np
import pandas as pd
import typing as tp

from fold.model_selection.base import BasePurgedCV


__all__ = ["PurgedWalkForwardCV"]


class PurgedWalkForwardCV(BasePurgedCV):
    """
    Purged walk-forward cross-validation.

    This class performs a walk-forward cross-validation where samples are split 
    into `n_folds` folds. In each round, `n_test_folds` contiguous folds are used 
    for testing, while the training set consists of between `min_train_folds` 
    and `max_train_folds` immediately preceding folds. Each sample must have 
    associated prediction and evaluation times to ensure that training and test 
    sets do not overlap in these intervals. Optionally, samples can be split 
    into folds spanning equal time intervals.

    Parameters
    ----------
    n_folds : int, optional
        Number of folds. Default is 10.
    n_test_folds : int, optional
        Number of folds used for the test set. Default is 1.
    min_train_folds : int, optional
        Minimum number of folds used for the train set. Default is 2.
    max_train_folds : int, optional
        Maximum number of folds used for the train set. If None, it is set to 
        `n_folds - n_test_folds`. Default is None.
    split_by_time : bool, optional
        If True, folds span equal time intervals. Otherwise, they contain 
        approximately equal numbers of samples. Default is False.
    purge_td : str, int, float, or pd.Timedelta, optional
        Time duration for purging. Default is 0.
    """

    def __init__(
        self,
        n_folds: tp.Optional[int] = 10,
        n_test_folds: tp.Optional[int] = 1,
        min_train_folds: tp.Optional[int] = 2,
        max_train_folds: tp.Optional[int] = None,
        split_by_time: tp.Optional[bool] = False,
        purge_td: tp.Optional[str | int | float | pd.Timedelta] = 0,
    ):
        BasePurgedCV.__init__(self, n_folds=n_folds, purge_td=purge_td)

        if n_test_folds >= self.n_folds - 1:
            raise ValueError(
                "n_test_folds must be between 1 and n_folds - 1"
            )

        self._n_test_folds = n_test_folds
        if min_train_folds >= self.n_folds - self.n_test_folds:
            raise ValueError(
                "min_train_folds must be between 1 and n_folds - n_test_folds"
            )

        self._min_train_folds = min_train_folds
        if max_train_folds is None:
            max_train_folds = self.n_folds - self.n_test_folds

        if max_train_folds > self.n_folds - self.n_test_folds:
            raise ValueError(
                "max_train_split must be between 1 and n_folds - n_test_folds"
            )

        self._max_train_folds = max_train_folds
        self._split_by_time = split_by_time
        self._fold_bounds = []

    @property
    def n_test_folds(self) -> int:
        """
        Returns the number of folds used in the test set.

        Returns
        -------
        int
            Number of test folds.
        """
        return self._n_test_folds

    @property
    def min_train_folds(self) -> int:
        """
        Returns the minimum number of folds used in the train set.

        Returns
        -------
        int
            Minimum number of train folds.
        """
        return self._min_train_folds

    @property
    def max_train_folds(self) -> int:
        """
        Returns the maximum number of folds used in the train set.

        Returns
        -------
        int
            Maximum number of train folds.
        """
        return self._max_train_folds

    @property
    def split_by_time(self) -> bool:
        """
        Returns whether the folds span identical time intervals.

        Returns
        -------
        bool
            True if folds span identical time intervals, otherwise False.
        """
        return self._split_by_time

    @property
    def fold_bounds(self) -> tp.List[int]:
        """
        Returns the boundaries of the folds.

        Returns
        -------
        list of int
            Fold boundaries.
        """
        return self._fold_bounds

    def compute_fold_bounds(self) -> tp.List[int]:
        """
        Compute the boundaries of the folds.

        If `split_by_time` is True, folds span equal time intervals.
        Otherwise, they contain approximately equal numbers of samples.

        Returns
        -------
        list of int
            List containing the fold (left) boundaries.
        """
        if self.split_by_time:
            full_time_span = self.pred_times.max() - self.pred_times.min()
            fold_time_span = full_time_span / self.n_folds
            fold_bounds_times = [
                self.pred_times.iloc[0] + fold_time_span * n
                for n in range(self.n_folds)
            ]
            return self.pred_times.searchsorted(fold_bounds_times)

        else:
            return [
                fold[0]
                for fold in np.array_split(self.indices, self.n_folds)
            ]

    def compute_train_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        """
        Compute the indices of the samples in the train set.

        Parameters
        ----------
        fold_bound : int
            The boundary index of the fold.
        count_folds : int
            The number of folds to consider.

        Returns
        -------
        np.ndarray
            Indices of the training samples.
        """
        if count_folds > self.max_train_folds:
            start_train = self.fold_bounds[count_folds - self.max_train_folds]

        else:
            start_train = 0

        train_indices = np.arange(start_train, fold_bound)
        train_indices = self.purge(train_indices, fold_bound, self.indices[-1])
        return train_indices

    def compute_test_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        """
        Compute the indices of the samples in the test set.

        Parameters
        ----------
        fold_bound : int
            The boundary index of the fold.
        count_folds : int
            The number of folds to consider.

        Returns
        -------
        np.ndarray
            Indices of the test samples.
        """
        if self.n_folds - count_folds > self.n_test_folds:
            end_test = self.fold_bounds[count_folds + self.n_test_folds]

        else:
            end_test = self.indices[-1] + 1

        return np.arange(fold_bound, end_test)

    def split(
        self,
        X: pd.Series | pd.DataFrame,
        y: tp.Optional[pd.Series] = None,
        pred_times: tp.Optional[pd.Index | pd.Series] = None,
        eval_times: tp.Optional[pd.Index | pd.Series] = None,
    ) -> tp.Iterable[tp.Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            The input data.
        y : pd.Series, optional
            The target variable. Default is None.
        pred_times : pd.Index or pd.Series, optional
            Prediction times associated with each sample. Default is None.
        eval_times : pd.Index or pd.Series, optional
            Evaluation times associated with each sample. Default is None.

        Yields
        ------
        tuple of np.ndarray
            The training and test indices for each split.
        """
        BasePurgedCV.split(self, X, y, pred_times=pred_times, eval_times=eval_times)
        self._fold_bounds = self.compute_fold_bounds()

        count_folds = 0
        for fold_bound in self.fold_bounds:
            if count_folds < self.min_train_folds:
                count_folds = count_folds + 1
                continue
            
            if self.n_folds - count_folds < self.n_test_folds:
                break
            
            test_indices = self.compute_test_set(fold_bound, count_folds)
            train_indices = self.compute_train_set(fold_bound, count_folds)

            count_folds = count_folds + 1
            yield train_indices, test_indices
