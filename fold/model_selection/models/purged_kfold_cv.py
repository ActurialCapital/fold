from itertools import combinations
import numpy as np
import pandas as pd
import typing as tp

from fold.model_selection.base import BasePurgedCV
from fold.utils.datetime import to_timedelta


__all__ = ["PurgedKFoldCV"]


class PurgedKFoldCV(BasePurgedCV):
    """
    Purged and embargoed combinatorial cross-validation.

    This class performs a cross-validation where samples are split into 
    `n_folds` folds. In each round of cross-validation, `n_test_folds` folds 
    are used for testing while the remaining folds are used for training. 
    Each sample must have associated prediction and evaluation times to ensure 
    that training and test sets do not overlap in these intervals. An embargo 
    period can also be specified to avoid contamination due to temporal 
    correlation.

    Parameters
    ----------
    n_folds : int, optional
        Number of folds. Default is 10.
    n_test_folds : int, optional
        Number of folds used for the test set. Default is 2.
    purge_td : str, int, float, or pd.Timedelta, optional
        Time duration for purging. Default is 0.
    embargo_td : str, int, float, or pd.Timedelta, optional
        Embargo time duration. Default is 0.
    """

    def __init__(
        self,
        n_folds: tp.Optional[int] = 10,
        n_test_folds: tp.Optional[int] = 2,
        purge_td: tp.Optional[str | int | float | pd.Timedelta] = 0,
        embargo_td: tp.Optional[str | int | float | pd.Timedelta] = 0,
    ):
        BasePurgedCV.__init__(self, n_folds=n_folds, purge_td=purge_td)

        if n_test_folds > self.n_folds - 1:
            raise ValueError("n_test_folds must be between 1 and n_folds - 1")
        self._n_test_folds = n_test_folds
        self._embargo_td = to_timedelta(embargo_td)

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
    def embargo_td(self) -> pd.Timedelta:
        """
        Returns the embargo period.

        Returns
        -------
        pd.Timedelta
            Embargo time duration.
        """
        return self._embargo_td

    def embargo(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        test_fold_end: int,
    ) -> np.ndarray:
        """
        Apply the embargo procedure to part of the train set.

        This method drops train set samples whose prediction time occurs within 
        the embargo period of the test set sample evaluation times. The embargo 
        is applied to the training set immediately following the end of the 
        test set.

        Parameters
        ----------
        train_indices : np.ndarray
            Indices of the training samples.
        test_indices : np.ndarray
            Indices of the test samples.
        test_fold_end : int
            End index of the test fold.

        Returns
        -------
        np.ndarray
            Updated training indices after applying the embargo.
        """
        last_test_eval_time = ( 
            self.eval_times
            .iloc[test_indices[test_indices <= test_fold_end]]
            .max()
        )
        min_train_index = len(
            self.pred_times[
                self.pred_times <= last_test_eval_time + self.embargo_td
            ]
        )
        if min_train_index < self.indices.shape[0]:
            allowed_indices = np.concatenate(
                (self.indices[:test_fold_end], self.indices[min_train_index:])
            )
            train_indices = np.intersect1d(train_indices, allowed_indices)
        return train_indices

    def compute_train_set(
        self,
        test_fold_bounds: tp.List[tp.Tuple[int, int]],
        test_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the indices of the samples in the training set.

        Parameters
        ----------
        test_fold_bounds : list of tuple of int
            List of bounds for the test folds.
        test_indices : np.ndarray
            Indices of the test samples.

        Returns
        -------
        np.ndarray
            Indices of the training samples.
        """
        train_indices = np.setdiff1d(self.indices, test_indices)
        for test_fold_start, test_fold_end in test_fold_bounds:
            train_indices = self.purge(train_indices, test_fold_start, test_fold_end)
            train_indices = self.embargo(train_indices, test_indices, test_fold_end)
        
        return train_indices

    def compute_test_set(
        self,
        fold_bound_list: tp.List[tp.Tuple[int, int]],
    ) -> tp.Tuple[tp.List[tp.Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.

        Parameters
        ----------
        fold_bound_list : list of tuple of int
            List of bounds for each fold.

        Returns
        -------
        tuple
            A tuple containing the test fold bounds and test indices.
        """
        test_indices = np.empty(0)
        
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            
            test_indices = np.union1d(
                test_indices, 
                self.indices[fold_start:fold_end]
            ).astype(int)
            
        return test_fold_bounds, test_indices

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
        fold_bounds = [
            (fold[0], fold[-1] + 1)
            for fold in np.array_split(self.indices, self.n_folds)
        ]
        selected_fold_bounds = list(combinations(fold_bounds, self.n_test_folds))
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices
