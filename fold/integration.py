import numpy as np
import pandas as pd
import typing as tp

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import indexable

from fold.model_selection.base import BaseModel


class WrapperSplit(BaseCrossValidator):
    """
    Scikit-learn compatible cross-validator.

    Parameters
    ----------
    model : BaseModel
        Base factory method.
    **kwargs
        Keyword arguments passed to the factory method.

    Attributes
    ----------
    model : BaseModel
        Base factory method.
    kwargs : tp.Dict[str, tp.Any]
        Keyword arguments passed to the factory method.

    """

    def __init__(self, model: BaseModel, **kwargs):
        self._model = model
        self._kwargs = kwargs

    @property
    def model(self) -> BaseModel:
        """
        Model.

        Returns
        -------
        BaseModel
            Base factory method.

        """
        return self._model

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        """
        Keyword arguments passed to the factory method.

        Returns
        -------
        tp.Dict[str, tp.Any]
            Keyword arguments passed to the factory method.

        """
        return self._kwargs

    def get_model(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> BaseModel:
        """
        Get factory method instance.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Returns
        -------
        BaseModel
            The model instance.

        """
        X, y, groups = indexable(X, y, groups)
        try:
            if isinstance(X, pd.Index):
                index = X
            elif hasattr(X, "index"):
                index = X.index
            else:
                raise ValueError("Must provide object index")

        except ValueError:
            index = pd.RangeIndex(stop=len(X))

        model = self.model(index, **self.kwargs)
        if model.n_samples != 2:
            raise ValueError(
                "Number of sets in the model must be 2: train and test"
            )
        return model

    def _iter_masks(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[tp.Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates boolean masks corresponding to train and test sets.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        tp.Tuple[np.ndarray, np.ndarray]
            Boolean masks corresponding to train and test sets.

        """
        model = self.get_model(X=X, y=y, groups=groups)
        for mask_arr in model.get_iter_split_mask_arrs():
            yield mask_arr[0], mask_arr[1]

    def _iter_train_masks(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[np.ndarray, None, None]:
        """
        Generates boolean masks corresponding to train sets.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        np.ndarray
            Boolean masks corresponding to train sets.

        """
        for train_mask_arr, _ in self._iter_masks(X=X, y=y, groups=groups):
            yield train_mask_arr

    def _iter_test_masks(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[np.ndarray, None, None]:
        """
        Generates boolean masks corresponding to test sets.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        np.ndarray
            Boolean masks corresponding to test sets.

        """
        for _, test_mask_arr in self._iter_masks(X=X, y=y, groups=groups):
            yield test_mask_arr

    def _iter_indices(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[tp.Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates integer indices corresponding to train and test sets.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        tp.Tuple[np.ndarray, np.ndarray]
            Integer indices corresponding to train and test sets.

        """
        for train_mask_arr, test_mask_arr in self._iter_masks(X=X, y=y, groups=groups):
            yield np.flatnonzero(train_mask_arr), np.flatnonzero(test_mask_arr)

    def _iter_train_indices(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[np.ndarray, None, None]:
        """
        Generates integer indices corresponding to train sets.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        np.ndarray
            Integer indices corresponding to train sets.

        """
        for train_indices, _ in self._iter_indices(X=X, y=y, groups=groups):
            yield train_indices

    def _iter_test_indices(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[np.ndarray, None, None]:
        """
        Generates integer indices corresponding to test sets.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        np.ndarray
            Integer indices corresponding to test sets.

        """
        for _, test_indices in self._iter_indices(X=X, y=y, groups=groups):
            yield test_indices

    def get_n_splits(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Returns
        -------
        int
            Number of splitting iterations.

        """
        model = self.get_model(X=X, y=y, groups=groups)
        return model.n_splits

    def split(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Generator[tp.Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : tp.Any, optional
            Features (default is None).
        y : tp.Any, optional
            Response (default is None).
        groups : tp.Any, optional
            Groups (default is None).

        Yields
        ------
        tp.Tuple[np.ndarray, np.ndarray]
            Indices of train and test sets.

        """
        return self._iter_indices(X=X, y=y, groups=groups)
