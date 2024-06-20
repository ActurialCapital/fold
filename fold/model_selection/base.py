import abc
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.config import Config, UpdateConfig
from fold.model_selection.duration import Duration
from fold.model_selection.coverage import Coverage
from fold.tools import (
    substitute,
    BaseTool,
    SelectPosition,
    PeriodTransformer,
    SplitPeriod,
    select_period
)
from fold.utils.indexing import (
    combine_index,
    select_index,
    create_split_labels,
    create_sample_labels
)
from fold.utils.resampler import CustomResampler
from fold.utils.datetime import prepare_dt_index, to_timedelta
from fold.utils import checks


__all__ = ["BaseModel", "BasePurgedCV"]


BaseModelT = tp.TypeVar("BaseModel", bound="BaseModel")


class BaseModel(Duration, Coverage):
    """
    A base model combining Duration and Coverage functionalities.

    Parameters
    ----------
    index : range
        Index associated with the model.
    splits : tp.Any
        Splits used in the model.
    fix_ranges : bool, optional
        Whether to fix periods (default is True).
    backwards : bool, optional
        Whether to calculate periods backwards (default is False).
    allow_zero_len : bool, optional
        Whether to allow zero-length periods (default is False).
    range_format : str, optional
        Format of the range (default is None).
    freq : tp.Optional[str | int | Offset | pd.Timedelta], optional
        Frequency of the index (default is None).
    constraints : BaseTool, optional
        Constraints applied (default is None).
    split_labels : range, optional
        Labels for splits (default is None).
    sample_labels : range, optional
        Labels for samples (default is None).

    """

    def __init__(
        self,
        index: range,
        splits: tp.Any,
        fix_ranges: tp.Optional[bool] = True,
        backwards: tp.Optional[bool] = False,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = None,
        constraints: tp.Optional[BaseTool] = None,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
    ):
        index = prepare_dt_index(index)

        updated_periods = []
        updated_constraints = []
        for i, split in enumerate(splits):
            config = UpdateConfig(
                split,
                index,
                fix_ranges=fix_ranges,
                backwards=backwards,
                allow_zero_len=allow_zero_len,
                range_format=range_format,
                freq=freq
            )

            if constraints is not None:
                is_valid_period = substitute(
                    constraints,
                    dict(index=index, i=i, split=config.updated_period),
                    eval_id="constraints"
                )

                if not is_valid_period:
                    updated_constraints.append(i)
                    continue

            updated_periods.append(config.updated_period)

        if len(updated_periods) == 0:
            raise ValueError("Must provide at least one range")

        splits_arr = np.asarray(updated_periods, dtype=object)

        self._split_labels = create_split_labels(
            splits_arr,
            updated_constraints,
            split_labels
        )
        self._sample_labels = create_sample_labels(
            splits_arr,
            sample_labels
        )
        self._ndim = 2 if len(self._sample_labels) > 1 else config.ndim
        self._index = index
        self._splits_arr = splits_arr
        
        super().__init__(
            index=self.index,
            splits_arr=self.splits_arr,
            n_splits=self.n_splits,
            n_samples=self.n_samples,
            split_labels=self.split_labels,
            sample_labels=self.sample_labels
        )
        
    @property
    def split_labels(self) -> pd.Index:
        """
        Get split labels.

        Returns
        -------
        Index
            The split labels.

        """
        return self._split_labels

    @property
    def sample_labels(self) -> pd.Index:
        """
        Get sample labels.

        Returns
        -------
        Index
            The sample labels.

        """
        return self._sample_labels

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions.

        Returns
        -------
        int
            The number of dimensions.

        """
        return self._ndim

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
        return self.splits_arr.shape[0]

    @property
    def n_samples(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            The number of samples.

        """
        return self.splits_arr.shape[1]

    @property
    def splits(self) -> pd.DataFrame:
        """
        Get the splits as a DataFrame.

        Returns
        -------
        Frame
            The splits as a DataFrame.

        """
        model = Config(
            self.split_labels,
            self.sample_labels,
            self.ndim
        )
        return model.apply(self._splits_arr)

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

    def select_period(
        self,
        split: tp.Optional[tp.Any] = None,
        sample: tp.Optional[tp.Any] = None
    ) -> tp.Any:
        return select_period(
            self.index,
            self.get_periods(split, sample)
        )

    @staticmethod
    def _map_period_object(
        obj: tp.Any,
        period: tp.Any,
        index: range,
        remap_to_obj: tp.Optional[bool] = True,
        obj_index: tp.Optional[range] = None,
        obj_freq: tp.Optional[str | int | Offset | pd.Timedelta] = None,
        silence_warnings: tp.Optional[bool] = False,
        freq: tp.Optional[str | int | Offset | pd.Timedelta] = None,
        **ready_range_kwargs,
    ) -> tp.Any:
        """
        Get a range that is ready to be mapped into an array-like object.

        Note
        ----
        If the object is Pandas-like and `obj_index` is not None, searches for 
        an index in the object. Once found, get the range that maps to the 
        object index. Finally, convert the range into the one that can be used 
        directly in indexing.

        Parameters:
        -----------
        obj : tp.Any
            The object to which the range will be mapped.
        period : tp.Any
            The range to prepare for mapping.
        remap_to_obj : bool, default=True
            Whether to remap the range to the object's index if possible.
        obj_index : tp.Optional[range], default=None
            The index of the object, if known.
        obj_freq : tp.Optional[str | int | Offset | pd.Timedelta], default=None
            The frequency of the object's index.
        silence_warnings : bool, default=False
            Whether to silence warnings during resampling.
        index : tp.Optional[range], default=None
            The index associated with the original range.
        freq : tp.Optional[str | int | Offset | pd.Timedelta], default=None
            The frequency of the original index.

        Returns:
        --------
        Any
            The prepared range or metadata along with the range, depending on 
            the settings.

        Raises:
        -------
        ValueError
            If the object index cannot be determined.
        """
        if (
            remap_to_obj and
            isinstance(obj, (pd.Index, pd.Series, pd.DataFrame)) or
            obj_index is not None
        ):
            if obj_index is None:
                if isinstance(obj, pd.Index):
                    obj_index = obj

                elif hasattr(obj, "index"):
                    obj_index = obj.index

                else:
                    raise ValueError("Must provide object index")

            if obj_index is None:
                raise ValueError("Must provide target index")

            obj_index = prepare_dt_index(obj_index)

            if index.equals(obj_index):
                target_range = period

            else:
                mask = SplitPeriod.get_period_mask(period, index)
                resampler = CustomResampler(
                    source_index=index,
                    target_index=obj_index,
                    source_freq=freq,
                    target_freq=obj_freq,
                )
                target_range = resampler.resample_source_mask(
                    mask,
                    silence_warnings=silence_warnings
                )

        else:
            obj_index = index
            target_range = period

        return PeriodTransformer(target_range, obj_index).period

    def train_test_split(
        self,
        obj: tp.Any,
        split: tp.Optional[tp.Any] = None,
        sample: tp.Optional[tp.Any] = None,
        squeeze_one_split: tp.Optional[bool] = True,
        squeeze_one_sample: tp.Optional[bool] = True,
        remap_to_obj: tp.Optional[bool] = True,
        obj_index: tp.Optional[range] = None,
        obj_freq: tp.Optional[str | int | Offset | pd.Timedelta] = None,
        range_format: tp.Optional[str] = "slice_or_any",
        right_inclusive: tp.Optional[bool] = False,
        silence_warnings: tp.Optional[bool] = False,
        freq: tp.Optional[str | int | Offset | pd.Timedelta] = None,
    ) -> tp.Any:
        """
        Take all periods from an array-like object and optionally column-stack 
        them.

        Parameters
        ----------
        obj : tp.Any
            The object containing periods.
        split : tp.Optional[tp.Any], optional
            Selection for splits (default is None).
        sample : tp.Optional[tp.Any], optional
            Selection for samples (default is None).
        squeeze_one_sample : bool, optional
            Whether to squeeze the result if only one split is selected 
            (default is True).
        squeeze_one_sample : bool, optional
            Whether to squeeze the result if only one sample is selected 
            (default is True).
        remap_to_obj : bool, optional
            Whether to remap the range to the object's index if possible 
            (default is True).
        obj_index : range, optional
            The index of the object, if known (default is None).
        obj_freq : tp.Optional[str | int | Offset | pd.Timedelta], optional
            The frequency of the object's index (default is None).
        range_format : str, optional
            Format of the range (default is "slice_or_any").
        right_inclusive : bool, optional
            Whether the right endpoint is inclusive (default is False).
        silence_warnings : bool, optional
            Whether to silence warnings during resampling (default is False).
        freq : tp.Optional[str | int | Offset | pd.Timedelta], optional
            The frequency of the original index (default is None).

        Returns
        -------
        tp.Any
            The result from taking periods from the object.

        """
        split_index = self.split_index(split)
        sample_index = self.sample_index(sample)

        n_splits = len(split_index)
        n_samples = len(sample_index)

        split_labels = (
            self.split_labels[split_index]
            if split is not None
            else self.split_labels
        )
        sample_labels = (
            self.sample_labels[sample_index]
            if sample is not None
            else self.sample_labels
        )

        def _get_range_meta(i, j):
            split_idx, sample_idx = split_index[i], sample_index[j]

            period = select_period(
                self.index,
                self.get_periods(split_idx, sample_idx)
            )

            transformer = PeriodTransformer(
                period,
                self.index,
                range_format=range_format
            )

            period_object = self._map_period_object(
                obj,
                transformer.period,
                self.index,
                remap_to_obj=remap_to_obj,
                obj_index=obj_index,
                obj_freq=obj_freq,
                range_format=range_format,
                silence_warnings=silence_warnings,
                freq=freq,
            )

            if isinstance(obj, BaseTool):
                params = dict(
                    split_idx=split_idx,
                    sample_idx=sample_idx,
                    period=period_object,
                    range_meta=transformer.model_output,
                )
                obj_slice = substitute(obj, params, eval_id="take_range")

            else:
                if isinstance(obj, (pd.Series, pd.DataFrame)):
                    obj_slice = obj.iloc[period_object]
                else:
                    obj_slice = obj[period_object]

            return obj_slice

        range_objs = []
        for i in range(n_splits):
            for j in range(n_samples):
                obj_slice = _get_range_meta(i, j)
                range_objs.append(obj_slice)

        if n_splits == 1 and squeeze_one_split and n_samples == 1 and squeeze_one_sample:
            return range_objs[0]

        if n_samples == 1 and squeeze_one_sample:
            keys = split_labels

        elif n_splits == 1 and squeeze_one_split:
            keys = sample_labels

        else:
            keys = combine_index((split_labels, sample_labels))

        return pd.Series(range_objs, index=keys, dtype=object)


class BasePurgedCV:
    """
    Base purged time series cross-validation.

    Time series cross-validation requires each sample has a prediction time, 
    at which the features are used to predict the response, and an evaluation
    time, at which the response is known and the error can be computed. 

    Importantly, it means that unlike in standard sklearn cross-validation, 
    the samples X, response y, `pred_times` and `eval_times` must all be 
    pandas DataFrames/Series having the same index. It is also assumed that 
    the samples are time-ordered with respect to the prediction time.

    Parameters
    ----------
    n_folds : int, optional
        Number of folds (default is 10).
    purge_td : tp.Optional[str | int | Offset | pd.Timedelta] , optional
        Purge period (default is 0).

    Attributes
    ----------
    n_folds : int
        Number of folds.
    purge_td : pd.Timedelta
        Purge period.
    pred_times : tp.Optional[pd.Series]
        Times at which predictions are made.
    eval_times : tp.Optional[pd.Series]
        Times at which the response becomes available and the error can be 
        computed.
    indices : tp.Optional[np.ndarray]
        Indices.

    """

    def __init__(
        self,
        n_folds: int = 10,
        purge_td: tp.Optional[str | int | Offset | pd.Timedelta] = 0
    ):
        self._n_folds = n_folds
        self._pred_times = None
        self._eval_times = None
        self._indices = None
        self._purge_td = to_timedelta(purge_td)

    @property
    def n_folds(self) -> int:
        """
        Number of folds.

        Returns
        -------
        int
            Number of folds.

        """
        return self._n_folds

    @property
    def purge_td(self) -> pd.Timedelta:
        """
        Purge period.

        Returns
        -------
        pd.Timedelta
            Purge period.

        """
        return self._purge_td

    @property
    def pred_times(self) -> tp.Optional[pd.Series]:
        """
        Times at which predictions are made.

        Returns
        -------
        tp.Optional[pd.Series]
            Times at which predictions are made.

        """
        return self._pred_times

    @property
    def eval_times(self) -> tp.Optional[pd.Series]:
        """
        Times at which the response becomes available and the error can be 
        computed.

        Returns
        -------
        tp.Optional[pd.Series]
            Times at which the response becomes available and the error can be 
            computed.

        """
        return self._eval_times

    @property
    def indices(self) -> tp.Optional[np.ndarray]:
        """
        Indices.

        Returns
        -------
        tp.Optional[np.ndarray]
            Indices.

        """
        return self._indices

    def purge(
        self,
        train_indices: np.ndarray,
        test_fold_start: int,
        test_fold_end: int,
    ) -> np.ndarray:
        """
        Purge part of the train sample.

        Given a left boundary index `test_fold_start` of the test sample and a 
        right boundary index `test_fold_end`, this method removes from the 
        train sample all the samples whose evaluation time is posterior to the 
        prediction time of the first test sample after the boundary.

        Parameters
        ----------
        train_indices : np.ndarray
            Training indices.
        test_fold_start : int
            Left boundary index of the test sample.
        test_fold_end : int
            Right boundary index of the test sample.

        Returns
        -------
        np.ndarray
            Purged train indices.

        """
        time_test_fold_start = self.pred_times.iloc[test_fold_start]
        eval_times = self.eval_times + self.purge_td
        train_indices_1 = np.intersect1d(
            train_indices, self.indices[eval_times < time_test_fold_start])
        train_indices_2 = np.intersect1d(
            train_indices, self.indices[test_fold_end:])
        return np.concatenate((train_indices_1, train_indices_2))

    @abc.abstractmethod
    def split(
        self,
        X: pd.Series | pd.DataFrame,
        y: tp.Optional[pd.Series] = None,
        pred_times: tp.Optional[pd.Index | pd.Series] = None,
        eval_times: tp.Optional[pd.Index | pd.Series] = None,
    ):
        """
        Yield the indices of the train and test sets.

        Parameters
        ----------
        X : pd.Series | pd.DataFrame
            Features.
        y : tp.Optional[pd.Series], optional
            Response (default is None).
        pred_times : tp.Optional[pd.Index | pd.Series], optional
            Prediction times (default is None).
        eval_times : tp.Optional[pd.Index | pd.Series], optional
            Evaluation times (default is None).

        Raises
        ------
        AssertionError
            If X is not an instance of pd.Series or pd.DataFrame.

        """
        if not checks.is_instance_of(X, (pd.Series, pd.DataFrame)):
            raise AssertionError(
                "Argument 'X' must be of type pd.Series or pd.DataFrame, not "
                f"{type(X)}"
            )

        if y is not None:
            checks.assert_instance_of(y, pd.Series, arg_name="y")
        if pred_times is None:
            pred_times = X.index
        if isinstance(pred_times, pd.Index):
            pred_times = pd.Series(pred_times, index=X.index)
        else:
            checks.assert_instance_of(
                pred_times, 
                pd.Series, 
                arg_name="pred_times"
            )
            checks.assert_index_equal(
                X.index, 
                pred_times.index, 
                check_names=False
            )
        if eval_times is None:
            eval_times = X.index
        if isinstance(eval_times, pd.Index):
            eval_times = pd.Series(eval_times, index=X.index)
        else:
            checks.assert_instance_of(
                eval_times, pd.Series, arg_name="eval_times")
            checks.assert_index_equal(
                X.index, eval_times.index, check_names=False)

        self._pred_times = pred_times
        self._eval_times = eval_times
        self._indices = np.arange(X.shape[0])
