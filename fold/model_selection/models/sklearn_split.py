import numpy as np
from sklearn.model_selection import BaseCrossValidator
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool
from fold.utils.datetime import prepare_dt_index


class SklearnFold(BaseModel):
    """
    Blends `fold` with `scikit-learn` cross-validator capabilities.

    Parameters
    ----------
    index : range
        The indices or data points to be split.
    sk_model : BaseCrossValidator
        An instance of a scikit-learn splitter, which must be a subclass of
        `sklearn.model_selection.BaseCrossValidator`.
    groups : str, float, int, bool, np.ndarray, optional
        Group labels for the samples, used for grouped cross-validation.
    constraints : BaseTool, optional
        Constraints to apply on the splits.
    split_labels : range, optional
        Labels for each split.
    sample_labels : list of str, optional
        Labels for the entire set of splits. Default is ["train", "test"].

    Notes
    -----
    This class creates a `FoldCV` instance for data splits based on a 
    scikit-learn cross-validator (`BaseCrossValidator`) provided as 
    `sk_model`.

    The `index` parameter is processed as a datetime index using 
    `prepare_dt_index`. The splitter's `split` method generates indices 
    for the splits.

    Examples
    --------
    >>> from sklearn.model_selection import KFold
    >>> index = pd.date_range("2020", "2021", freq="D")
    >>> sk_model = KFold(n_splits=2)
    >>> cv = SklearnFold(index, sk_model)
    >>> print(cv.get_bounds(index_bounds=True))
    bound               start        end
    split sample                        
    0     sample_0 2020-07-03 2021-01-02
          sample_1 2020-01-01 2020-07-03
    1     sample_0 2020-01-01 2020-07-03
          sample_1 2020-07-03 2021-01-02
    """

    def __init__(
        self,
        index: range,
        sk_model: BaseCrossValidator,
        groups: tp.Optional[str | float | int | bool | np.ndarray] = None,
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
    ):
        index = prepare_dt_index(index)
        if not issubclass(type(sk_model), BaseCrossValidator):
            raise AssertionError(
                f"Argument `sk_model` must be of type `BaseCrossValidator`, not {type(sk_model)}"
            )

        indices_generator = sk_model.split(
            np.arange(len(index))[:, None], 
            groups=groups
        )

        super().__init__(
            index,
            splits=list(indices_generator),
            constraints=constraints,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )
