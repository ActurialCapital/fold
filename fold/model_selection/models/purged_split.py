import numpy as np
import pandas as pd
import typing as tp

from fold.model_selection.base import BaseModel, BasePurgedCV
from fold.utils.datetime import prepare_dt_index


class PurgedSplit(BaseModel):
    """
    A class used to represent a PurgedSplit for model selection.

    Parameters
    ----------
    index : range
        The index range to be used for splitting.
    purged_splitter : BasePurgedCV
        An instance of BasePurgedCV that handles the splitting logic.
    pred_times : pd.Index or pd.Series, optional
        Prediction times, by default None.
    eval_times : pd.Index or pd.Series, optional
        Evaluation times, by default None.
    split_labels : range, optional
        Labels for the splits, by default None.
    sample_labels : range, optional
        Labels for the samples, by default None.
    **kwargs : dict
        Additional keyword arguments to be passed to the BaseModel constructor.

    Raises
    ------
    AssertionError
        If `purged_splitter` is not an instance of `BasePurgedCV`.

    Notes
    -----
    This class uses the `prepare_dt_index` function to prepare the datetime 
    index and validates that `purged_splitter` is an instance of `BasePurgedCV`.
    The split indices are generated using the `purged_splitter.split` method.


    """

    def __init__(
        self,
        index: range,
        purged_splitter: BasePurgedCV,
        pred_times: tp.Optional[pd.Index | pd.Series] = None,
        eval_times: tp.Optional[pd.Index | pd.Series] = None,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
        **kwargs,
    ):
        index = prepare_dt_index(index)
        
        if not issubclass(type(purged_splitter), BasePurgedCV):
            raise AssertionError(
                "Argument `purged_splitter` must be of type `BasePurgedCV`, "
                f"not {type(purged_splitter)}"
            )

        if sample_labels is None:
            sample_labels = ["train", "test"]

        indices_generator = purged_splitter.split(
            pd.Series(np.arange(len(index)), index=index),
            pred_times=pred_times,
            eval_times=eval_times,
        )

        super().__init__(
            index,
            list(indices_generator),
            split_labels=split_labels,
            sample_labels=sample_labels,
            **kwargs,
        )
