import typing as tp

from fold.model_selection.base import BaseModelT


def train_test_split(
    obj: tp.Any,
    model: BaseModelT | tp.Callable,
    model_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    train_test_split_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    _model_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    _train_test_split_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    **var_kwargs,
) -> tp.Any:
    """
    Split an index and extract data from an object.

    Parameters
    ----------
    obj : Any
        The object from which to split and extract data.
    model : BaseModelT, callable
        The model object or callable used for splitting data.
    model_kwargs : dict, optional
        Keyword arguments to be passed to the factory method specified by 
        `model` for splitting data.
    train_test_split_kwargs : dict, optional
        Keyword arguments to be passed to the `BaseModel.train_test_split` method 
        for selecting data after splitting.
    _model_kwargs : dict, optional
        Internal keyword arguments used within the model logic.
    _train_test_split_kwargs : dict, optional
        Internal keyword arguments used within the 
        `BaseModel.train_test_split` method.
    **var_kwargs
        Additional keyword arguments. If `model_kwargs` or 
        `train_test_split_kwargs` are not specified, these will be used as 
        `model_kwargs` or `train_test_split_kwargs` respectively. An error will
        be raised if both `model_kwargs` and `train_test_split_kwargs` are 
        provided.

    Returns
    -------
    Any
        The result of applying the `BaseModel.train_test_split` method to the 
        input `obj` with the specified `train_test_split_kwargs`.

    Raises
    ------
    ValueError
        If required arguments are not provided (e.g., `model` is `None`) or 
        when conflicting keyword arguments are supplied (`model_kwargs` and 
        `train_test_split_kwargs` both provided).

    """
    model_kwargs = (
        {}
        if model_kwargs is None
        else dict(model_kwargs)
    )

    train_test_split_kwargs = (
        {}
        if train_test_split_kwargs is None
        else dict(train_test_split_kwargs)
    )

    _model_kwargs = _model_kwargs or {}
    _train_test_split_kwargs = _train_test_split_kwargs or {}

    if len(var_kwargs) > 0:
        if len(model_kwargs) == 0 and len(train_test_split_kwargs) > 0:
            model_kwargs = var_kwargs

        elif len(model_kwargs) > 0 and len(train_test_split_kwargs) == 0:
            train_test_split_kwargs = var_kwargs

        elif len(model_kwargs) == 0 and len(train_test_split_kwargs) == 0:
            train_test_split_kwargs = var_kwargs

        else:
            raise ValueError(
                "Pass keyword arguments as model_kwargs or train_test_split_kwargs"
            )

    if model is None:
        raise ValueError("Must provide a model.")

    for k, v in _train_test_split_kwargs.items():
        if k not in train_test_split_kwargs:
            train_test_split_kwargs[k] = v

    return model.train_test_split(obj, **train_test_split_kwargs)
