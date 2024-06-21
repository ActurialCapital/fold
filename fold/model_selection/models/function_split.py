import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import substitute, BaseTool, BasePeriod, SplitPeriod
from fold.utils.datetime import prepare_dt_index, infer_index_freq
from fold.utils import checks


class FunctionSplit(BaseModel):
    """
    Split index with a custom split function.

    In a while-loop, substitutes templates in `split_args` and 
    `split_kwargs` and passes them to `split_func`, which should return 
    either a split (see `new_split` in `.split_range`, also 
    supports a single range if it's not an iterable) or None to abrupt the 
    while-loop.

    If `fix_ranges` is True, the returned split is then converted into a 
    fixed split using `.split_range` and the bounds of its sets are 
    measured using `.get_range_bounds`.

    Each template substitution has the following information:

    * `split_idx`: Current split index, starting at 0
    * `splits`: Nested list of splits appended up to this point
    * `bounds`: Nested list of bounds appended up to this point
    * `prev_start`: Left bound of the previous split
    * `prev_end`: Right bound of the previous split

    Parameters
    ----------
    index : range
        The range of the index to be split.
    split_func : Callable
        The custom function to perform the split.
    split_args : Tuple[Any, ...], optional
        Arguments to pass to the split function, by default None.
    split_kwargs : Dict[str, Any], optional
        Keyword arguments to pass to the split function, by default None.
    fix_ranges : bool, optional
        Whether to convert the split into a fixed split, by default True.
    split : int, float, slice, BaseTool, BasePeriod, optional
        The specific split to apply, by default None.
    allow_zero_len : bool, optional
        Whether to allow zero-length splits, by default False.
    range_format : str, optional
        Format for the range, by default None.
    freq : str, int, float, Offset, pd.Timedelta, optional
        Frequency of the index, by default "auto".
    index_bounds : bool, optional
        Whether to use index bounds, by default False.
    right_inclusive : bool, optional
        Whether the right bound is inclusive, by default False.
    constraints : BaseTool, optional
        Constraints for the splits, by default None.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the splits, by default None.
    sample_labels : range, optional
        Labels for the samples, by default None.

    Raises
    ------
    ValueError
        If the split function returns more than one range when `split` is provided.
    ValueError
        If all splits do not have the same number of sets.

    Examples
    --------
    >>> from fold.tools import Key
    >>> def split_func(index, prev_start):
    ...     if prev_start is None:
    ...         prev_start = index[0]
    ...         new_start = prev_start + pd.offsets.MonthBegin(1)
    ...         new_end = new_start + pd.DateOffset(years=1)
    ...     if new_end > index[-1] + index.freq:
    ...         return None
    ...     return [
    ...         slice(new_start, new_start + pd.offsets.MonthBegin(9)),
    ...         slice(new_start + pd.offsets.MonthBegin(9), new_end)
    ...     ]
    >>> model = FunctionSplit(
    ...     data.index,
    ...     split_func=split_func,
    ...     split_args=(Key("index"), Key("prev_start")),
    ...     index_bounds=True,
    ...     sample_labels=["IS", "OOS"],
    ...     fix_ranges=True
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound             start        end
    split sample                      
    0     IS     2010-11-01 2011-08-01
          OOS    2011-08-01 2011-11-01
    1     IS     2010-12-01 2011-09-01
          OOS    2011-09-01 2011-12-01
    2     IS     2011-01-01 2011-10-01
                    ...        ...
    149   OOS    2024-01-01 2024-04-01
    150   IS     2023-05-01 2024-02-01
          OOS    2024-02-01 2024-05-01
    151   IS     2023-06-01 2024-03-01
          OOS    2024-03-01 2024-06-01
    ...
    """

    def __init__(
        self,
        index: range,
        split_func: tp.Callable,
        split_args: tp.Tuple[tp.Any, ...] = None,
        split_kwargs: tp.Dict[str, tp.Any] = None,
        fix_ranges: tp.Optional[bool] = True,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        index_bounds: tp.Optional[bool] = False,
        right_inclusive: tp.Optional[bool] = False,
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None,
    ):

        index = prepare_dt_index(index)

        try:
            freq = infer_index_freq(index, freq, allow_numeric=False)
        except Exception:
            freq = None
        split_args = split_args or ()
        split_kwargs = split_kwargs or {}

        splits = []
        bounds = []
        split_idx = 0
        n_sets = None
        while True:
            _template_context = dict(
                split_idx=split_idx,
                splits=splits,
                bounds=bounds,
                prev_start=bounds[-1][0][0] if len(bounds) > 0 else None,
                prev_end=bounds[-1][-1][1] if len(bounds) > 0 else None,
                index=index,
                fix_ranges=fix_ranges,
                split_args=split_args,
                split_kwargs=split_kwargs,
                backwards=backwards,
                allow_zero_len=allow_zero_len,
                range_format=range_format,
                freq=freq,
                index_bounds=index_bounds,
                right_inclusive=right_inclusive,
            )
            _split_func = substitute(
                split_func,
                _template_context,
                eval_id="split_func"
            )
            _split_args = substitute(
                split_args,
                _template_context,
                eval_id="split_args"
            )
            _split_kwargs = substitute(
                split_kwargs,
                _template_context,
                eval_id="split_kwargs"
            )
            new_split = _split_func(*_split_args, **_split_kwargs)
            if new_split is None:
                break

            if not checks.is_iterable(new_split):
                new_split = (new_split,)

            if fix_ranges or split is not None:
                model = SplitPeriod(
                    period=slice(None),
                    index=index,
                    allow_zero_len=allow_zero_len,
                    range_format=range_format,
                    freq=freq
                )
                new_split = model.split(new_split, backwards=backwards)

            if split is not None:
                if len(new_split) > 1:
                    raise ValueError(
                        "Split function must return only one range if split "
                        "is already provided"
                    )

                model = SplitPeriod(
                    period=new_split[0],
                    index=index,
                    allow_zero_len=allow_zero_len,
                    range_format=range_format,
                    freq=freq
                )
                new_split = model.split(split, backwards=backwards)

            if n_sets is None:
                n_sets = len(new_split)

            elif n_sets != len(new_split):
                raise ValueError(
                    "All splits must have the same number of sets"
                )

            splits.append(new_split)
            if fix_ranges:
                split_bounds = tuple(
                    map(
                        lambda x: SplitPeriod.get_period_bounds(
                            x,
                            index=index,
                            index_bounds=index_bounds,
                            right_inclusive=right_inclusive,
                            freq=freq
                        ),
                        new_split,
                    )
                )
                bounds.append(split_bounds)
            split_idx += 1

        super().__init__(
            index,
            splits,
            fix_ranges=fix_ranges,
            backwards=backwards,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )
