import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
from pandas.core.groupby import GroupBy
from pandas.core.resample import Resampler
import typing as tp

from fold.model_selection.base import BaseModel
from fold.tools import BaseTool, BasePeriod, SplitPeriod
from fold.utils.datetime import prepare_dt_index, infer_index_freq
from fold.utils.grouper import get_grouper


class GrouperSplit(BaseModel):
    """
    Split by group.

    Parameters
    ----------
    index : range
        The range of the index to be split.
    by : str, GroupBy, Resampler, Offset, BaseTool, pd.Timedelta
        The criteria by which to group the index.
    groupby_kwargs : Dict[str, Any], optional
        Keyword arguments for the groupby operation, by default None.
    grouper_kwargs : Dict[str, Any], optional
        Keyword arguments for the grouper, by default None.
    split : int, float, slice, BaseTool, BasePeriod, optional
        The specific split to apply, by default None.
    allow_zero_len : bool, optional
        Whether to allow zero-length splits, by default False.
    range_format : str, optional
        Format for the range, by default None.
    freq : str, int, float, Offset, pd.Timedelta, optional
        Frequency of the index, by default "auto".
    constraints : BaseTool, optional
        Constraints for the splits, by default None.
    backwards : bool, optional
        Whether to split in reverse order, by default False.
    split_labels : range, optional
        Labels for the splits, by default None.
    sample_labels : range, optional
        Labels for the samples, by default None.

    Examples
    --------
    Example 1: Split by year start
    ```pycon
    >>> model = GrouperSplit(
    ...     index,
    ...     by="YS",
    ...     split=0.5,
    ...     sample_labels=["IS", "OOS"]
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound            start        end
         sample                      
    2010 IS     2010-10-06 2010-11-18
         OOS    2010-11-18 2011-01-01
    2011 IS     2011-01-01 2011-07-02
         OOS    2011-07-02 2012-01-01
    2012 IS     2012-01-01 2012-07-02
         OOS    2012-07-02 2013-01-01
    2013 IS     2013-01-01 2013-07-02
         OOS    2013-07-02 2014-01-01
    2014 IS     2014-01-01 2014-07-02
         OOS    2014-07-02 2015-01-01
    2015 IS     2015-01-01 2015-07-02
         OOS    2015-07-02 2016-01-01
    2016 IS     2016-01-01 2016-07-02
         OOS    2016-07-02 2017-01-01
    2017 IS     2017-01-01 2017-07-02
         OOS    2017-07-02 2018-01-01
    2018 IS     2018-01-01 2018-07-02
         OOS    2018-07-02 2019-01-01
    2019 IS     2019-01-01 2019-07-02
         OOS    2019-07-02 2020-01-01
    2020 IS     2020-01-01 2020-07-02
         OOS    2020-07-02 2021-01-01
    2021 IS     2021-01-01 2021-07-02
         OOS    2021-07-02 2022-01-01
    2022 IS     2022-01-01 2022-07-02
         OOS    2022-07-02 2023-01-01
    2023 IS     2023-01-01 2023-07-02
         OOS    2023-07-02 2024-01-01
    2024 IS     2024-01-01 2024-03-23
         OOS    2024-03-23 2024-06-14
    ```
    Example 2: Complete year start
    ```pycon
    >>> from fold.tools import Function
    >>> def is_split_complete(index, split):
    ...     first_range = split[0]
    ...     first_index = index[first_range][0]
    ...     last_range = split[-1]
    ...     last_index = index[last_range][-1]
    ...     return first_index.is_year_start and last_index.is_year_end
    >>> 
    >>> model = GrouperSplit(
    ...     index,
    ...     by="YS",
    ...     split=0.5,
    ...     constraints=Function(is_split_complete),
    ...     sample_labels=['IS', 'OOS']
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound            start        end
         sample                      
    2011 IS     2011-01-01 2011-07-02
         OOS    2011-07-02 2012-01-01
    2012 IS     2012-01-01 2012-07-02
         OOS    2012-07-02 2013-01-01
    2013 IS     2013-01-01 2013-07-02
         OOS    2013-07-02 2014-01-01
    2014 IS     2014-01-01 2014-07-02
         OOS    2014-07-02 2015-01-01
    2015 IS     2015-01-01 2015-07-02
         OOS    2015-07-02 2016-01-01
    2016 IS     2016-01-01 2016-07-02
         OOS    2016-07-02 2017-01-01
    2017 IS     2017-01-01 2017-07-02
         OOS    2017-07-02 2018-01-01
    2018 IS     2018-01-01 2018-07-02
         OOS    2018-07-02 2019-01-01
    2019 IS     2019-01-01 2019-07-02
         OOS    2019-07-02 2020-01-01
    2020 IS     2020-01-01 2020-07-02
         OOS    2020-07-02 2021-01-01
    2021 IS     2021-01-01 2021-07-02
         OOS    2021-07-02 2022-01-01
    2022 IS     2022-01-01 2022-07-02
         OOS    2022-07-02 2023-01-01
    2023 IS     2023-01-01 2023-07-02
         OOS    2023-07-02 2024-01-01
    ```

    Example 3: Using `by=index.year`
    ```pycon
    >>> model = GrouperSplit(
    ...     index,
    ...     by=index.year,
    ...     split=0.5,
    ...     constraints=Function(is_split_complete)
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    ```
    bound              start        end
         sample                        
    2011 sample_0 2011-01-01 2011-07-02
         sample_1 2011-07-02 2012-01-01
    2012 sample_0 2012-01-01 2012-07-02
         sample_1 2012-07-02 2013-01-01
    2013 sample_0 2013-01-01 2013-07-02
         sample_1 2013-07-02 2014-01-01
    2014 sample_0 2014-01-01 2014-07-02
         sample_1 2014-07-02 2015-01-01
    2015 sample_0 2015-01-01 2015-07-02
         sample_1 2015-07-02 2016-01-01
    2016 sample_0 2016-01-01 2016-07-02
         sample_1 2016-07-02 2017-01-01
    2017 sample_0 2017-01-01 2017-07-02
         sample_1 2017-07-02 2018-01-01
    2018 sample_0 2018-01-01 2018-07-02
         sample_1 2018-07-02 2019-01-01
    2019 sample_0 2019-01-01 2019-07-02
         sample_1 2019-07-02 2020-01-01
    2020 sample_0 2020-01-01 2020-07-02
         sample_1 2020-07-02 2021-01-01
    2021 sample_0 2021-01-01 2021-07-02
         sample_1 2021-07-02 2022-01-01
    2022 sample_0 2022-01-01 2022-07-02
         sample_1 2022-07-02 2023-01-01
    2023 sample_0 2023-01-01 2023-07-02
         sample_1 2023-07-02 2024-01-01
    ```

    Example 4: Complete month end
    ```pycon
    >>> def is_month_end(index, split):
    ...     last_range = split[-1]
    ...     return index[last_range][-1].is_month_end
    >>> 
    >>> model = GrouperSplit(
    ...     index,
    ...     by="M",
    ...     constraints=Function(is_month_end)
    ... )
    >>> print(model.get_bounds(index_bounds=True))
    bound              start        end
         sample                        
    2011 sample_0 2011-01-01 2011-07-02
         sample_1 2011-07-02 2012-01-01
    2012 sample_0 2012-01-01 2012-07-02
         sample_1 2012-07-02 2013-01-01
    2013 sample_0 2013-01-01 2013-07-02
         sample_1 2013-07-02 2014-01-01
    2014 sample_0 2014-01-01 2014-07-02
         sample_1 2014-07-02 2015-01-01
    2015 sample_0 2015-01-01 2015-07-02
         sample_1 2015-07-02 2016-01-01
    2016 sample_0 2016-01-01 2016-07-02
         sample_1 2016-07-02 2017-01-01
    2017 sample_0 2017-01-01 2017-07-02
         sample_1 2017-07-02 2018-01-01
    2018 sample_0 2018-01-01 2018-07-02
         sample_1 2018-07-02 2019-01-01
    2019 sample_0 2019-01-01 2019-07-02
         sample_1 2019-07-02 2020-01-01
    2020 sample_0 2020-01-01 2020-07-02
         sample_1 2020-07-02 2021-01-01
    2021 sample_0 2021-01-01 2021-07-02
         sample_1 2021-07-02 2022-01-01
    2022 sample_0 2022-01-01 2022-07-02
         sample_1 2022-07-02 2023-01-01
    2023 sample_0 2023-01-01 2023-07-02
         sample_1 2023-07-02 2024-01-01
    ```
    """

    def __init__(
        self,
        index: range,
        by: str | GroupBy | Resampler | Offset | BaseTool | pd.Timedelta,
        groupby_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        grouper_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        split: tp.Optional[int | float | slice | BaseTool | BasePeriod] = None,
        allow_zero_len: tp.Optional[bool] = False,
        range_format: tp.Optional[str] = None,
        freq: tp.Optional[str | int | float | Offset | pd.Timedelta] = "auto",
        constraints: tp.Optional[BaseTool] = None,
        backwards: tp.Optional[bool] = False,
        split_labels: tp.Optional[range] = None,
        sample_labels: tp.Optional[range] = None
    ):
        index = prepare_dt_index(index)

        try:
            freq = infer_index_freq(index, freq, allow_numeric=False)
        except Exception:
            freq = None

        grouper_kwargs = grouper_kwargs or {}
        grouper = get_grouper(
            index,
            by,
            groupby_kwargs=groupby_kwargs,
            **grouper_kwargs
        )
        splits = []
        indices = []
        for i, new_split in enumerate(grouper.iter_group_idxs()):
            if split is not None:
                model = SplitPeriod(
                    period=new_split,
                    index=index,
                    allow_zero_len=allow_zero_len,
                    range_format=range_format,
                    freq=freq
                )
                new_split = model.split(split, backwards=backwards)

            else:
                new_split = [new_split]
            splits.append(new_split)
            indices.append(i)

        if split_labels is None:
            split_labels = grouper.get_index()[indices]

        super().__init__(
            index,
            splits,
            backwards=backwards,
            allow_zero_len=allow_zero_len,
            range_format=range_format,
            freq=freq,
            constraints=constraints,
            split_labels=split_labels,
            sample_labels=sample_labels,
        )

