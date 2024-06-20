import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
from pandas.core.groupby import GroupBy
from pandas.core.resample import Resampler
from numba import jit
import typing as tp

from fold.tools import BaseTool
from fold.utils.datetime import to_freq
from fold.utils import checks


CustomGrouperT = tp.TypeVar("CustomGrouperT", bound="CustomGrouper")


class CustomGrouper:
    """
    Class that exposes methods to group index.

    Parameters:
    ----------
    index : pd.Index
        The original index.
    group_by : Optional[GroupByLike], optional
        Defines how to group the index. by default None.
        `group_by` can be:
            * boolean (False for no grouping, True for one group),
            * integer (level by position),
            * string (level by name),
            * sequence of integers or strings that is shorter than `index` 
            (multiple levels),
            * any other sequence that has the same length as `index` 
            (group per index).
    def_lvl_name : Hashable, optional
        Default level name, by default 'group'.
    allow_enable : bool, optional
        Whether to allow enabling grouping, by default True.
        Set `allow_enable` to False to prohibit grouping if `group_by` is None.
    allow_disable : bool, optional
        Whether to allow disabling grouping, by default True.
        Set `allow_disable` to False to prohibit disabling of grouping if 
        `BaseGrouper.group_by` is not None.
    allow_modify : bool, optional
        Whether to allow modifying groups, by default True. 
        Set `allow_modify` to False to prohibit modifying groups 
        (you can still change their labels).
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        index: pd.Index,
        group_by: tp.Optional[bool | range | BaseTool] = None,
        def_lvl_name: tp.Hashable = "group",
        allow_enable: tp.Optional[bool] = True,
        allow_disable: tp.Optional[bool] = True,
        allow_modify: tp.Optional[bool] = True,
        **kwargs,
    ):
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if group_by is False:
            group_by = None
        else:
            group_by = self.group_by_to_index(
                index,
                group_by,
                def_lvl_name=def_lvl_name
            )

        self._index = index
        self._group_by = group_by
        self._def_lvl_name = def_lvl_name
        self._allow_enable = allow_enable
        self._allow_disable = allow_disable
        self._allow_modify = allow_modify

    @staticmethod
    def select_levels(
        index: pd.Index,
        levels: tp.Union[tp.Union[str, int], tp.Sequence[tp.Union[str, int]]],
        strict: bool = True,
    ) -> pd.Index:
        """
        Build a new index by selecting one or multiple `levels` from `index`.

        Parameters
        ----------
        index : pd.Index
            Input index.
        levels : tp.MaybeLevelSequence
            Sequence of levels to select from `index`.
        strict : bool, optional
            If True, raise KeyError for levels not found, by default True.

        Returns
        -------
        pd.Index
            New index containing selected levels from `index`.
        """
        was_multiindex = True
        if not isinstance(index, pd.MultiIndex):
            was_multiindex = False
            index = pd.MultiIndex.from_arrays([index])
        except_mode = False
        levels_to_select = list()
        if isinstance(levels, str) or not checks.is_sequence(levels):
            levels = [levels]
            single_mode = True
        else:
            single_mode = False

        for level in levels:
            if level in index.names:
                for level_pos in [i for i, x in enumerate(index.names) if x == level]:
                    if level_pos not in levels_to_select:
                        levels_to_select.append(level_pos)
            elif checks.is_int(level):
                if level < 0:
                    new_level = index.nlevels + level
                    if new_level < 0:
                        raise KeyError(f"Level at position {level} not found")
                    level = new_level
                if 0 <= level < index.nlevels:
                    if level not in levels_to_select:
                        levels_to_select.append(level)
                else:
                    raise KeyError(f"Level at position {level} not found")
            elif strict:
                raise KeyError(f"Level '{level}' not found")
        if except_mode:
            levels_to_select = list(
                set(range(index.nlevels)).difference(levels_to_select))
        if len(levels_to_select) == 0:
            if strict:
                raise ValueError("No levels to select")
            if not was_multiindex:
                return index.get_level_values(0)
            return index
        if len(levels_to_select) == 1 and single_mode:
            return index.get_level_values(levels_to_select[0])
        levels = [index.get_level_values(level) for level in levels_to_select]
        return pd.MultiIndex.from_arrays(levels)

    @classmethod
    def group_by_to_index(
        cls,
        index: pd.Index,
        group_by: bool | range | BaseTool,
        def_lvl_name: tp.Hashable = "group",
    ) -> bool | pd.Index:
        """
        Convert mapper `group_by` to `pd.Index`.

        Parameters:
        ----------
        index : pd.Index
            The original index.
        group_by : GroupByLike
            Defines how to group the index.

        Returns:
        -------
        bool | pd.Index
            The converted group mapper.
        """

        if group_by is None or group_by is False:
            return group_by

        if group_by is True:
            group_by = pd.Index(["group"] * len(index), name=def_lvl_name)

        elif isinstance(index, pd.MultiIndex) or isinstance(group_by, (int, str)):
            if isinstance(group_by, (int, str)):
                group_by = cls.select_levels(index, group_by)
            elif (
                isinstance(group_by, (tuple, list))
                and not isinstance(group_by[0], pd.Index)
                and len(group_by) <= len(index.names)
            ):
                try:
                    group_by = cls.select_levels(index, group_by)
                except (IndexError, KeyError):
                    pass

        if not isinstance(group_by, pd.Index):
            if isinstance(group_by[0], pd.Index):
                group_by = pd.MultiIndex.from_arrays(group_by)
            else:
                group_by = pd.Index(group_by, name=def_lvl_name)

        if len(group_by) != len(index):
            raise ValueError("group_by and index must have the same length")

        return group_by

    @classmethod
    def group_by_to_groups_and_index(
        cls,
        index: pd.Index,
        group_by: bool | range | BaseTool,
        def_lvl_name: tp.Hashable = "group",
    ) -> tp.Tuple[np.ndarray, pd.Index]:
        """
        Return array of group indices pointing to the original index, and 
        grouped index.

        Parameters:
        ----------
        index : pd.Index
            The original index.
        group_by : GroupByLike
            Defines how to group the index.

        Returns:
        -------
        Tuple[np.ndarray, pd.Index]
            Array of group indices and grouped index.
        """
        if group_by is None or group_by is False:
            return np.arange(len(index)), index

        group_by = cls.group_by_to_index(index, group_by, def_lvl_name)
        codes, uniques = pd.factorize(group_by)
        if not isinstance(uniques, pd.Index):
            new_index = pd.Index(uniques)
        else:
            new_index = uniques
        if isinstance(group_by, pd.MultiIndex):
            new_index.names = group_by.names
        elif isinstance(group_by, (pd.Index, pd.Series)):
            new_index.name = group_by.name
        return codes, new_index

    @classmethod
    def iter_group_map(cls, group_map: np.ndarray) -> tp.Generator:
        """
        Iterate over indices of each group in a group map.

        Parameters:
        ----------
        group_map : GroupMap
            Tuple containing group indices and group lengths.

        Yields:
        -------
        np.ndarray
            Indices of each group.
        """
        group_idxs, group_lens = group_map
        group_start = 0
        group_end = 0
        for group in range(len(group_lens)):
            group_len = group_lens[group]
            group_end += group_len
            yield group_idxs[group_start:group_end]
            group_start += group_len

    @classmethod
    def from_pd_group_by(
        cls,
        pd_group_by: GroupBy | Resampler | Offset | pd.Timedelta,
        **kwargs,
    ) -> CustomGrouperT:
        """
        Build a `BaseGrouper` instance from a pandas `GroupBy` object.

        Parameters:
        ----------
        pd_group_by : pd.core.groupby.GroupBy
            Pandas `GroupBy` object.

        Returns:
        -------
        BaseGrouperT
            The created `BaseGrouper` instance.
        """
        from fold.base.merging import concat_arrays

        if not isinstance(pd_group_by, (GroupBy, Resampler)):
            raise TypeError(
                "pd_group_by must be an instance of GroupBy or Resampler")
        indices = list(pd_group_by.indices.values())
        group_lens = np.asarray(list(map(len, indices)))
        groups = np.full(int(np.sum(group_lens)), 0, dtype=np.int_)
        group_start_idxs = np.cumsum(group_lens)[1:] - group_lens[1:]
        groups[group_start_idxs] = 1
        groups = np.cumsum(groups)
        index = pd.Index(concat_arrays(indices))
        group_by = pd.Index(
            list(pd_group_by.indices.keys()), name="group")[groups]
        return cls(
            index=index,
            group_by=group_by,
            **kwargs,
        )

    @property
    def index(self) -> pd.Index:
        """Original index."""
        return self._index

    @property
    def group_by(self) -> bool | pd.Index:
        """Mapper for grouping."""
        return self._group_by

    @property
    def def_lvl_name(self) -> tp.Hashable:
        """Default level name."""
        return self._def_lvl_name

    @property
    def allow_enable(self) -> bool:
        """Whether to allow enabling grouping."""
        return self._allow_enable

    @property
    def allow_disable(self) -> bool:
        """Whether to allow disabling grouping."""
        return self._allow_disable

    @property
    def allow_modify(self) -> bool:
        """Whether to allow changing groups."""
        return self._allow_modify

    def is_grouped(self, group_by: tp.Optional[bool | range | BaseTool] = None) -> bool:
        """
        Check whether index is grouped.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.

        Returns:
        -------
        bool
            True if the index is grouped, False otherwise.
        """
        if group_by is False:
            return False
        if group_by is None:
            group_by = self.group_by
        return group_by is not None

    def is_grouping_enabled(self, group_by: tp.Optional[bool | range | BaseTool] = None) -> bool:
        """
        Check whether grouping has been enabled.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.

        Returns:
        -------
        bool
            True if grouping is enabled, False otherwise.
        """
        return self.group_by is None and self.is_grouped(group_by=group_by)

    def is_grouping_disabled(self, group_by: tp.Optional[bool | range | BaseTool] = None) -> bool:
        """
        Check whether grouping has been disabled.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.

        Returns:
        -------
        bool
            True if grouping is disabled, False otherwise.
        """
        return self.group_by is not None and not self.is_grouped(group_by=group_by)

    def is_grouping_modified(self, group_by: tp.Optional[bool | range | BaseTool] = None) -> bool:
        """
        Check whether grouping has been modified.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.

        Returns:
        -------
        bool
            True if grouping has been modified, False otherwise.
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False
        group_by = self.group_by_to_index(
            self.index, group_by, def_lvl_name=self.def_lvl_name)
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            if not pd.Index.equals(group_by, self.group_by):
                groups1 = self.group_by_to_groups_and_index(
                    self.index,
                    group_by,
                    def_lvl_name=self.def_lvl_name,
                )[0]
                groups2 = self.group_by_to_groups_and_index(
                    self.index,
                    self.group_by,
                    def_lvl_name=self.def_lvl_name,
                )[0]
                if not np.array_equal(groups1, groups2):
                    return True
            return False
        return True

    def check_group_by(
        self,
        group_by: tp.Optional[bool | range | BaseTool] = None,
        allow_enable: tp.Optional[bool] = None,
        allow_disable: tp.Optional[bool] = None,
        allow_modify: tp.Optional[bool] = None,
    ) -> None:
        """
        Check passed `group_by` object against restrictions.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.
        allow_enable : bool, optional
            Whether to allow enabling grouping.
        allow_disable : bool, optional
            Whether to allow disabling grouping.
        allow_modify : bool, optional
            Whether to allow modifying groups.

        Raises:
        ------
        ValueError
            If the specified `group_by` violates the restrictions.
        """
        if allow_enable is None:
            allow_enable = self.allow_enable
        if allow_disable is None:
            allow_disable = self.allow_disable
        if allow_modify is None:
            allow_modify = self.allow_modify

        if self.is_grouping_enabled(group_by=group_by):
            if not allow_enable:
                raise ValueError("Enabling grouping is not allowed")
        elif self.is_grouping_disabled(group_by=group_by):
            if not allow_disable:
                raise ValueError("Disabling grouping is not allowed")
        elif self.is_grouping_modified(group_by=group_by):
            if not allow_modify:
                raise ValueError("Modifying groups is not allowed")

    def resolve_group_by(self, group_by: tp.Optional[bool | range | BaseTool] = None, **kwargs) -> bool | pd.Index:
        """
        Resolve `group_by` from either object variable or keyword argument.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        bool | pd.Index
            The resolved group mapper.
        """
        if group_by is None:
            group_by = self.group_by
        if group_by is False and self.group_by is None:
            group_by = None
        self.check_group_by(group_by=group_by, **kwargs)
        return self.group_by_to_index(self.index, group_by, def_lvl_name=self.def_lvl_name)

    def get_groups_and_index(self, group_by: tp.Optional[bool | range | BaseTool] = None, **kwargs) -> tp.Tuple[np.ndarray, pd.Index]:
        """
        Return array of group indices pointing to the original index, and 
        grouped index.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        Tuple[np.ndarray, pd.Index]
            Array of group indices and grouped index.
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        return self.group_by_to_groups_and_index(
            self.index,
            group_by,
            def_lvl_name=self.def_lvl_name
        )

    def get_groups(self, **kwargs) -> np.ndarray:
        """
        Return groups array.

        Parameters:
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        np.ndarray
            Array of group labels.
        """
        return self.get_groups_and_index(**kwargs)[0]

    def get_index(self, **kwargs) -> pd.Index:
        """
        Return grouped index.

        Parameters:
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        pd.Index
            The grouped index.
        """
        return self.get_groups_and_index(**kwargs)[1]

    def is_sorted(self, group_by: tp.Optional[bool | range | BaseTool] = None, **kwargs) -> bool:
        """
        Return whether groups are monolithic and sorted.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        bool
            True if groups are monolithic and sorted, False otherwise.
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        groups = self.get_groups(group_by=group_by)
        return checks.is_sorted(groups)

    @staticmethod
    @jit(cache=True)
    def get_group_map_nb(groups: np.ndarray, n_groups: int) -> np.ndarray:
        """
        Build the map between groups and indices.

        Returns an array with indices segmented by group and an array with group 
        lengths.
        Works well for unsorted group arrays.

        Parameters:
        ----------
        groups : np.ndarray
            Array of group labels.
        n_groups : int
            Number of unique groups.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing group indices and group lengths.
        """
        group_lens_out = np.full(n_groups, 0, dtype=np.int_)
        for g in range(groups.shape[0]):
            group = groups[g]
            group_lens_out[group] += 1

        group_start_idxs = np.cumsum(group_lens_out) - group_lens_out
        group_idxs_out = np.empty((groups.shape[0],), dtype=np.int_)
        group_i = np.full(n_groups, 0, dtype=np.int_)
        for g in range(groups.shape[0]):
            group = groups[g]
            group_idxs_out[group_start_idxs[group] + group_i[group]] = g
            group_i[group] += 1

        return group_idxs_out, group_lens_out

    def get_group_map(self, group_by: tp.Optional[bool | range | BaseTool] = None, **kwargs) -> np.ndarray:
        """
        Return the group map.

        Parameters:
        ----------
        group_by : GroupByLike, optional
            Optional custom grouping specification.
        **kwargs : Any
            Additional keyword arguments.

        Returns:
        -------
        GroupMap
            Tuple containing group indices and group lengths.
        """
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        if group_by is None or group_by is False:  # no grouping
            return np.arange(len(self.index)), np.full(len(self.index), 1)
        groups, new_index = self.get_groups_and_index(group_by=group_by)
        func = self.get_group_map_nb.py_func
        return func(groups, len(new_index))

    def iter_group_idxs(self, **kwargs) -> tp.Generator[np.ndarray, None, None]:
        """
        Iterate over indices of each group.

        Parameters:
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Yields:
        -------
        np.ndarray
            Indices of each group.
        """
        group_map = self.get_group_map(**kwargs)
        return self.iter_group_map(group_map)


def to_period(index: pd.Index, freq: str | int | float | Offset | pd.Timedelta) -> pd.PeriodIndex:
    """
    Convert the index to a PeriodIndex.

    Parameters:
    -----------
    freq : tp.Union[Offset, str, int, float, pd.Timedelta, np.timedelta64, timedelta]
        Frequency string or object.
    shift : bool, optional
        Shift the PeriodIndex, by default False.

    Returns:
    --------
    pd.PeriodIndex
        Converted PeriodIndex.
    """
    if isinstance(index, pd.DatetimeIndex):
        index = index.tz_localize(None).to_period(freq)
        
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(
            f"Cannot convert index of type {type(index)} to period"
        )
        
    return index


def get_grouper(
    index: pd.Index,
    by: str | CustomGrouper | GroupBy | Resampler | Offset | BaseTool | pd.Timedelta,
    groupby_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    **kwargs
) -> CustomGrouper:
    """
    Get an index grouper.

    Parameters:
    -----------
    by : tp.AnyGroupByLike
        Grouping specifier.
    groupby_kwargs : tp.Union[None, tp.Dict[str, tp.Any]], optional
        Keyword arguments for groupby methods, by default None.
    **kwargs
        Additional keyword arguments.

    Returns:
    --------
    BaseGrouper
        Index grouper.
    """
    groupby_kwargs = groupby_kwargs or {}
    if isinstance(by, CustomGrouper):
        if len(kwargs) > 0:
            return by.replace(**kwargs)
        return by

    if isinstance(by, (GroupBy, Resampler)):
        return CustomGrouper.from_pd_group_by(by, **kwargs)

    try:
        return CustomGrouper(index=index, group_by=by, **kwargs)
    except Exception:
        pass

    if isinstance(index, pd.DatetimeIndex):

        try:
            return CustomGrouper(
                index=index,
                group_by=to_period(index, to_freq(by)),
                **kwargs
            )
        except Exception:
            pass

        try:
            pd_group_by = pd.Series(index=index, dtype=object).resample(
                to_freq(by),
                **groupby_kwargs
            )
            return CustomGrouper.from_pd_group_by(pd_group_by, **kwargs)
        except Exception:
            pass

    pd_group_by = pd.Series(
        index=index,
        dtype=object
    ).groupby(
        by,
        axis=0,
        **groupby_kwargs
    )
    return CustomGrouper.from_pd_group_by(pd_group_by, **kwargs)
