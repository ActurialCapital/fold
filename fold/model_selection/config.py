from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset
import typing as tp

from fold.tools import FixedPeriod, RelativePeriod, SplitPeriod
from fold.utils import checks


class Config:
    """
    Config utility class.

    Attributes
    ----------
    index : Index
        The index.
    columns : Index
        The columns.
    ndim : int
        The number of dimensions.

    """

    def __init__(self, index: pd.Index, columns: pd.Index, ndim: int):
        self._index = index
        self._columns = columns
        self._ndim = ndim

    @property
    def index(self) -> pd.Index:
        """
        Get the index.

        Returns
        -------
        pd.Index
            The index.
        """
        return self._index

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns.

        Returns
        -------
        pd.Index
            The columns.
        """
        return self._columns

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
    def object_name(self) -> str | None:
        """
        Get the name of the Pandas Series/DataFrame.

        Returns
        -------
        str or None
            The name of the Series/DataFrame, or None if it does not have a 
            name.
        """
        if len(self.columns) == 1:
            name = self.columns[0]

            if name == 0:
                name = None

        else:
            name = None

        return name

    def to_ndim(
        self,
        arg: tp.Any
    ) -> np.ndarray | pd.Index | pd.Series | pd.DataFrame:
        """
        Try to softly bring `arg` to the specified number of dimensions `ndim`
        (max 2).

        Parameters
        ----------
        arg : tp.Any
            The array-like object to reshape.

        Returns
        -------
        np.ndarray or pd.Index or pd.Series or pd.DataFrame
            The reshaped array-like object.

        Notes
        -----
        This function attempts to reshape the input `arg` to match the 
        specified `ndim`. It handles soft conversions between 1-dimensional and 
        2-dimensional arrays, including downgrading DataFrame to Series 
        (axis 1) and upgrading Series to DataFrame (axis 1).
        """
        if checks.is_any_array(arg):
            if checks.is_index(arg):
                arg = arg.to_series()

            else:
                arg = arg

        elif checks.is_mapping_like(arg):
            if checks.is_namedtuple(arg):
                arg = arg._asdict()

            arg = pd.Series(arg)

        if self.ndim == 1:
            if arg.ndim == 2:
                if arg.shape[1] == 1:
                    if checks.is_frame(arg):
                        return arg.iloc[:, 0]
                    # downgrade
                    return arg[:, 0]

        if self.ndim == 2:
            if arg.ndim == 1:
                if checks.is_series(arg):
                    return arg.to_frame()
                # upgrade
                return arg[:, None]

        return arg

    def apply(self, arr: tp.Any) -> pd.Series | pd.DataFrame:
        """
        Apply operation to array.

        Parameters
        ----------
        arr : tp.Any
            The array-like object.

        Returns
        -------
        pd.Series | pd.DataFrame
            The series or dataframe.

        """
        arr = self.to_ndim(np.asarray(arr))

        if arr.ndim == 1:
            return pd.Series(arr, index=self.index, name=self.object_name)

        if arr.ndim == 2:
            if arr.shape[1] == 1 and self.ndim == 1:
                return pd.Series(arr[:, 0], index=self.index, name=self.name)

            return pd.DataFrame(arr, index=self.index, columns=self.columns)

        raise ValueError(f"{arr.ndim}-d input is not supported")


@dataclass
class UpdateConfig:
    """
    Configuration for updating periods.

    Attributes
    ----------
    split : Any
        The split criteria.
    index : range
        The index range.
    fix_ranges : bool, optional
        Whether to fix ranges, by default True.
    backwards : bool, optional
        Whether to perform operations backwards, by default False.
    allow_zero_len : bool, optional
        Whether to allow zero length periods, by default False.
    range_format : Optional[str], optional
        The format of the range, by default None.
    freq : Optional[Union[str, int, BaseOffset, pd.Timedelta]], optional
        The frequency of the periods, by default None.
    ndim : int, optional
        The number of dimensions, by default 2.
    updated_period : Sequence[Any], optional
        The updated periods, by default None.

    Methods
    -------
    __post_init__()
        Initialize updated periods based on split criteria.
    """
    split: tp.Any
    index: range
    fix_ranges: bool = True
    backwards: bool = False
    allow_zero_len: bool = False
    range_format: tp.Optional[str] = None
    freq: tp.Optional[str | int | BaseOffset | pd.Timedelta] = None
    ndim: int = 2
    updated_period: tp.Sequence[tp.Any] = None

    def __post_init__(self):
        """
        Post-initialization to setup updated periods based on split criteria.
        """
        if checks.is_number(self.split) or checks.is_td_like(self.split):
            model = SplitPeriod(
                period=slice(None),
                index=self.index,
                allow_zero_len=self.allow_zero_len,
                range_format=self.range_format,
                freq=self.freq
            )
            split_period = True
            periods = model.split(self.split, self.backwards)
            self.ndim = 2

        elif (
            checks.is_range_relative(self.split) or not
            checks.is_sequence(self.split) or
            checks.is_np_array(self.split)
        ):
            split_period = False
            periods = [self.split]
            self.ndim = 1

        else:
            split_period = False
            periods = self.split
            self.ndim = 2

        if self.fix_ranges and not split_period:
            model = SplitPeriod(
                period=slice(None),
                index=self.index,
                allow_zero_len=self.allow_zero_len,
                range_format=self.range_format,
                freq=self.freq
            )
            periods = model.split(periods, self.backwards)

        self.updated_period = []
        for period in periods:

            if checks.is_number(period) or checks.is_td_like(period):
                period = RelativePeriod(length=period)

            if not isinstance(period, (FixedPeriod, RelativePeriod)):
                if checks.is_sequence(period):
                    self.updated_period.append(FixedPeriod(period))

                else:
                    self.updated_period.append(period)

            else:
                self.updated_period.append(period)
