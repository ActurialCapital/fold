import warnings
from datetime import datetime, time #timezone, timedelta, tzinfo, date
import re
from dataclasses import dataclass
import typing as tp

import dateparser
from numba import jit
import numpy as np

import pandas as pd
from pandas.tseries.offsets import BaseOffset as Offset
from pandas.tseries.frequencies import to_offset as pd_to_offset

from fold.utils import checks


FreqHandlerT = tp.TypeVar("FreqHandlerT", bound="FreqHandler")

@dataclass
class FreqHandler:
    """
    Frequency handler.

    Parameters
    ----------
    year : Optional[int], default None
        Year.
    month : Optional[int], default None
        Month.
    day : Optional[int], default None
        Day of month.
    weekday : Optional[int], default None
        Day of week.
    hour : Optional[int], default None
        Hour.
    minute : Optional[int], default None
        Minute.
    second : Optional[int], default None
        Second.
    nanosecond : Optional[int], default None
        Nanosecond.

    """
    year: tp.Optional[int] = None
    month: tp.Optional[int] = None
    day: tp.Optional[int] = None
    weekday: tp.Optional[int] = None
    hour: tp.Optional[int] = None
    minute: tp.Optional[int] = None
    second: tp.Optional[int] = None
    nanosecond: tp.Optional[int] = None

    def has_date(self) -> bool:
        """Whether any date component is set."""
        return ( 
            self.year is not None or 
            self.month is not None or 
            self.day is not None
        )
    
    def has_weekday(self) -> bool:
        """Whether the weekday component is set."""
        return self.weekday is not None

    def has_time(self) -> bool:
        """Whether any time component is set."""
        return (
            self.hour is not None or 
            self.minute is not None or 
            self.second is not None or
            self.nanosecond is not None
        )
    
    def to_time(self) -> time:
        """Convert to a `datetime.time` instance."""
        return time(
            hour=self.hour if self.hour is not None else 0,
            minute=self.minute if self.minute is not None else 0,
            second=self.second if self.second is not None else 0,
            microsecond=self.nanosecond // 1000 if self.nanosecond is not None else 0,
        )
    
    @classmethod
    def parse_time_str(
        cls, 
        time_str: str, 
        **parse_kwargs
    ) -> FreqHandlerT:
        """Parse `DTC` instance from a time string."""
        from dateutil.parser import parser

        result = parser()._parse(time_str, **parse_kwargs)[0]
        if result.microsecond is None:
            nanosecond = None
        else:
            nanosecond = result.microsecond * 1000
        return cls(
            year=result.year,
            month=result.month,
            day=result.day,
            weekday=result.weekday,
            hour=result.hour,
            minute=result.minute,
            second=result.second,
            nanosecond=nanosecond,
        )

us_ns = 1000
"""Microsecond (nanoseconds)."""

ms_ns = us_ns * 1000
"""Millisecond (nanoseconds)."""

s_ns = ms_ns * 1000
"""Second (nanoseconds)."""

m_ns = s_ns * 60
"""Minute (nanoseconds)."""

h_ns = m_ns * 60
"""Hour (nanoseconds)."""

d_ns = h_ns * 24
"""Day (nanoseconds)."""

w_ns = d_ns * 7
"""Week (nanoseconds)."""

y_ns = (d_ns * 438291) // 1200
"""Year (nanoseconds)."""

q_ns = y_ns // 4
"""Quarter (nanoseconds)."""

mo_ns = q_ns // 3
"""Month (nanoseconds)."""

semi_mo_ns = mo_ns // 2
"""Semi-month (nanoseconds)."""

h_td = np.timedelta64(h_ns, "ns")
"""Hour (timedelta)."""

d_td = np.timedelta64(d_ns, "ns")
"""Day (timedelta)."""

w_td = np.timedelta64(w_ns, "ns")
"""Week (timedelta)."""

semi_mo_td = np.timedelta64(semi_mo_ns, "ns")
"""Semi-month (timedelta)."""

mo_td = np.timedelta64(mo_ns, "ns")
"""Month (timedelta)."""

q_td = np.timedelta64(q_ns, "ns")
"""Quarter (timedelta)."""

y_td = np.timedelta64(y_ns, "ns")
"""Year (timedelta)."""


sharp_freq_str_config = (
    dict(
        m="m",
        M="M",
    )
)


config_freq = (
    dict(
        ns="ns",
        nano="ns",
        nanos="ns",
        nanosecond="ns",
        nanoseconds="ns",
        us="us",
        micro="us",
        micros="us",
        microsecond="us",
        microseconds="us",
        ms="ms",
        milli="ms",
        millis="ms",
        millisecond="ms",
        milliseconds="ms",
        s="s",
        sec="s",
        secs="s",
        second="s",
        seconds="s",
        t="m",
        min="m",
        mins="m",
        minute="m",
        minutes="m",
        h="h",
        hour="h",
        hours="h",
        hourly="h",
        d="d",
        day="d",
        days="d",
        daily="d",
        w="W",
        wk="W",
        wks="W",
        week="W",
        weeks="W",
        weekly="W",
        m="m",
        M="M",
        mo="M",
        month="M",
        months="M",
        monthly="M",
        q="Q",
        quarter="Q",
        quarters="Q",
        quarterly="Q",
        y="Y",
        year="Y",
        years="Y",
        yearly="Y",
        annual="Y",
        annually="Y",
    )
)


def _split_freq_str(
    freq_str: str,
    sharp_mapping: tp.Any = None,
    fuzzy_mapping: tp.Any = None,
) -> tp.Optional[tp.Tuple[int, str]]:
    """
    Split (human-readable) frequency into multiplier and unambiguous unit.

    Can be used both as offset and timedelta.

    For mappings, see `sharp_freq_str_config` and `config_freq`.
    Sharp (case-sensitive) mappings are considered first, fuzzy 
    (case-insensitive) mappings second. If a mapping returns None, will return 
    the original unit.

    The following case-sensitive units are returned:
    * "ns" for nanosecond
    * "us" for microsecond
    * "ms" for millisecond
    * "s" for second
    * "m" for minute
    * "h" for hour
    * "d" for day
    * "W" for week
    * "M" for month
    * "Q" for quarter
    * "Y" for year

    If a unit isn't recognized, will return the original unit.

    Parameters
    ----------
    freq_str : str
        The frequency string to split.
    sharp_mapping : dict, optional
        A dictionary of case-sensitive mappings for units.
    fuzzy_mapping : dict, optional
        A dictionary of case-insensitive mappings for units.

    Returns
    -------
    tp.Optional[tp.Tuple[int, str]]
        A tuple containing the multiplier and the unambiguous unit. If the input 
        string cannot be parsed, returns None.

    Raises
    ------
    ValueError
        If the frequency string does not contain a unit or if the unit is invalid.
    """
    freq_str = "".join(freq_str.strip().split())
    match = re.match(r"^(\d*)\s*([a-zA-Z-]+)$", freq_str)

    if not match:
        return None

    if match.group(1) == "" and match.group(2).isnumeric():
        raise ValueError("Frequency must contain unit")

    if match.group(1) == "":
        multiplier = 1

    else:
        multiplier = int(match.group(1))

    if match.group(2) == "":
        raise ValueError("Frequency must contain unit")

    else:
        unit = match.group(2)

    if sharp_mapping is not None:
        sharp_mapping = dict(sharp_mapping)

        if unit in sharp_mapping:
            if sharp_mapping[unit] is None:
                return multiplier, unit

            return multiplier, sharp_mapping[unit]

    if unit in sharp_freq_str_config:
        if sharp_freq_str_config[unit] is None:
            return multiplier, unit

        return multiplier, sharp_freq_str_config[unit]

    if fuzzy_mapping is not None:
        fuzzy_mapping = dict(fuzzy_mapping)

        if unit.lower() in fuzzy_mapping:
            if fuzzy_mapping[unit.lower()] is None:
                return multiplier, unit

            return multiplier, fuzzy_mapping[unit.lower()]

    if unit.lower() in config_freq:
        if config_freq[unit.lower()] is None:
            return multiplier, unit

        return multiplier, config_freq[unit.lower()]

    return multiplier, unit


def _prepare_offset_str(offset_str: str, allow_space: bool = False) -> str:
    """
    Prepare offset frequency string.

    To include multiple units, separate them with comma, semicolon, or space if 
    `allow_space` is True. The output becomes comma-separated.

    Parameters
    ----------
    offset_str : str
        The offset frequency string to prepare.
    allow_space : bool, optional
        If True, allow spaces as delimiters in addition to commas and 
        semicolons.

    Returns
    -------
    str
        The prepared offset frequency string.

    """
    from pkg_resources import parse_version

    if parse_version(pd.__version__) < parse_version("2.2.0"):
        year_prefix = "AS"

    else:
        year_prefix = "YS"

    if allow_space:
        freq_parts = re.split(r"[,;\s]", offset_str)

    else:
        freq_parts = re.split(r"[,;]", offset_str)

    updated_frequency_parts = []
    for freq_part in freq_parts:
        freq_part = " ".join(freq_part.strip().split())

        if freq_part == "":
            continue

        split = _split_freq_str(freq_part, sharp_mapping=dict(MS=None))

        if split is None:
            return offset_str

        multiplier, unit = split

        if unit == "m":
            unit = "min"

        elif unit == "W":
            unit = "W-MON"

        elif unit == "M":
            unit = "MS"

        elif unit == "Q":
            unit = "QS"

        elif unit == "Y":
            unit = "YS"

        elif unit.lower() in ("mon", "monday"):
            unit = "W-MON"

        elif unit.lower() in ("tue", "tuesday"):
            unit = "W-TUE"

        elif unit.lower() in ("wed", "wednesday"):
            unit = "W-WED"

        elif unit.lower() in ("thu", "thursday"):
            unit = "W-THU"

        elif unit.lower() in ("fri", "friday"):
            unit = "W-FRI"

        elif unit.lower() in ("sat", "saturday"):
            unit = "W-SAT"

        elif unit.lower() in ("sun", "sunday"):
            unit = "W-SUN"

        elif unit.lower() in ("jan", "january"):
            unit = year_prefix + "-JAN"

        elif unit.lower() in ("feb", "february"):
            unit = year_prefix + "-FEB"

        elif unit.lower() in ("mar", "march"):
            unit = year_prefix + "-MAR"

        elif unit.lower() in ("apr", "april"):
            unit = year_prefix + "-APR"

        elif unit.lower() == "may":
            unit = year_prefix + "-MAY"

        elif unit.lower() in ("jun", "june"):
            unit = year_prefix + "-JUN"

        elif unit.lower() in ("jul", "july"):
            unit = year_prefix + "-JUL"

        elif unit.lower() in ("aug", "august"):
            unit = year_prefix + "-AUG"

        elif unit.lower() in ("sep", "september"):
            unit = year_prefix + "-SEP"

        elif unit.lower() in ("oct", "october"):
            unit = year_prefix + "-OCT"

        elif unit.lower() in ("nov", "november"):
            unit = year_prefix + "-NOV"

        elif unit.lower() in ("dec", "december"):
            unit = year_prefix + "-DEC"
        updated_frequency_parts.append(str(multiplier) + str(unit))
    return " ".join(updated_frequency_parts)


def to_offset(
    freq: tp.Optional[tp.Union[str, int, Offset, pd.Timedelta]]
) -> Offset:
    """
    Convert a frequency-like object to `pd.DateOffset`.

    Parameters
    ----------
    freq : str, int, Offset, or pd.Timedelta, optional
        The frequency-like object to convert.

    Returns
    -------
    Offset
        The converted frequency as a `pd.DateOffset`.

    """
    if isinstance(freq, Offset):
        return freq

    if isinstance(freq, str):
        freq = _prepare_offset_str(freq)

    return pd_to_offset(freq)


def _prepare_timedelta_str(timedelta_str: str, allow_space: bool = False) -> str:
    """
    Prepare timedelta frequency string.

    To include multiple units, separate them with comma, semicolon, or space if 
    `allow_space` is True. The output becomes comma-separated.

    Parameters
    ----------
    timedelta_str : str
        The timedelta frequency string to prepare.
    allow_space : bool, optional
        If True, allow spaces as delimiters in addition to commas and 
        semicolons.

    Returns
    -------
    str
        The prepared timedelta frequency string.

    """
    if allow_space:
        freq_parts = re.split(r"[,;\s]", timedelta_str)

    else:
        freq_parts = re.split(r"[,;]", timedelta_str)

    updated_frequency_parts = []
    for freq_part in freq_parts:
        freq_part = " ".join(freq_part.strip().split())

        if freq_part == "":
            continue
        split = _split_freq_str(freq_part)

        if split is None:
            return timedelta_str

        multiplier, unit = split
        if unit == "m":
            unit = "min"

        elif unit == "W":
            multiplier *= 7
            unit = "d"

        elif unit == "M":
            multiplier *= mo_ns / d_ns
            unit = "d"

        elif unit == "Q":
            multiplier *= q_ns / d_ns
            unit = "d"

        elif unit == "Y":
            multiplier *= y_ns / d_ns
            unit = "d"
        updated_frequency_parts.append(str(multiplier) + str(unit))

    return " ".join(updated_frequency_parts)


def _offset_to_timedelta(offset: Offset) -> pd.Timedelta:
    """
    Convert a pandas Offset to a pandas Timedelta.

    Parameters
    ----------
    offset : Offset
        The offset to be converted.

    Returns
    -------
    pd.Timedelta
        The corresponding timedelta.

    Examples
    --------
    ```pycon
    >>> _offset_to_timedelta(pd.offsets.Day(2))
    Timedelta('2 days 00:00:00')

    >>> _offset_to_timedelta(pd.offsets.Hour(3))
    Timedelta('0 days 03:00:00')
    ```
    """

    if isinstance(offset, (pd.offsets.BusinessHour, pd.offsets.CustomBusinessHour)):
        return pd.Timedelta(h_td * offset.n)

    if isinstance(offset, (pd.offsets.BusinessDay, pd.offsets.CustomBusinessDay)):
        return pd.Timedelta(d_td * offset.n)

    if isinstance(offset, pd.offsets.Week):
        return pd.Timedelta(w_td * offset.n)

    if isinstance(offset, (pd.offsets.SemiMonthBegin, pd.offsets.SemiMonthEnd)):
        return pd.Timedelta(semi_mo_td * offset.n)

    if isinstance(
        offset,
        (
            pd.offsets.MonthBegin,
            pd.offsets.MonthEnd,
            pd.offsets.BusinessMonthBegin,
            pd.offsets.BusinessMonthEnd,
            pd.offsets.CustomBusinessMonthBegin,
            pd.offsets.CustomBusinessMonthEnd,
            pd.offsets.WeekOfMonth,
            pd.offsets.LastWeekOfMonth,
        ),
    ):
        return pd.Timedelta(mo_td * offset.n)

    if isinstance(
        offset,
        (
            pd.offsets.QuarterBegin,
            pd.offsets.QuarterEnd,
            pd.offsets.BQuarterBegin,
            pd.offsets.BQuarterEnd,
            pd.offsets.FY5253Quarter,
        ),
    ):
        return pd.Timedelta(q_td * offset.n)

    if isinstance(
        offset,
        (
            pd.offsets.YearBegin,
            pd.offsets.YearEnd,
            pd.offsets.BYearBegin,
            pd.offsets.BYearEnd,
            pd.offsets.Easter,
            pd.offsets.FY5253,
        ),
    ):
        return pd.Timedelta(y_td * offset.n)

    return pd.Timedelta(offset)


def _fix_timedelta_precision(freq: pd.Timedelta) -> pd.Timedelta:
    """
    Fix the precision of a timedelta to nanoseconds.

    Parameters
    ----------
    freq : pd.Timedelta
        The timedelta whose precision is to be fixed.

    Returns
    -------
    pd.Timedelta
        The timedelta with nanosecond precision.

    Examples
    --------
    ```pycon
    >>> _fix_timedelta_precision(pd.Timedelta('1 days 02:03:04.567890123'))
    Timedelta('1 days 02:03:04.567890123')
    ```
    """
    if hasattr(freq, "unit") and freq.unit != "ns":
        freq = freq.as_unit("ns", round_ok=False)
    return freq


def to_timedelta(
    freq: tp.Union[str, int, Offset, pd.Timedelta],
    approximate: bool = False
) -> pd.Timedelta:
    """
    Convert a frequency-like object to a pandas Timedelta.

    Parameters
    ----------
    freq : str, int, Offset, or pd.Timedelta
        The frequency-like object to be converted.
    approximate : bool, optional
        If True, approximate the frequency using `_offset_to_timedelta`.

    Returns
    -------
    pd.Timedelta
        The corresponding timedelta.

    Examples
    --------
    ```pycon
    >>> to_timedelta('3D')
    Timedelta('3 days 00:00:00')

    >>> to_timedelta(pd.offsets.Hour(3))
    Timedelta('0 days 03:00:00')
    ```
    """
    if not isinstance(freq, pd.Timedelta):
        if isinstance(freq, str):
            freq = " ".join(freq.strip().split())

        if isinstance(freq, str) and freq.startswith("-"):
            neg_td = True
            freq = freq[1:]

        else:
            neg_td = False

        if isinstance(freq, str):
            freq = _prepare_timedelta_str(freq)

        if not isinstance(freq, Offset):
            try:
                if isinstance(freq, str) and not freq[0].isdigit():
                    freq = pd.Timedelta(1, unit=freq)
                else:
                    freq = pd.Timedelta(freq)

            except Exception as e1:
                try:
                    freq = to_offset(freq)
                except Exception:
                    raise e1

        if isinstance(freq, Offset):
            if approximate:
                freq = _offset_to_timedelta(freq)

            else:
                freq = pd.Timedelta(freq)

        if neg_td:
            freq = -freq
    return _fix_timedelta_precision(freq)


def to_timedelta64(
    freq: tp.Union[str, int, Offset, pd.Timedelta]
) -> np.timedelta64:
    """
    Convert a frequency-like object to a numpy timedelta64.

    Parameters
    ----------
    freq : str, int, Offset, or pd.Timedelta
        The frequency-like object to be converted.

    Returns
    -------
    np.timedelta64
        The corresponding numpy timedelta64.

    Examples
    --------
    ```pycon
    >>> to_timedelta64('1D')
    numpy.timedelta64(1,'D')

    >>> to_timedelta64(pd.offsets.Hour(3))
    numpy.timedelta64(10800000000000,'ns')
    ```
    """
    if not isinstance(freq, np.timedelta64):
        if not isinstance(freq, pd.Timedelta):
            freq = to_timedelta(freq)
        freq = freq.to_timedelta64()

    if freq.dtype != np.dtype("timedelta64[ns]"):
        return freq.astype("timedelta64[ns]")

    return freq


def time_to_timedelta(
    t: tp.Union[str, time, FreqHandler],
    **kwargs
) -> pd.Timedelta:
    """
    Convert a time-like object into `pd.Timedelta`.

    Parameters
    ----------
    t : str, time, or FreqHandler
        The time-like object to be converted.
    **kwargs : dict
        Additional arguments passed to `FreqHandler.parse_time_str` if 
        `t` is a string.

    Returns
    -------
    pd.Timedelta
        The resulting `pd.Timedelta`.

    Raises
    ------
    ValueError
        If the time string has a date or weekday component, or lacks a time 
        component.

    Examples
    --------
    ```pycon
    >>> time_to_timedelta("12:34:56")
    Timedelta('0 days 12:34:56')

    >>> time_to_timedelta(time(12, 34, 56))
    Timedelta('0 days 12:34:56')
    ```
    """
    if isinstance(t, str):
        t = FreqHandler.parse_time_str(t, **kwargs)

    if isinstance(t, FreqHandler):
        if t.has_date():
            raise ValueError("Time string has a date component")

        if t.has_weekday():
            raise ValueError("Time string has a weekday component")

        if not t.has_time():
            raise ValueError("Time string doesn't have a time component")

        t = t.to_time()

    return pd.Timedelta(
        hours=t.hour if t.hour is not None else 0,
        minutes=t.minute if t.minute is not None else 0,
        seconds=t.second if t.second is not None else 0,
        milliseconds=(
            t.microsecond // 1000
        ) 
        if t.microsecond is not None else 0,
        microseconds=(
            t.microsecond % 1000
        ) 
        if t.microsecond is not None else 0,
    )


def to_timestamp(
    dt: tp.Union[str, int, pd.Timestamp],
    parse_with_dateparser: tp.Optional[bool] = True,
    dateparser_kwargs: tp.Union[None, tp.Dict[str, tp.Any]] = None,
    unit: str = "ns",
    **kwargs,
) -> tp.Optional[pd.Timestamp]:
    """
    Parse the input as a `pd.Timestamp`.

    Parameters
    ----------
    dt : str, int, or pd.Timestamp
        The datetime-like object to be parsed.
    parse_with_dateparser : bool, optional
        If True, use dateparser to parse string inputs.
    dateparser_kwargs : dict, optional
        Additional arguments passed to `dateparser.parse`.
    unit : str, default "ns"
        The unit of the input when `dt` is a number.
    **kwargs : dict
        Additional arguments passed to `pd.Timestamp`.

    Returns
    -------
    pd.Timestamp
        The parsed `pd.Timestamp`.

    Raises
    ------
    ValueError
        If the input cannot be parsed as a `pd.Timestamp`.

    Notes
    -----
    This function supports various datetime-like inputs and attempts to parse
    them into a `pd.Timestamp`. If the input is a string, both Pandas and 
    dateparser (if enabled) will be used for parsing. The `tz` argument allows
    specifying the desired time zone for the output.

    Examples
    --------
    ```pycon
    >>> to_timestamp("2023-01-01 00:00:00")
    Timestamp('2023-01-01 00:00:00')

    >>> to_timestamp(1672531200000, unit='ms')
    Timestamp('2023-01-01 00:00:00')

    >>> to_timestamp("now")
    Timestamp('2023-06-19 12:34:56.789012')

    ```
    """
    dateparser_kwargs = dateparser_kwargs or {}

    if checks.is_number(dt):
        dt = pd.Timestamp(dt, tz="utc", unit=unit, **kwargs)

    elif isinstance(dt, str):
        dt = " ".join(dt.strip().split())

        if dt.lower() == "now":
            dt = pd.Timestamp.now()

        elif dt.lower() == "today":
            dt = pd.Timestamp.now().floor("1D")

        elif dt.lower() == "yesterday":
            dt = pd.Timestamp.now().floor("1D") - pd.Timedelta(days=1)

        elif dt.lower() == "tomorrow":
            dt = pd.Timestamp.now().floor("1D") + pd.Timedelta(days=1)

        else:
            try:
                dt = pd.Timestamp(dt, **kwargs)

            except Exception:
                if parse_with_dateparser:
                    try:
                        import dateparser

                        settings = dateparser_kwargs.get("settings", {})
                        settings["RELATIVE_BASE"] = settings.get(
                            "RELATIVE_BASE",
                            pd.Timestamp.now().to_pydatetime(),
                        )
                        dateparser_kwargs["settings"] = settings
                        dt = dateparser.parse(dt, **dateparser_kwargs)
                        
                        if dt is not None:
                            dt = pd.Timestamp(dt, **kwargs)
                            
                        else:
                            raise ValueError(
                                f"Could not parse the timestamp {dt}"
                            )

                    except Exception:
                        raise ValueError(f"Could not parse the timestamp {dt}")

                else:
                    raise ValueError(f"Could not parse the timestamp {dt}")

    elif not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt, **kwargs)

    return dt


def to_datetime(dt: tp.Union[str, int, pd.Timestamp], **kwargs) -> datetime:
    """
    Parse the input as a `datetime.datetime`.

    Parameters
    ----------
    dt : str, int, or pd.Timestamp
        The datetime-like object to be parsed.
    **kwargs : dict
        Additional arguments passed to `to_timestamp`.

    Returns
    -------
    datetime.datetime
        The parsed `datetime.datetime` object.

    Examples
    --------
    >>> to_datetime("2023-01-01 00:00:00")
    datetime.datetime(2023, 1, 1, 0, 0)

    >>> to_datetime(1672531200000, unit='ms')
    datetime.datetime(2023, 1, 1, 0, 0)
    """
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
        
    return to_timestamp(dt, **kwargs).to_pydatetime()


def to_freq(
    freq: tp.Optional[tp.Union[str, int, Offset, pd.Timedelta]],
    allow_offset: bool = True,
    keep_offset: bool = False
) -> tp.Union[Offset, pd.Timedelta]:
    """
    Convert a frequency-like object to a pandas DateOffset or Timedelta.

    Parameters
    ----------
    freq : str, int, Offset, or pd.Timedelta, optional
        The frequency-like object to be converted.
    allow_offset : bool, optional
        If True, allow conversion to pandas DateOffset.
    keep_offset : bool, optional
        If True, keep the original DateOffset if applicable.

    Returns
    -------
    Offset or pd.Timedelta
        The corresponding DateOffset or Timedelta.

    Examples
    --------
    ```pycon
    >>> to_freq('1D')
    Timedelta('1 days 00:00:00')

    >>> to_freq(pd.offsets.Hour(3))
    Timedelta('0 days 03:00:00')
    ```
    """
    if isinstance(freq, pd.Timedelta):
        return freq

    if allow_offset and isinstance(freq, Offset):
        if not keep_offset:
            try:
                td_freq = to_timedelta(freq)
                if to_offset(td_freq) == freq:
                    freq = td_freq
                else:
                    warnings.warn(f"Ambiguous frequency {freq}", stacklevel=2)
            except Exception:
                pass
        return freq

    if allow_offset:
        try:
            return to_freq(
                to_offset(freq),
                allow_offset=True,
                keep_offset=keep_offset
            )

        except Exception:
            return to_timedelta(freq)

    return to_timedelta(freq)


def date_range(
    start: tp.Optional[tp.Union[str, int, pd.Timestamp]] = None,
    end: tp.Optional[tp.Union[str, int, pd.Timestamp]] = None,
    *,
    periods: tp.Optional[int] = None,
    freq: tp.Optional[tp.Union[str, int, Offset, pd.Timedelta]] = None,
    inclusive: str = "left",
    timestamp_kwargs: tp.Union[tp.Dict[str, tp.Any]] = None,
    freq_kwargs: tp.Union[tp.Dict[str, tp.Any]] = None,
    **kwargs,
) -> pd.DatetimeIndex:
    """
    Generate a fixed frequency DatetimeIndex, preprocessing the inputs.

    Parameters
    ----------
    start : str, int, or pd.Timestamp, optional
        The start time for the date range.
    end : str, int, or pd.Timestamp, optional
        The end time for the date range.
    periods : int, optional
        Number of periods to generate.
    freq : str, int, Offset, or pd.Timedelta, optional
        Frequency string or frequency object.
    tz : str, int, float, timedelta, tzinfo, optional
        Time zone for the resulting DatetimeIndex.
    inclusive : str, default "left"
        Include boundaries; "left", "right", "both", or "neither".
    timestamp_kwargs : dict, optional
        Additional arguments passed to `to_timestamp`.
    freq_kwargs : dict, optional
        Additional arguments passed to `to_freq`.
    **kwargs : dict
        Additional arguments passed to `pd.date_range`.

    Returns
    -------
    pd.DatetimeIndex
        A fixed frequency DatetimeIndex.

    Notes
    -----
    This function preprocesses `start` and `end` with `to_timestamp`, `freq`
    with `to_freq`, and `tz` with `to_timezone`.

    If `start` and `periods` are None, `start` is set to the beginning of the 
    Unix epoch. Similarly, if `end` and `periods` are None, `end` is set to the 
    current date and time.

    Examples
    --------
    ```pycon
    >>> date_range(start='2022-01-01', end='2022-01-10', freq='D')
    DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
                   '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                   '2022-01-09', '2022-01-10'],
                  dtype='datetime64[ns]', freq='D')

    >>> date_range(periods=5, freq='H')
    DatetimeIndex(['1970-01-01 00:00:00+00:00', '1970-01-01 01:00:00+00:00',
                   '1970-01-01 02:00:00+00:00', '1970-01-01 03:00:00+00:00',
                   '1970-01-01 04:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq='H')
    ```
    """
    timestamp_kwargs = timestamp_kwargs or {}
    freq_kwargs = freq_kwargs or {}

    if freq is None and (start is None or end is None or periods is None):
        freq = "1D"

    if freq is not None:
        freq = to_freq(freq, **freq_kwargs)

    if start is not None:
        start = to_timestamp(start, **timestamp_kwargs)

    if end is not None:
        end = to_timestamp(end, **timestamp_kwargs)

    if periods is None:
        if start is None:
            start = to_timestamp(0, **timestamp_kwargs).tz_localize(None)

        if end is None:
            end = to_timestamp("now", **timestamp_kwargs).tz_localize(None)
    else:
        if start is None and end is None:
            start = to_timestamp(0, **timestamp_kwargs).tz_localize(None)

    return pd.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        inclusive=inclusive,
        **kwargs,
    )


def prepare_dt_index(
    index: tp.Union[range, tp.Sequence[tp.Any]],
    parse_index: tp.Optional[bool] = True,
    parse_with_dateparser: tp.Optional[bool] = True,
    dateparser_kwargs: tp.Union[None, tp.Dict[str, tp.Any]] = None,
    **kwargs,
) -> pd.Index:
    """
    Try converting an index to a datetime index.

    Parameters
    ----------
    index : range or sequence of any
        The index to be converted to a datetime index.
    parse_index : bool, optional
        If True, attempts to parse the index using Pandas and dateparser. 
        Default is True.
    parse_with_dateparser : bool, optional
        If True, uses dateparser for parsing in addition to Pandas. 
        Default is True.
    dateparser_kwargs : dict, optional
        Arguments to pass to `dateparser.parse`. Default is None.
    **kwargs : dict
        Additional arguments to pass to `pd.to_datetime`.

    Returns
    -------
    pd.Index
        The converted datetime index.

    Raises
    ------
    Exception
        If parsing fails and `parse_index` and `parse_with_dateparser` are both 
        True.

    Notes
    -----
    `dateparser_kwargs` are passed to `dateparser.parse` while `**kwargs` are
    passed to `pd.to_datetime`.

    Examples
    --------
    ```pycon
    >>> prepare_dt_index(['2020-01-01', '2020-01-02'])
    DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]', freq=None)

    >>> prepare_dt_index(range(10))
    Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='object')
    ```
    """
    dateparser_kwargs = dateparser_kwargs or {}

    if not isinstance(index, pd.Index):
        if isinstance(index, str):
            if parse_index:
                try:
                    parsed_index = pd.to_datetime(index, **kwargs)
                    if not isinstance(parsed_index, pd.Timestamp) and "utc" not in kwargs:
                        parsed_index = pd.to_datetime(
                            index,
                            utc=True,
                            **kwargs
                        )
                    index = [parsed_index]

                except Exception:
                    if parse_with_dateparser:
                        try:
                            parsed_index = dateparser.parse(
                                index,
                                **dateparser_kwargs
                            )
                            if parsed_index is None:
                                raise Exception

                            index = pd.to_datetime(parsed_index, **kwargs)
                            index = [index]

                        except Exception:
                            pass

        try:
            index = pd.Index(index)

        except Exception:
            index = pd.Index([index])

    if isinstance(index, pd.DatetimeIndex):
        return index

    if index.dtype == object:
        if parse_index:
            try:
                pd.to_datetime(index[[0]], **kwargs)
                try:
                    parsed_index = pd.to_datetime(index, **kwargs)
                    if (
                        not isinstance(parsed_index, pd.DatetimeIndex)
                        and isinstance(parsed_index[0], datetime)
                        and "utc" not in kwargs
                    ):
                        parsed_index = pd.to_datetime(
                            index,
                            utc=True,
                            **kwargs
                        )
                    return parsed_index

                except Exception:
                    if parse_with_dateparser:
                        try:
                            def wrapper(x):
                                i = dateparser.parse(x, **dateparser_kwargs)
                                if i is None:
                                    raise Exception
                                return i

                            return pd.to_datetime(index.map(wrapper), **kwargs)

                        except Exception:
                            pass

            except Exception:
                pass

    return index


def try_align_to_dt_index(
    source_index: tp.Union[range, tp.Sequence[tp.Any]],
    target_index: pd.Index,
    **kwargs
) -> pd.Index:
    """
    Try aligning an index to another datetime index.

    Parameters
    ----------
    source_index : range or sequence of any
        The source index to be aligned.
    target_index : pd.Index
        The target datetime index to align to.
    **kwargs : dict
        Additional arguments passed to `prepare_dt_index`.

    Returns
    -------
    pd.Index
        The aligned datetime index.

    Notes
    -----
    Keyword arguments are passed to `prepare_dt_index`.

    Examples
    --------
    ```pycon
    >>> source_index = ['2020-01-01', '2020-01-02']
    >>> target_index = pd.date_range(start='2020-01-01', periods=2, freq='D')
    >>> try_align_to_dt_index(source_index, target_index)
    DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]', freq=None)
    ```
    """
    source_index = prepare_dt_index(source_index, **kwargs)
    if isinstance(source_index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if source_index.tz is None and target_index.tz is not None:
            source_index = source_index.tz_localize(target_index.tz)

        elif source_index.tz is not None and target_index.tz is not None:
            source_index = source_index.tz_convert(target_index.tz)

    return source_index


def try_align_dt_to_index(
    dt: tp.Union[str, int, pd.Timestamp],
    target_index: pd.Index,
    **kwargs
) -> tp.Union[str, int, pd.Timestamp]:
    """
    Try aligning a datetime-like object to another datetime index.

    Parameters
    ----------
    dt : str, int, or pd.Timestamp
        The datetime-like object to align.
    target_index : pd.Index
        The target datetime index to align to.
    **kwargs : dict
        Additional arguments passed to `to_timestamp`.

    Returns
    -------
    str, int, or pd.Timestamp
        The aligned datetime-like object.

    Notes
    -----
    Keyword arguments are passed to `to_timestamp`.

    Examples
    --------
    ```pycon
    >>> dt = '2020-01-01'
    >>> target_index = pd.date_range(start='2020-01-01', periods=1, freq='D')
    >>> try_align_dt_to_index(dt, target_index)
    Timestamp('2020-01-01 00:00:00', freq='D')
    ```
    """
    if not isinstance(target_index, pd.DatetimeIndex):
        return dt

    dt = to_timestamp(dt, **kwargs)

    if dt.tzinfo is None and target_index.tz is not None:
        dt = dt.tz_localize(target_index.tz)

    elif dt.tzinfo is not None and target_index.tz is not None:
        dt = dt.tz_convert(target_index.tz)

    return dt


@jit(cache=True)
def _min_count_nb(arr: np.ndarray) -> tp.Tuple[int, float, int]:
    """
    Get the first position, the value, and the count of the array's minimum.

    """
    mini = 0
    minv = arr[0]
    minc = 1
    for i in range(1, len(arr)):
        if arr[i] == minv:
            minc += 1
        elif arr[i] < minv:
            mini = i
            minv = arr[i]
            minc = 1
    return mini, minv, minc


def auto_detect_freq(index: pd.Index) -> tp.Union[Offset, pd.Timedelta]:
    """
    Auto-detect frequency from a datetime index.

    Parameters
    ----------
    index : pd.Index
        The datetime index from which to detect frequency.

    Returns
    -------
    Offset or pd.Timedelta
        The detected frequency.

    Notes
    -----
    Returns the minimal frequency if it is encountered in most of the index.

    Examples
    --------
    ```pycon
    >>> index = pd.date_range(start='2020-01-01', periods=10, freq='D')
    >>> auto_detect_freq(index)
    Timedelta('1 days 00:00:00')
    ```
    """
    diff_values = index.values[1:] - index.values[:-1]

    if len(diff_values) > 0:
        mini, _, minc = _min_count_nb(diff_values)

        if minc / len(index) > 0.5:
            return index[mini + 1] - index[mini]

    return None


def parse_index_freq(index: pd.DatetimeIndex) -> tp.Union[Offset, pd.Timedelta]:
    """
    Parse frequency from a datetime index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The datetime index from which to parse frequency.

    Returns
    -------
    Offset or pd.Timedelta
        The parsed frequency.

    Examples
    --------
    ```pycon
    >>> index = pd.date_range(start='2020-01-01', periods=10, freq='D')
    >>> parse_index_freq(index)
    Timedelta('1 days 00:00:00')
    ```
    """
    if index.freqstr is not None:
        return to_freq(index.freqstr)

    if index.freq is not None:
        return to_freq(index.freq)

    if len(index) >= 3:
        freq = pd.infer_freq(index)

        if freq is not None:
            return to_freq(freq)

    return None


def infer_index_freq(
    index: pd.Index,
    freq: tp.Optional[tp.Union[str, int, Offset, pd.Timedelta]] = None,
    allow_offset: bool = True,
    allow_numeric: bool = True,
    freq_from_n: tp.Union[None, bool, int] = 20,
) -> tp.Union[None, int, float, tp.Union[Offset, pd.Timedelta]]:
    """
    Infer frequency of a datetime index if `freq` is None, otherwise convert to
    the required frequency.

    Parameters
    ----------
    index : pd.Index
        The index from which to infer frequency.
    freq : str, int, Offset, pd.Timedelta, optional
        The frequency to be used or converted. 
        * If None, the frequency is inferred from the index. 
        * If "auto", the frequency is automatically detected. 
        * If "index_[method_name]", the method is applied to the 
        `pd.TimedeltaIndex` derived from the differences between each pair of 
        index points.
    allow_offset : bool, default=True
        If True, allows the output frequency to be a `Offset` instance.
    allow_numeric : bool, default=True
        If True, allows the output frequency to be a numeric value (int or float).
    freq_from_n : None, bool, int, default=20
        If an int, limits the index to the first or last N index points 
        respectively for frequency inference. If None, the entire index is 
        considered. If a boolean, raises a ValueError if True.

    Returns
    -------
    None, int, float, Offset, pd.Timedelta
        The inferred or converted frequency of the index. The type of the 
        return value depends on the input parameters and the inferred frequency.
        Returns None if the frequency cannot be determined.

    Raises
    ------
    ValueError
        If `freq_from_n` is True.

    Examples
    --------
    ```pycon
    >>> index = pd.date_range(start='2020-01-01', periods=100, freq='D')
    >>> infer_index_freq(index, freq='auto')
    'D'
    ```
    """
    if isinstance(freq_from_n, bool):
        if freq_from_n:
            raise ValueError("freq_from_n cannot be True")
        freq_from_n = None

    if isinstance(index, pd.DatetimeIndex):
        if freq is None:
            freq = parse_index_freq(index)

        elif isinstance(freq, str):

            if freq == "auto" or freq.startswith("index_"):
                updated_frequency = parse_index_freq(index)

                if updated_frequency is not None:
                    freq = updated_frequency

                else:
                    if freq_from_n is None:
                        index_subset = index

                    else:
                        if freq_from_n >= 0:
                            index_subset = index[:freq_from_n]

                        else:
                            index_subset = index[freq_from_n:]

                    if freq.lower() == "auto":
                        freq = auto_detect_freq(index_subset)

                    else:
                        method_name = freq.lower().replace("index_", "")
                        freq = getattr(
                            index_subset[1:] - index_subset[:-1],
                            method_name
                        )()
    if freq is None:
        return None

    if checks.is_number(freq) and allow_numeric:
        return freq

    return to_freq(freq, allow_offset=allow_offset)
