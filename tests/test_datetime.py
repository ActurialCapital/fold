from datetime import (
    datetime as _datetime,
    timedelta as _timedelta,
    timezone as _timezone
)
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest

from fold.utils import datetime as dt


def test_to_offsets():
    assert dt.to_offset("d") == to_offset("1d")
    assert dt.to_offset("day") == to_offset("1d")
    assert dt.to_offset("m") == to_offset("1min")
    assert dt.to_offset("1m") == to_offset("1min")
    assert dt.to_offset("1 m") == to_offset("1min")
    assert dt.to_offset("1 minute") == to_offset("1min")
    assert dt.to_offset("2 minutes") == to_offset("2min")
    assert dt.to_offset("1 hour, 2 minutes") == to_offset("1h 2min")
    assert dt.to_offset("1 hour; 2 minutes") == to_offset("1h 2min")
    assert dt.to_offset("2 weeks") == pd.offsets.Week(weekday=0) * 2
    assert dt.to_offset("2 months") == pd.offsets.MonthBegin() * 2
    assert dt.to_offset("2 quarter") == pd.offsets.QuarterBegin(startingMonth=1) * 2
    assert dt.to_offset("2 years") == pd.offsets.YearBegin() * 2


def test_to_timedelta():
    assert dt.to_timedelta("d") == pd.to_timedelta("1d")
    assert dt.to_timedelta("day") == pd.to_timedelta("1d")
    assert dt.to_timedelta("m") == pd.to_timedelta("1min")
    assert dt.to_timedelta("1m") == pd.to_timedelta("1min")
    assert dt.to_timedelta("1 m") == pd.to_timedelta("1min")
    assert dt.to_timedelta("1 minute") == pd.to_timedelta("1min")
    assert dt.to_timedelta("2 minutes") == pd.to_timedelta("2min")
    assert dt.to_timedelta("1 hour, 2 minutes") == pd.to_timedelta("1h 2min")
    assert dt.to_timedelta("1 hour; 2 minutes") == pd.to_timedelta("1h 2min")
    assert dt.to_timedelta("2 weeks") == pd.Timedelta(days=14)
    assert dt.to_timedelta("2 months") == pd.Timedelta(days=365.2425 / 12 * 2)
    assert dt.to_timedelta("2 quarter") == pd.Timedelta(days=365.2425 / 4 * 2)
    assert dt.to_timedelta("2 years") == pd.Timedelta(days=365.2425 * 2)


# def test_get_local_tz():
#     assert dt.get_local_tz().utcoffset(
#         _datetime.now()) == _datetime.now().astimezone(None).utcoffset()


# def test_to_timezone():
#     assert dt.to_timezone("utc", to_fixed_offset=True) == _timezone.utc
#     assert isinstance(dt.to_timezone("Europe/Berlin", to_fixed_offset=True), _timezone)
#     assert dt.to_timezone("+0500") == _timezone(_timedelta(hours=5))
#     assert dt.to_timezone(_timezone(_timedelta(hours=1))) == _timezone(_timedelta(hours=1))
#     assert dt.to_timezone(3600) == _timezone(_timedelta(hours=1))
#     assert dt.to_timezone(1800) == _timezone(_timedelta(hours=0.5))
#     with pytest.raises(Exception):
#         dt.to_timezone("+05")
