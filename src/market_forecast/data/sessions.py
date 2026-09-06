"""Exchange-calendar helpers used for staleness checks and gap detection."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

_LOOKAHEAD_DAYS = 5


@lru_cache(maxsize=4)
def _calendar(name: str) -> mcal.MarketCalendar:
    return mcal.get_calendar(name)


def trading_sessions(start: date, end: date, calendar: str = "XNYS") -> pd.DatetimeIndex:
    schedule = _calendar(calendar).schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index).normalize()


def last_closed_session(calendar: str = "XNYS", now: datetime | None = None) -> pd.Timestamp:
    """Most recent session whose closing auction has already happened."""
    reference = pd.Timestamp(now or datetime.now(tz=None)).tz_localize("UTC")
    end = (reference + timedelta(days=_LOOKAHEAD_DAYS)).date()
    start = (reference - timedelta(days=30)).date()
    schedule = _calendar(calendar).schedule(start_date=start, end_date=end)
    closed = schedule[schedule["market_close"] <= reference]
    if closed.empty:
        raise RuntimeError(f"no closed session found for calendar {calendar}")
    return pd.Timestamp(closed.index[-1]).normalize()


def missing_sessions(index: pd.DatetimeIndex, calendar: str = "XNYS") -> pd.DatetimeIndex:
    """Sessions the exchange held between the first and last observation but the data lacks."""
    if len(index) == 0:
        return pd.DatetimeIndex([])
    normalised = pd.DatetimeIndex(index).normalize()
    expected = trading_sessions(normalised.min().date(), normalised.max().date(), calendar)
    return expected.difference(normalised)


def unexpected_sessions(index: pd.DatetimeIndex, calendar: str = "XNYS") -> pd.DatetimeIndex:
    """Rows dated on days the exchange was closed, which vendors occasionally emit."""
    if len(index) == 0:
        return pd.DatetimeIndex([])
    normalised = pd.DatetimeIndex(index).normalize()
    expected = trading_sessions(normalised.min().date(), normalised.max().date(), calendar)
    return normalised.difference(expected)
