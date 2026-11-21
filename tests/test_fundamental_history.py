"""Tests for publication-date-aware fundamental lookup."""

from datetime import date

import pytest

from app.data.fundamental_history import (
    FundamentalHistoryError,
    PointInTimeFundamentals,
)


def test_future_filing_cannot_affect_prior_as_of_snapshot():
    history = PointInTimeFundamentals.from_records([
        {
            "symbol": "ABC", "metric": "roe", "value": 10,
            "period_end": "2020-03-31", "available_at": "2020-06-01",
            "source": "filing-2020",
        },
        {
            "symbol": "ABC", "metric": "roe", "value": 20,
            "period_end": "2021-03-31", "available_at": "2021-06-01",
            "source": "filing-2021",
        },
    ])
    assert history.as_of(date(2020, 12, 31)) == {"ABC": {"roe": 10.0}}
    assert history.source_for("ABC", "roe", date(2020, 12, 31)) == "filing-2020"


def test_duplicate_as_reported_observation_fails_closed():
    record = {
        "symbol": "ABC", "metric": "eps", "value": 1,
        "period_end": "2020-03-31", "available_at": "2020-06-01",
        "source": "filing",
    }
    with pytest.raises(FundamentalHistoryError, match="Duplicate"):
        PointInTimeFundamentals.from_records([record, record])
