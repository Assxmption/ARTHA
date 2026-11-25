"""Tests for point-in-time security identity and membership."""

from datetime import date

import pytest

from app.data.security_master import (
    PointInTimeSecurityMaster,
    SecurityMasterError,
)


@pytest.fixture
def master():
    return PointInTimeSecurityMaster.from_records([
        {
            "security_id": "ISSUER-A",
            "canonical_symbol": "ALPHA",
            "listed_from": "2010-01-01",
            "delisted_on": "2020-06-30",
            "aliases": "ALPHA_OLD;ALPHA.NS",
        },
        {
            "security_id": "ISSUER-B",
            "canonical_symbol": "BETA",
            "listed_from": "2018-01-01",
        },
    ])


def test_membership_is_as_of_and_aliases_canonicalize(master):
    assert master.symbols_at(date(2015, 1, 1)) == ("ALPHA",)
    assert master.symbols_at(date(2019, 1, 1)) == ("ALPHA", "BETA")
    assert master.symbols_at(date(2021, 1, 1)) == ("BETA",)
    assert master.canonicalize("alpha_old") == "ALPHA"
    assert master.canonicalize("ALPHA.NS") == "ALPHA"


def test_future_only_security_cannot_change_prior_snapshot(master):
    before = master.snapshot(date(2015, 1, 1))
    after_master = PointInTimeSecurityMaster.from_records([
        {
            "security_id": "ISSUER-A",
            "canonical_symbol": "ALPHA",
            "listed_from": "2010-01-01",
        },
        {
            "security_id": "ISSUER-FUTURE",
            "canonical_symbol": "FUTURE",
            "listed_from": "2025-01-01",
        },
    ])
    after = after_master.snapshot(date(2015, 1, 1))
    assert before["active_symbols"] == after["active_symbols"]


def test_alias_collision_fails_closed():
    with pytest.raises(SecurityMasterError, match="multiple securities"):
        PointInTimeSecurityMaster.from_records([
            {"security_id": "A", "canonical_symbol": "ALPHA", "listed_from": "2010-01-01"},
            {"security_id": "B", "canonical_symbol": "BETA", "listed_from": "2010-01-01", "aliases": "ALPHA"},
        ])


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(SecurityMasterError, match="not found"):
        PointInTimeSecurityMaster.from_file(tmp_path / "missing.json")
