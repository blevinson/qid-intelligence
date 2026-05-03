#!/usr/bin/env python3
"""
Tests for polymarket_ingest.py — pure-function unit tests + mock pipeline test.
Run: python -m pytest backend/scripts/test_polymarket_ingest.py -v
  or: python backend/scripts/test_polymarket_ingest.py
"""

import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(__file__))

from polymarket_ingest import (
    matches_theme,
    parse_float,
    parse_ts,
    end_date_str,
    fetch_all_theme_markets,
    upsert_markets,
    insert_price_snaps,
)


# ── matches_theme ─────────────────────────────────────────────────────────────

class TestMatchesTheme:
    def test_single_keyword_match(self):
        result = matches_theme("Will Iran launch a strike before year end?")
        assert "iran" in result

    def test_multiple_keyword_match(self):
        result = matches_theme("Russia Ukraine ceasefire by Q3?")
        assert "russia" in result
        assert "ukraine" in result
        assert "ceasefire" in result

    def test_case_insensitive(self):
        assert matches_theme("TAIWAN strait conflict") == matches_theme("taiwan strait conflict")
        assert len(matches_theme("TAIWAN strait conflict")) > 0

    def test_no_match_returns_empty(self):
        assert matches_theme("Will the Lakers win the championship?") == []

    def test_empty_string(self):
        assert matches_theme("") == []

    def test_partial_keyword_match(self):
        # "escalat" matches "escalation", "escalate", etc.
        assert "escalat" in matches_theme("Escalation risk in the South China Sea?")

    def test_substring_match_in_word(self):
        # "gold" should match "gold price forecast"
        assert "gold" in matches_theme("Will gold prices hit $3000?")


# ── parse_float ───────────────────────────────────────────────────────────────

class TestParseFloat:
    def test_string_float(self):
        assert parse_float("0.75") == 0.75

    def test_integer_string(self):
        assert parse_float("1") == 1.0

    def test_actual_float(self):
        assert parse_float(0.5) == 0.5

    def test_none_returns_none(self):
        assert parse_float(None) is None

    def test_empty_string_returns_none(self):
        assert parse_float("") is None

    def test_non_numeric_string_returns_none(self):
        assert parse_float("abc") is None

    def test_zero(self):
        assert parse_float("0") == 0.0
        assert parse_float(0) == 0.0


# ── parse_ts ──────────────────────────────────────────────────────────────────

class TestParseTs:
    def test_z_format(self):
        result = parse_ts("2025-06-15T12:00:00Z")
        assert result == datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_z_format_with_microseconds(self):
        result = parse_ts("2025-06-15T12:00:00.123456Z")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6
        assert result.tzinfo is not None

    def test_tz_offset_format(self):
        result = parse_ts("2025-06-15T12:00:00+00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_none_returns_none(self):
        assert parse_ts(None) is None

    def test_empty_string_returns_none(self):
        assert parse_ts("") is None

    def test_invalid_string_returns_none(self):
        assert parse_ts("not-a-date") is None

    def test_naive_datetime_gets_utc(self):
        # "%Y-%m-%dT%H:%M:%SZ" parses as naive then gets UTC attached
        result = parse_ts("2025-01-01T00:00:00Z")
        assert result.tzinfo == timezone.utc


# ── end_date_str ──────────────────────────────────────────────────────────────

class TestEndDateStr:
    def test_normal_date(self):
        assert end_date_str({"endDate": "2025-12-31T00:00:00Z"}) == "2025-12-31"

    def test_date_only(self):
        assert end_date_str({"endDate": "2025-12-31"}) == "2025-12-31"

    def test_missing_key_returns_none(self):
        assert end_date_str({}) is None

    def test_none_value_returns_none(self):
        assert end_date_str({"endDate": None}) is None

    def test_empty_string_returns_none(self):
        assert end_date_str({"endDate": ""}) is None


# ── Pipeline mock test ────────────────────────────────────────────────────────

class TestPipeline:
    """Mock-based end-to-end test of fetch → upsert_markets → insert_price_snaps."""

    FAKE_MARKETS_PAGE = [
        {
            "conditionId": "mkt-001",
            "question": "Will Iran attack Israel before 2026?",
            "bestBid": "0.45",
            "bestAsk": "0.55",
            "lastTradePrice": "0.50",
            "volume24hr": "125000.0",
            "endDate": "2025-12-31T00:00:00Z",
            "resolved": False,
            "closed": False,
            "startDate": "2024-01-01T00:00:00Z",
        },
        {
            "conditionId": "mkt-002",
            "question": "Will the Lakers win the 2025 championship?",  # no theme match
            "bestBid": "0.20",
            "bestAsk": "0.22",
            "lastTradePrice": "0.21",
            "volume24hr": "5000.0",
            "endDate": "2025-06-30T00:00:00Z",
            "resolved": False,
            "closed": False,
        },
        {
            "conditionId": "mkt-003",
            "question": "Ukraine ceasefire by Q2 2025?",
            "bestBid": "0.30",
            "bestAsk": "0.35",
            "lastTradePrice": "0.32",
            "volume24hr": "88000.0",
            "endDate": "2025-06-30T00:00:00Z",
            "resolved": False,
            "closed": False,
            "startDate": "2024-06-01T00:00:00Z",
        },
    ]

    def _make_conn(self):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        return conn, cursor

    def test_fetch_all_theme_markets_filters_correctly(self):
        with patch("polymarket_ingest.fetch_markets_page") as mock_fetch:
            # First call (active page 0) returns 3 markets; second call returns empty → stop
            mock_fetch.side_effect = [self.FAKE_MARKETS_PAGE, []]

            result = fetch_all_theme_markets()

        # Only mkt-001 and mkt-003 match geopolitical themes
        assert len(result) == 2
        ids = {m["_market_id"] for m in result}
        assert ids == {"mkt-001", "mkt-003"}

    def test_fetch_all_theme_markets_tags_attached(self):
        with patch("polymarket_ingest.fetch_markets_page") as mock_fetch:
            mock_fetch.side_effect = [self.FAKE_MARKETS_PAGE, []]
            result = fetch_all_theme_markets()

        mkt001 = next(m for m in result if m["_market_id"] == "mkt-001")
        assert "iran" in mkt001["_matched_tags"]
        assert "israel" in mkt001["_matched_tags"]

    def test_upsert_markets_returns_row_count(self):
        conn, cursor = self._make_conn()

        with patch("polymarket_ingest.fetch_markets_page") as mock_fetch, \
             patch("psycopg2.extras.execute_values"):
            mock_fetch.side_effect = [self.FAKE_MARKETS_PAGE, []]
            markets = fetch_all_theme_markets()
            count = upsert_markets(conn, markets)

        assert count == 2
        conn.commit.assert_called_once()

    def test_insert_price_snaps_returns_row_count(self):
        conn, cursor = self._make_conn()
        snap_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        with patch("polymarket_ingest.fetch_markets_page") as mock_fetch, \
             patch("psycopg2.extras.execute_values"):
            mock_fetch.side_effect = [self.FAKE_MARKETS_PAGE, []]
            markets = fetch_all_theme_markets()
            count = insert_price_snaps(conn, markets, snap_time)

        assert count == 2
        conn.commit.assert_called_once()

    def test_full_pipeline_mock(self):
        """Smoke test: fetch → upsert_markets → insert_price_snaps without DB errors."""
        conn, cursor = self._make_conn()
        snap_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        with patch("polymarket_ingest.fetch_markets_page") as mock_fetch, \
             patch("psycopg2.extras.execute_values"):
            mock_fetch.side_effect = [self.FAKE_MARKETS_PAGE, [], self.FAKE_MARKETS_PAGE, []]
            markets = fetch_all_theme_markets()
            n_markets = upsert_markets(conn, markets)
            n_snaps = insert_price_snaps(conn, markets, snap_time)

        assert n_markets == 2
        assert n_snaps == 2
        assert conn.commit.call_count == 2


# ── Entry point for direct execution ─────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suites = [
        TestMatchesTheme,
        TestParseFloat,
        TestParseTs,
        TestEndDateStr,
        TestPipeline,
    ]
    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        for name in [m for m in dir(suite_cls) if m.startswith("test_")]:
            try:
                getattr(suite, name)()
                print(f"  PASS  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite_cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
