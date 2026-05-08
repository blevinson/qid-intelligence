"""Tests for _fetch_crucix_idea_counts in panel_build.py (FIN-1584 / TFARM-99).

Covers:
- Happy path: rows from crucix_trade_ideas produce correct n_ideas_24h / n_ideas_7d
- 7-day rolling: idea on day D counts in n_ideas_7d for days D..D+6
- Empty table fallback: returns empty DataFrame (caller 0-fills)
- Exception path: cursor error returns empty DataFrame
- Regression guard: dt column is named "dt" not "index" (melt_to_long bug FIN-1584)
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Adjust import path so the script can be imported standalone.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from panel_build import _fetch_crucix_idea_counts


def _make_conn(rows: list[dict]) -> MagicMock:
    """Return a mock psycopg2 connection whose cursor yields `rows`."""
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows

    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_single_ticker_single_day():
    start = date(2026, 5, 1)
    end = date(2026, 5, 1)
    rows = [{"ticker": "NVDA", "idea_date": date(2026, 5, 1), "n_ideas": 3}]
    conn = _make_conn(rows)

    result = _fetch_crucix_idea_counts(conn, start, end)

    assert not result.empty
    assert set(result.columns) >= {"ticker", "dt", "n_ideas_24h", "n_ideas_7d"}
    nvda = result[result["ticker"] == "NVDA"]
    assert len(nvda) == 1
    assert int(nvda.iloc[0]["n_ideas_24h"]) == 3
    assert int(nvda.iloc[0]["n_ideas_7d"]) == 3


def test_happy_path_multi_ticker():
    start = date(2026, 5, 1)
    end = date(2026, 5, 2)
    rows = [
        {"ticker": "AAPL", "idea_date": date(2026, 5, 1), "n_ideas": 2},
        {"ticker": "TSLA", "idea_date": date(2026, 5, 2), "n_ideas": 1},
    ]
    conn = _make_conn(rows)

    result = _fetch_crucix_idea_counts(conn, start, end)

    assert not result.empty
    aapl_may1 = result[(result["ticker"] == "AAPL") & (result["dt"] == date(2026, 5, 1))]
    assert int(aapl_may1.iloc[0]["n_ideas_24h"]) == 2
    tsla_may2 = result[(result["ticker"] == "TSLA") & (result["dt"] == date(2026, 5, 2))]
    assert int(tsla_may2.iloc[0]["n_ideas_24h"]) == 1


def test_7d_rolling_accumulates_across_days():
    """An idea on day 1 should contribute to n_ideas_7d on days 1-7."""
    start = date(2026, 5, 1)
    end = date(2026, 5, 7)
    rows = [{"ticker": "SPY", "idea_date": date(2026, 5, 1), "n_ideas": 2}]
    conn = _make_conn(rows)

    result = _fetch_crucix_idea_counts(conn, start, end)

    spy = result[result["ticker"] == "SPY"].set_index("dt")
    # May 1: 24h = 2, 7d = 2
    assert int(spy.loc[date(2026, 5, 1), "n_ideas_24h"]) == 2
    assert int(spy.loc[date(2026, 5, 1), "n_ideas_7d"]) == 2
    # May 7: 24h = 0 (no idea that day), 7d = 2 (within 7d window)
    assert int(spy.loc[date(2026, 5, 7), "n_ideas_24h"]) == 0
    assert int(spy.loc[date(2026, 5, 7), "n_ideas_7d"]) == 2


def test_idea_outside_7d_window_not_counted():
    """An idea on day 0 (load warmup) should NOT appear in n_ideas_7d on day 8."""
    start = date(2026, 5, 8)
    end = date(2026, 5, 8)
    # The warmup fetches from start-6d = 2026-05-02. Idea on 2026-05-01 is outside.
    rows = [{"ticker": "QQQ", "idea_date": date(2026, 5, 1), "n_ideas": 5}]
    conn = _make_conn(rows)

    result = _fetch_crucix_idea_counts(conn, start, end)
    # QQQ will not appear (no data in the load window), or appear with 0 counts.
    qqq = result[result["ticker"] == "QQQ"]
    if not qqq.empty:
        assert int(qqq.iloc[0]["n_ideas_7d"]) == 0


# ---------------------------------------------------------------------------
# dt column name regression (FIN-1584 bug: rename{"index"→"dt"} was wrong)
# ---------------------------------------------------------------------------

def test_dt_column_named_correctly():
    """Regression: _melt_to_long must rename 'idea_date' → 'dt', not 'index'→'dt'."""
    start = date(2026, 5, 1)
    end = date(2026, 5, 1)
    rows = [{"ticker": "MSFT", "idea_date": date(2026, 5, 1), "n_ideas": 1}]
    conn = _make_conn(rows)

    result = _fetch_crucix_idea_counts(conn, start, end)

    assert "dt" in result.columns, "dt column missing — likely _melt_to_long rename bug"
    assert "idea_date" not in result.columns


# ---------------------------------------------------------------------------
# Empty-table fallback
# ---------------------------------------------------------------------------

def test_empty_table_returns_empty_dataframe():
    conn = _make_conn([])
    result = _fetch_crucix_idea_counts(conn, date(2026, 5, 1), date(2026, 5, 7))
    assert result.empty
    assert list(result.columns) == ["ticker", "dt", "n_ideas_24h", "n_ideas_7d"]


# ---------------------------------------------------------------------------
# Exception path
# ---------------------------------------------------------------------------

def test_cursor_exception_returns_empty_dataframe():
    conn = MagicMock()
    conn.cursor.side_effect = Exception("DB connection lost")

    result = _fetch_crucix_idea_counts(conn, date(2026, 5, 1), date(2026, 5, 7))

    assert result.empty
    assert list(result.columns) == ["ticker", "dt", "n_ideas_24h", "n_ideas_7d"]
