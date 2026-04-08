"""
Crucix Intelligence Ingestion Service.
Polls macro_sweeps from TSDB and ingests into Graphiti knowledge graph.
Each sweep becomes a structured episode that Graphiti processes into
entities (VIX, WTI, S&P500, etc.) and relationships (regime transitions).
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import psycopg2
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('qid.crucix_ingestion')

SWEEP_QUERY = """
SELECT time, regime, regime_reasons, suppress, bias_direction, threshold,
       vix, vix_change_pct, vix_regime, sp500_change_pct, sp500_range_pct,
       wti, wti_day_change_pct, tlt_change_pct, hyg_change_pct,
       conflict_events, conflict_fatalities,
       sources_ok, sources_total, llm_ideas, news_count
FROM macro_sweeps
WHERE time > %s
ORDER BY time ASC
"""


def sweep_to_narrative(sweep: Dict[str, Any]) -> str:
    """Convert a Crucix macro sweep into a natural language narrative
    that Graphiti can extract entities and relationships from."""
    ts = sweep["time"]
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()

    parts = [f"Crucix macro intelligence sweep at {ts}."]

    regime = sweep.get("regime", "unknown")
    reasons = sweep.get("regime_reasons") or []
    parts.append(f"Market regime classification: {regime}.")
    if reasons:
        parts.append(f"Regime reasons: {'; '.join(reasons)}.")

    vix = sweep.get("vix")
    vix_chg = sweep.get("vix_change_pct")
    if vix is not None:
        vix_str = f"VIX is at {vix:.1f}"
        if vix_chg is not None:
            vix_str += f" ({vix_chg:+.1f}% change)"
        parts.append(vix_str + ".")

    sp500 = sweep.get("sp500_change_pct")
    if sp500 is not None:
        parts.append(f"S&P 500 change: {sp500:+.2f}%.")

    wti = sweep.get("wti")
    wti_chg = sweep.get("wti_day_change_pct")
    if wti is not None:
        wti_str = f"WTI crude oil at ${wti:.2f}"
        if wti_chg is not None:
            wti_str += f" ({wti_chg:+.1f}% intraday)"
        parts.append(wti_str + ".")

    tlt = sweep.get("tlt_change_pct")
    if tlt is not None:
        parts.append(f"TLT (bonds) change: {tlt:+.2f}%.")

    hyg = sweep.get("hyg_change_pct")
    if hyg is not None:
        parts.append(f"HYG (high-yield credit) change: {hyg:+.2f}%.")

    suppress = sweep.get("suppress", False)
    threshold = sweep.get("threshold", 0.6)
    bias = sweep.get("bias_direction")
    if suppress:
        parts.append("Trading SUPPRESSED due to extreme conditions.")
    if bias:
        parts.append(f"Directional bias: {bias}.")
    if threshold != 0.6:
        parts.append(f"ML score threshold adjusted to {threshold}.")

    conflict_events = sweep.get("conflict_events", 0)
    conflict_fatalities = sweep.get("conflict_fatalities", 0)
    if conflict_events > 0:
        parts.append(f"Active conflict: {conflict_events} events, {conflict_fatalities} fatalities.")

    return " ".join(parts)


def sweep_to_json_episode(sweep: Dict[str, Any]) -> str:
    """Convert sweep to structured JSON for Graphiti JSON episode type."""
    data = {
        "event_type": "crucix_macro_sweep",
        "timestamp": sweep["time"].isoformat() if hasattr(sweep["time"], "isoformat") else str(sweep["time"]),
        "regime": sweep.get("regime"),
        "regime_reasons": sweep.get("regime_reasons") or [],
        "indicators": {
            "vix": sweep.get("vix"),
            "vix_change_pct": sweep.get("vix_change_pct"),
            "sp500_change_pct": sweep.get("sp500_change_pct"),
            "wti": sweep.get("wti"),
            "wti_day_change_pct": sweep.get("wti_day_change_pct"),
            "tlt_change_pct": sweep.get("tlt_change_pct"),
            "hyg_change_pct": sweep.get("hyg_change_pct"),
        },
        "trading_params": {
            "suppress": sweep.get("suppress", False),
            "bias_direction": sweep.get("bias_direction"),
            "threshold": sweep.get("threshold", 0.6),
        },
        "conflict": {
            "events": sweep.get("conflict_events", 0),
            "fatalities": sweep.get("conflict_fatalities", 0),
        },
    }
    return json.dumps(data)


class CrucixIngestionService:
    """Polls Crucix macro_sweeps from TSDB and ingests into Graphiti."""

    def __init__(
        self,
        tsdb_dsn: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        group_id: Optional[str] = None,
        poll_interval_seconds: int = 300,
    ):
        self._tsdb_dsn = tsdb_dsn or os.environ.get(
            "QID_TSDB_DSN",
            f"host={os.environ.get('QID_DB_HOST', 'localhost')} "
            f"port={os.environ.get('QID_DB_PORT', '5432')} "
            f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
            f"user={os.environ.get('QID_DB_USER', 'qid')} "
            f"password={os.environ.get('QID_DB_PASS', 'qid')}",
        )
        self._neo4j_uri = neo4j_uri or Config.NEO4J_URI
        self._neo4j_user = neo4j_user or Config.NEO4J_USER
        self._neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        self._group_id = group_id or Config.GRAPHITI_GROUP_ID

        self._poll_interval = poll_interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_sweep_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self._graphiti: Optional[Graphiti] = None

        # Stats
        self._sweeps_ingested = 0
        self._errors = 0

    def _get_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            self._graphiti = Graphiti(
                self._neo4j_uri,
                self._neo4j_user,
                self._neo4j_password,
            )
        return self._graphiti

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="CrucixIngestion"
        )
        self._thread.start()
        logger.info(f"CrucixIngestion started (poll every {self._poll_interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._graphiti:
            asyncio.run(self._graphiti.close())
            self._graphiti = None
        logger.info(
            f"CrucixIngestion stopped: {self._sweeps_ingested} ingested, {self._errors} errors"
        )

    def _poll_loop(self):
        while self._running:
            try:
                self._poll_and_ingest()
            except Exception as e:
                logger.error(f"Poll cycle failed: {e}")
                self._errors += 1
            for _ in range(self._poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _poll_and_ingest(self):
        conn = psycopg2.connect(self._tsdb_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(SWEEP_QUERY, (self._last_sweep_time,))
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return

        logger.info(f"Found {len(rows)} new sweeps to ingest")
        graphiti = self._get_graphiti()
        loop = asyncio.new_event_loop()

        try:
            for row in rows:
                sweep = dict(zip(columns, row))
                sweep_time = sweep["time"]

                # Ingest as narrative text episode
                narrative = sweep_to_narrative(sweep)

                # Also ingest structured JSON
                json_body = sweep_to_json_episode(sweep)

                try:
                    loop.run_until_complete(
                        graphiti.add_episode(
                            name=f"crucix_sweep_{sweep_time.strftime('%Y%m%d_%H%M%S') if hasattr(sweep_time, 'strftime') else str(sweep_time)}",
                            episode_body=narrative,
                            source=EpisodeType.text,
                            source_description="Crucix macro intelligence sweep - narrative",
                            reference_time=sweep_time if hasattr(sweep_time, "tzinfo") else datetime.now(timezone.utc),
                            group_id=self._group_id,
                        )
                    )
                    self._sweeps_ingested += 1
                    self._last_sweep_time = sweep_time
                    logger.info(
                        f"Ingested sweep {sweep_time}: regime={sweep.get('regime')}, "
                        f"VIX={sweep.get('vix')}, WTI={sweep.get('wti')}"
                    )
                except Exception as e:
                    logger.error(f"Failed to ingest sweep {sweep_time}: {e}")
                    self._errors += 1
        finally:
            loop.close()

    def ingest_one(self, sweep: Dict[str, Any]):
        """Manually ingest a single sweep dict (for testing)."""
        graphiti = self._get_graphiti()
        narrative = sweep_to_narrative(sweep)
        sweep_time = sweep.get("time", datetime.now(timezone.utc))

        asyncio.run(
            graphiti.add_episode(
                name=f"crucix_sweep_{int(time.time())}",
                episode_body=narrative,
                source=EpisodeType.text,
                source_description="Crucix macro intelligence sweep",
                reference_time=sweep_time if hasattr(sweep_time, "tzinfo") else datetime.now(timezone.utc),
                group_id=self._group_id,
            )
        )
        self._sweeps_ingested += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "sweeps_ingested": self._sweeps_ingested,
            "errors": self._errors,
            "last_sweep_time": str(self._last_sweep_time),
            "poll_interval_seconds": self._poll_interval,
        }
