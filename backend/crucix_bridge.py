#!/usr/bin/env python3
"""
Crucix Intelligence Bridge - Standalone service.
Runs as a k8s Deployment alongside Crucix + Neo4j.

Two modes:
1. Poll: Periodically queries macro_sweeps from TSDB for new rows
2. Webhook: Accepts POST /ingest from Crucix server after each sweep

Also exposes /health and /stats endpoints.
"""

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import psycopg2
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

# --- Config from env ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j.qid.svc.cluster.local:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
# NEO4J_AUTH secret is in format "neo4j/password" -- extract just the password part
_neo4j_auth = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_PASSWORD = _neo4j_auth.split("/", 1)[1] if "/" in _neo4j_auth else _neo4j_auth
GROUP_ID = os.environ.get("GRAPHITI_GROUP_ID", "qid_intelligence")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8090"))
TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'timescaledb.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', 'qid')}"
))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

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

# --- State ---
graphiti = None
last_sweep_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
stats = {"sweeps_ingested": 0, "errors": 0, "last_ingest": None, "started_at": None}
running = True


def sweep_to_narrative(sweep: dict) -> str:
    ts = sweep.get("time", "")
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()

    parts = [f"Crucix macro intelligence sweep at {ts}."]

    regime = sweep.get("regime", "unknown")
    reasons = sweep.get("regime_reasons") or []
    parts.append(f"Market regime classification: {regime}.")
    if reasons:
        parts.append(f"Regime reasons: {'; '.join(str(r) for r in reasons)}.")

    vix = sweep.get("vix")
    if vix is not None:
        vix_str = f"VIX is at {vix:.1f}"
        vix_chg = sweep.get("vix_change_pct")
        if vix_chg is not None:
            vix_str += f" ({vix_chg:+.1f}% change)"
        parts.append(vix_str + ".")

    sp500 = sweep.get("sp500_change_pct")
    if sp500 is not None:
        parts.append(f"S&P 500 change: {sp500:+.2f}%.")

    wti = sweep.get("wti")
    if wti is not None:
        wti_str = f"WTI crude oil at ${wti:.2f}"
        wti_chg = sweep.get("wti_day_change_pct")
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

    conflict = sweep.get("conflict_events", 0)
    fatalities = sweep.get("conflict_fatalities", 0)
    if conflict > 0:
        parts.append(f"Active conflict: {conflict} events, {fatalities} fatalities.")

    return " ".join(parts)


async def ingest_sweep(sweep: dict):
    global graphiti, last_sweep_time
    if graphiti is None:
        llm_config = LLMConfig(
            api_key=OPENAI_API_KEY or None,
            model=LLM_MODEL,
            small_model=LLM_MODEL,
            base_url=OPENAI_BASE_URL,
        )
        llm_client = OpenAIClient(config=llm_config)
        embedder_config = OpenAIEmbedderConfig(
            api_key=OPENAI_API_KEY or None,
            base_url=OPENAI_BASE_URL,
        )
        embedder = OpenAIEmbedder(config=embedder_config)
        graphiti = Graphiti(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder,
        )

    narrative = sweep_to_narrative(sweep)
    sweep_time = sweep.get("time", datetime.now(timezone.utc))
    if not hasattr(sweep_time, "tzinfo"):
        sweep_time = datetime.now(timezone.utc)

    ts_str = sweep_time.strftime("%Y%m%d_%H%M%S") if hasattr(sweep_time, "strftime") else str(int(time.time()))

    await graphiti.add_episode(
        name=f"crucix_sweep_{ts_str}",
        episode_body=narrative,
        source=EpisodeType.text,
        source_description="Crucix macro intelligence sweep",
        reference_time=sweep_time,
        group_id=GROUP_ID,
    )
    stats["sweeps_ingested"] += 1
    stats["last_ingest"] = datetime.now(timezone.utc).isoformat()
    last_sweep_time = sweep_time
    print(f"[Bridge] Ingested sweep {ts_str}: regime={sweep.get('regime')}, VIX={sweep.get('vix')}, WTI={sweep.get('wti')}")


def poll_tsdb():
    global last_sweep_time
    try:
        conn = psycopg2.connect(TSDB_DSN)
        with conn.cursor() as cur:
            cur.execute(SWEEP_QUERY, (last_sweep_time,))
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"[Bridge] TSDB poll failed: {e}")
        stats["errors"] += 1
        return []

    return [dict(zip(columns, row)) for row in rows]


def poll_loop():
    print(f"[Bridge] Poll loop started (interval={POLL_INTERVAL}s)")
    while running:
        sweeps = poll_tsdb()
        if sweeps:
            print(f"[Bridge] Found {len(sweeps)} new sweeps")
            loop = asyncio.new_event_loop()
            for sweep in sweeps:
                try:
                    loop.run_until_complete(ingest_sweep(sweep))
                except Exception as e:
                    print(f"[Bridge] Ingest error: {e}")
                    stats["errors"] += 1
            loop.close()

        for _ in range(POLL_INTERVAL):
            if not running:
                break
            time.sleep(1)


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress access logs

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "neo4j": NEO4J_URI}).encode())
        elif self.path == "/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/ingest":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                sweep = json.loads(body)
                loop = asyncio.new_event_loop()
                loop.run_until_complete(ingest_sweep(sweep))
                loop.close()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())
            except Exception as e:
                print(f"[Bridge] Webhook ingest error: {e}")
                stats["errors"] += 1
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global running
    stats["started_at"] = datetime.now(timezone.utc).isoformat()

    print(f"[Bridge] Crucix Intelligence Bridge starting")
    print(f"[Bridge] Neo4j: {NEO4J_URI}")
    print(f"[Bridge] TSDB: {TSDB_DSN.split('password=')[0]}...")
    print(f"[Bridge] Group: {GROUP_ID}")
    print(f"[Bridge] Poll interval: {POLL_INTERVAL}s")
    print(f"[Bridge] HTTP port: {HTTP_PORT}")

    def shutdown(sig, frame):
        global running
        print(f"\n[Bridge] Shutting down...")
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Start poll thread
    poll_thread = Thread(target=poll_loop, daemon=True, name="CrucixPoll")
    poll_thread.start()

    # Start HTTP server (webhook + health)
    server = HTTPServer(("0.0.0.0", HTTP_PORT), BridgeHandler)
    server.timeout = 1

    print(f"[Bridge] Ready. Listening on :{HTTP_PORT}")

    while running:
        server.handle_request()

    print("[Bridge] Stopped.")


if __name__ == "__main__":
    main()
