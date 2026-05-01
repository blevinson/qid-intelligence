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
from graphiti_core.llm_client.config import LLMConfig

from llm import ClaudeCodeLLMClient, NoopCrossEncoder, TEIEmbedder

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
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")

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
# Poll watermark: by default skip the entire historical macro_sweeps table
# (we're event-driven via /ideas + /ingest now). Override with
# BRIDGE_POLL_FROM=YYYY-MM-DDTHH:MM:SS+00:00 to backfill from a specific
# point. Set BRIDGE_DISABLE_POLL=true to turn the poll loop off entirely.
_poll_from = os.environ.get("BRIDGE_POLL_FROM")
if _poll_from:
    last_sweep_time = datetime.fromisoformat(_poll_from)
else:
    last_sweep_time = datetime.now(timezone.utc)
DISABLE_POLL = os.environ.get("BRIDGE_DISABLE_POLL", "").lower() in ("1", "true", "yes")
stats = {"sweeps_ingested": 0, "errors": 0, "last_ingest": None, "started_at": None}
running = True

# Single dedicated event loop for all Graphiti work — neo4j's async driver
# binds to the loop that first awaits it, so we keep one loop alive in a
# worker thread and submit coroutines via run_coroutine_threadsafe.
_worker_loop: asyncio.AbstractEventLoop | None = None


def _start_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is not None:
        return _worker_loop
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    Thread(target=_run, daemon=True, name="GraphitiLoop").start()
    _worker_loop = loop
    return loop


def _submit(coro):
    """Schedule coro on the worker loop. Returns a concurrent.futures.Future."""
    loop = _start_worker_loop()
    import asyncio as _asyncio  # local alias for clarity
    return _asyncio.run_coroutine_threadsafe(coro, loop)


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


def _get_graphiti() -> Graphiti:
    global graphiti
    if graphiti is None:
        llm_config = LLMConfig(model=LLM_MODEL, small_model=LLM_MODEL)
        graphiti = Graphiti(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
            llm_client=ClaudeCodeLLMClient(config=llm_config),
            embedder=TEIEmbedder(),
            cross_encoder=NoopCrossEncoder(),
        )
    return graphiti


def idea_to_narrative(idea: dict) -> str:
    """Render a crucix trade idea as a Graphiti episode narrative."""
    parts: list[str] = []
    ticker = idea.get("ticker") or idea.get("symbol") or "UNKNOWN"
    side = idea.get("side") or idea.get("direction") or "long"
    sector = idea.get("sector")
    confidence = idea.get("confidence")
    thesis = idea.get("thesis") or idea.get("rationale") or ""

    head = f"Crucix trade idea: {side.upper()} {ticker}"
    if sector:
        head += f" ({sector})"
    if confidence is not None:
        head += f", confidence {confidence}"
    parts.append(head + ".")

    if thesis:
        parts.append(f"Thesis: {thesis}")

    catalysts = idea.get("catalysts") or []
    if catalysts:
        parts.append("Catalysts: " + "; ".join(str(c) for c in catalysts) + ".")

    risks = idea.get("risks") or []
    if risks:
        parts.append("Risks: " + "; ".join(str(r) for r in risks) + ".")

    entry = idea.get("entry")
    target = idea.get("target")
    stop = idea.get("stop")
    levels = []
    if entry is not None:
        levels.append(f"entry {entry}")
    if target is not None:
        levels.append(f"target {target}")
    if stop is not None:
        levels.append(f"stop {stop}")
    if levels:
        parts.append("Levels: " + ", ".join(levels) + ".")

    sources = idea.get("sources") or []
    if sources:
        parts.append(f"Sources: {len(sources)} cited.")

    return " ".join(parts)


async def ingest_idea(idea: dict):
    g = _get_graphiti()
    narrative = idea_to_narrative(idea)

    ts = idea.get("time") or idea.get("timestamp")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)
    if not isinstance(ts, datetime):
        ts = datetime.now(timezone.utc)

    ticker = idea.get("ticker") or idea.get("symbol") or "UNK"
    name = f"crucix_idea_{ticker}_{ts.strftime('%Y%m%d_%H%M%S')}"

    await g.add_episode(
        name=name,
        episode_body=narrative,
        source=EpisodeType.text,
        source_description="Crucix trade idea",
        reference_time=ts,
        group_id=GROUP_ID,
    )
    stats["ideas_ingested"] = stats.get("ideas_ingested", 0) + 1
    stats["last_idea"] = datetime.now(timezone.utc).isoformat()
    print(f"[Bridge] Ingested idea {name}")


async def ingest_sweep(sweep: dict):
    global last_sweep_time
    g = _get_graphiti()
    narrative = sweep_to_narrative(sweep)
    sweep_time = sweep.get("time", datetime.now(timezone.utc))
    if not hasattr(sweep_time, "tzinfo"):
        sweep_time = datetime.now(timezone.utc)

    ts_str = sweep_time.strftime("%Y%m%d_%H%M%S") if hasattr(sweep_time, "strftime") else str(int(time.time()))

    await g.add_episode(
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
            for sweep in sweeps:
                try:
                    _submit(ingest_sweep(sweep)).result(timeout=600)
                except Exception as e:
                    print(f"[Bridge] Ingest error: {e}")
                    stats["errors"] += 1

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

    def _ingest_endpoint(self, handler):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
            _submit(handler(payload)).result(timeout=600)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
        except Exception as e:
            print(f"[Bridge] {self.path} ingest error: {e}")
            stats["errors"] += 1
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        if self.path == "/ingest":
            self._ingest_endpoint(ingest_sweep)
        elif self.path == "/ideas":
            # Async path: parse body, ack 202 immediately, ingest in background.
            # Each idea triggers a Haiku entity-extraction pass which can take
            # 30-60s, so we never want the HTTP caller blocking that long.
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"bad json: {e}"}).encode())
                return

            ideas = payload.get("ideas") if isinstance(payload, dict) and "ideas" in payload else (
                payload if isinstance(payload, list) else [payload]
            )

            def _worker(items):
                for idea in items:
                    try:
                        _submit(ingest_idea(idea)).result(timeout=600)
                    except Exception as e:
                        print(f"[Bridge] /ideas worker error on {idea.get('ticker','?')}: {e}")
                        stats["errors"] += 1

            Thread(target=_worker, args=(ideas,), daemon=True, name="IdeasWorker").start()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"accepted": True, "count": len(ideas)}).encode())
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

    # Spin up the dedicated Graphiti event loop thread before any worker
    # tries to submit work to it.
    _start_worker_loop()

    # Start poll thread (skipped when BRIDGE_DISABLE_POLL=true — event-
    # driven /ideas + /ingest path is the default now)
    if DISABLE_POLL:
        print("[Bridge] Poll loop disabled (BRIDGE_DISABLE_POLL=true)")
    else:
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
