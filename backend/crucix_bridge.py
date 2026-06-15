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
import re
import signal
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Thread

import psycopg2
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig

# Claude (Anthropic-protocol) LLM + in-cluster TEI embedder + no-op reranker.
# Replaces Graphiti's default OpenAI clients so the bridge authenticates via
# the claude CLI subprocess (ANTHROPIC_* env in-pod) instead of an OpenAI key.
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
# LLM model for Graphiti entity extraction. Routed through the claude CLI
# (ClaudeCodeLLMClient), which honors ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
# in-pod. Haiku is the right default for high-frequency extraction.
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
LLM_SMALL_MODEL = os.environ.get("LLM_SMALL_MODEL", LLM_MODEL)
# Max seconds to wait for a single Graphiti add_episode call. GraphitiEmit can
# block on Neo4j or OpenAI; without a timeout the HTTP server thread stalls and
# the liveness probe kills the container (FIN-2008).
GRAPHITI_TIMEOUT = int(os.environ.get("GRAPHITI_TIMEOUT_SECONDS", "240"))

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
# One persistent event loop owns ALL graphiti work. The neo4j async driver AND
# the ClaudeCodeLLMClient concurrency semaphore bind to the loop that creates
# them, so every coroutine must run on the SAME loop. The threaded HTTP server
# and the poll thread dispatch onto it via run_coroutine_threadsafe.
BRIDGE_LOOP = None      # asyncio loop running forever in a background thread (set in main)
_GRAPHITI = None        # singleton Graphiti client, built once on BRIDGE_LOOP
last_sweep_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
stats = {
    "sweeps_ingested": 0,
    "ideas_ingested": 0,
    "errors": 0,
    "last_ingest": None,
    "started_at": None,
}
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


def _build_graphiti() -> Graphiti:
    """Construct the Graphiti client (Claude LLM + TEI embedder + no-op reranker).

    Called ONCE, inside a coroutine on BRIDGE_LOOP, so the neo4j async driver and
    the LLM-client concurrency semaphore bind to that single long-lived loop. A
    per-request loop (asyncio.run) breaks because those primitives are loop-bound
    ('Future/Semaphore attached to a different event loop').

    Indices/constraints are NOT built here — they already exist in the qid Neo4j
    graph, and build_indices_and_constraints on every call adds latency (FIN-2008).
    """
    llm_config = LLMConfig(
        model=LLM_MODEL or "claude-haiku-4-5",
        small_model=LLM_SMALL_MODEL or LLM_MODEL or "claude-haiku-4-5",
    )
    return Graphiti(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
        llm_client=ClaudeCodeLLMClient(config=llm_config),
        embedder=TEIEmbedder(),
        cross_encoder=NoopCrossEncoder(),
    )


async def _make_graphiti() -> Graphiti:
    """Build the client while a loop is running so loop-bound primitives bind to it."""
    return _build_graphiti()


def _run_on_loop(coro, timeout: float):
    """Submit a coroutine to the shared BRIDGE_LOOP from any thread and block for it.
    All graphiti calls funnel through here so they share one event loop."""
    fut = asyncio.run_coroutine_threadsafe(coro, BRIDGE_LOOP)
    return fut.result(timeout=timeout)


def _parse_iso(value) -> datetime:
    """Parse an ISO-8601 timestamp (crucix sends idea.time as an ISO string),
    returning a tz-aware UTC datetime. Falls back to now(UTC) on any failure."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        s = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


async def ingest_sweep(sweep: dict, g: Graphiti):
    global last_sweep_time

    narrative = sweep_to_narrative(sweep)
    sweep_time = sweep.get("time", datetime.now(timezone.utc))
    if not hasattr(sweep_time, "tzinfo"):
        sweep_time = datetime.now(timezone.utc)

    ts_str = sweep_time.strftime("%Y%m%d_%H%M%S") if hasattr(sweep_time, "strftime") else str(int(time.time()))

    await asyncio.wait_for(
        g.add_episode(
            name=f"crucix_sweep_{ts_str}",
            episode_body=narrative,
            source=EpisodeType.text,
            source_description="Crucix macro intelligence sweep",
            reference_time=sweep_time,
            group_id=GROUP_ID,
        ),
        timeout=GRAPHITI_TIMEOUT,
    )
    stats["sweeps_ingested"] += 1
    stats["last_ingest"] = datetime.now(timezone.utc).isoformat()
    last_sweep_time = sweep_time
    print(f"[Bridge] Ingested sweep {ts_str}: regime={sweep.get('regime')}, VIX={sweep.get('vix')}, WTI={sweep.get('wti')}")


async def ingest_idea(idea: dict, g: Graphiti):
    """Ingest a single crucix trade idea as a JSON episode.

    Crucix POSTs ideas (see crucix/lib/graphiti_emit.mjs) shaped like:
        {time, title, type, ticker, confidence, rationale, risk, horizon,
         signals, shares_per_1k, source}
    where `type` is one of LONG|SHORT|HEDGE|WATCH|AVOID.

    The episode body is JSON consumed downstream by
    ml-order-flow-detection/scripts/crucix_idea_outcome.py, which requires
    keys: ticker, direction, thesis, horizon (plus confidence, title). It
    skips episodes whose `type` == "IDEA_OUTCOME", so we set type = direction
    (LONG/SHORT/etc.), never "IDEA_OUTCOME".

    The episode NAME must match the archive regex
        ^crucix_idea_([A-Z0-9]+)_(\\d{8})_(\\d{6})$
    so the ticker is uppercased and stripped of spaces.
    """
    ref_time = _parse_iso(idea.get("time"))

    # Strip ALL non-alphanumerics (not just spaces) so dot/punct tickers
    # (BRK.B, BF.B, RDS-A) still match the archive name regex ^[A-Z0-9]+$.
    raw_ticker = re.sub(r"[^A-Z0-9]", "", str(idea.get("ticker") or "").upper())
    ticker = raw_ticker or "UNKNOWN"
    direction = str(idea.get("type") or "WATCH").upper()
    ts_str = ref_time.strftime("%Y%m%d_%H%M%S")
    name = f"crucix_idea_{ticker}_{ts_str}"

    body = json.dumps({
        "type":          direction,          # LONG|SHORT|HEDGE|WATCH|AVOID (never IDEA_OUTCOME)
        "ticker":        ticker,
        "direction":     direction,          # downstream reads `direction`
        "confidence":    idea.get("confidence", "MEDIUM"),
        "thesis":        idea.get("rationale", ""),   # downstream reads `thesis`
        "risk":          idea.get("risk", ""),
        "horizon":       idea.get("horizon", ""),
        "title":         idea.get("title", ""),
        "shares_per_1k": idea.get("shares_per_1k"),
        "signals":       idea.get("signals", []),
    })

    await asyncio.wait_for(
        g.add_episode(
            name=name,
            episode_body=body,
            source=EpisodeType.json,
            source_description="Crucix trade idea",
            reference_time=ref_time,
            group_id=GROUP_ID,
        ),
        timeout=GRAPHITI_TIMEOUT,
    )
    stats["ideas_ingested"] += 1
    stats["last_ingest"] = datetime.now(timezone.utc).isoformat()
    print(f"[Bridge] Ingested idea {name}: {direction} {ticker} conf={idea.get('confidence')}")


_idea_gate: "asyncio.Semaphore | None" = None


def _get_idea_gate() -> "asyncio.Semaphore":
    # Serialize idea ingestion to ONE add_episode at a time. The claude CLI runs
    # at CLAUDE_CODE_CONCURRENCY=1 (one subprocess globally), so firing N ideas
    # concurrently time-shares that single permit and balloons each idea's
    # add_episode wall-time ~N×, blowing its GRAPHITI_TIMEOUT. Serialized, each
    # idea owns the permit for its ~80s and finishes well under the timeout.
    global _idea_gate
    if _idea_gate is None:
        _idea_gate = asyncio.Semaphore(1)
    return _idea_gate


async def _ingest_idea_safe(idea: dict):
    """Background-friendly wrapper: /ideas fires these onto BRIDGE_LOOP without
    blocking the HTTP response (graphiti add_episode is many LLM calls = slow),
    so failures are logged + counted here rather than surfaced to the caller.
    The gate processes ideas one-at-a-time (see _get_idea_gate)."""
    async with _get_idea_gate():
        try:
            await ingest_idea(idea, _GRAPHITI)
        except Exception as e:
            print(f"[Bridge] Idea ingest error: {e!r}")
            stats["errors"] += 1


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
                    _run_on_loop(ingest_sweep(sweep, _GRAPHITI), GRAPHITI_TIMEOUT + 30)
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

    def do_POST(self):
        if self.path == "/ingest":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                sweep = json.loads(body)
                _run_on_loop(ingest_sweep(sweep, _GRAPHITI), GRAPHITI_TIMEOUT + 30)
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
        elif self.path == "/ideas":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                # Accept {ideas:[...]} | bare list | single object.
                if isinstance(payload, dict) and "ideas" in payload:
                    ideas = payload["ideas"]
                elif isinstance(payload, list):
                    ideas = payload
                else:
                    ideas = [payload]
                if not isinstance(ideas, list):
                    ideas = [ideas]

                # Fire each idea onto the persistent loop and return immediately.
                # add_episode is many sequential LLM calls (slow); blocking the
                # node's POST for a whole batch would time it out. Errors are
                # logged + counted in _ingest_idea_safe; progress shows in /stats.
                for idea in ideas:
                    asyncio.run_coroutine_threadsafe(_ingest_idea_safe(idea), BRIDGE_LOOP)

                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "accepted": len(ideas)}).encode())
            except Exception as e:
                print(f"[Bridge] Ideas ingest error: {e}")
                stats["errors"] += 1
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global running, BRIDGE_LOOP, _GRAPHITI
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

    # Start the single persistent asyncio loop and build the Graphiti client ON
    # it, so the neo4j driver + LLM-client semaphore bind to this one loop. All
    # ingest work (poll thread + HTTP handlers) dispatches here via _run_on_loop.
    BRIDGE_LOOP = asyncio.new_event_loop()
    Thread(target=BRIDGE_LOOP.run_forever, daemon=True, name="BridgeLoop").start()
    _GRAPHITI = asyncio.run_coroutine_threadsafe(_make_graphiti(), BRIDGE_LOOP).result(timeout=60)
    print("[Bridge] Graphiti client ready on persistent loop")

    # Start poll thread — unless disabled. The TSDB->graphiti sweep mirror is a
    # legacy path that re-ingests the entire sweep history on startup (last-seen
    # defaults to 2020 => thousands of sweeps) and monopolizes BRIDGE_LOOP, which
    # would starve idea ingestion. BRIDGE_DISABLE_POLL=true (set on the deploy)
    # turns it off; P0 only needs the /ideas + /ingest webhook paths.
    if os.environ.get("BRIDGE_DISABLE_POLL", "").strip().lower() in ("1", "true", "yes"):
        print("[Bridge] Poll loop DISABLED (BRIDGE_DISABLE_POLL)")
    else:
        poll_thread = Thread(target=poll_loop, daemon=True, name="CrucixPoll")
        poll_thread.start()

    # Use a threading server so POST /ingest (slow Graphiti call) never blocks
    # GET /health. Each request gets its own thread (FIN-2008).
    class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = _ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), BridgeHandler)
    server.timeout = 1

    print(f"[Bridge] Ready. Listening on :{HTTP_PORT}")

    while running:
        server.handle_request()

    print("[Bridge] Stopped.")


if __name__ == "__main__":
    main()
