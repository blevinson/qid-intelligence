#!/usr/bin/env python3
"""
Financial Simulation Service - LLM-powered market agent simulation.

Watches Crucix regime sweeps via TSDB, then has financial personas
generate market commentary via the LLM proxy. Each persona's analysis
is ingested into Graphiti, creating emergent sentiment signals in the
knowledge graph.

Lightweight alternative to full OASIS simulation -- uses direct LLM
calls to generate persona reactions rather than the heavy OASIS framework.
OASIS can be wired in later for multi-round social dynamics (Phase 4b).

Runs as a k8s Deployment alongside crucix-bridge and Neo4j.
"""

import asyncio
import importlib.util as _ilu
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from threading import Thread

import psycopg2
from openai import OpenAI

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

_fp_spec = _ilu.spec_from_file_location(
    "financial_personas",
    os.path.join(_script_dir, "app", "services", "financial_personas.py"),
)
_fp_mod = _ilu.module_from_spec(_fp_spec)
_fp_spec.loader.exec_module(_fp_mod)
FINANCIAL_PERSONAS = _fp_mod.FINANCIAL_PERSONAS
generate_regime_prompt = _fp_mod.generate_regime_prompt

# --- Config ---
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j.qid.svc.cluster.local:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
_neo4j_auth = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_PASSWORD = _neo4j_auth.split("/", 1)[1] if "/" in _neo4j_auth else _neo4j_auth
GROUP_ID = os.environ.get("GRAPHITI_GROUP_ID", "qid_intelligence")
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
POLL_INTERVAL = int(os.environ.get("SIM_POLL_INTERVAL", "600"))

# --- State ---
graphiti = None
llm_client_oai = None
last_sim_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
stats = {
    "simulations_run": 0,
    "posts_generated": 0,
    "episodes_ingested": 0,
    "errors": 0,
    "last_sim": None,
    "started_at": None,
}
running = True


def _init_graphiti():
    global graphiti
    if graphiti is not None:
        return
    llm_config = LLMConfig(
        api_key=OPENAI_API_KEY or None,
        model=LLM_MODEL,
        small_model=LLM_MODEL,
        base_url=OPENAI_BASE_URL,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=OPENAI_API_KEY or None,
        base_url=OPENAI_BASE_URL,
    )
    graphiti = Graphiti(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
    )


def _init_llm():
    global llm_client_oai
    if llm_client_oai is not None:
        return
    llm_client_oai = OpenAI(
        api_key=OPENAI_API_KEY or "sk-proxy",
        base_url=OPENAI_BASE_URL,
    )


def get_latest_sweep():
    try:
        conn = psycopg2.connect(TSDB_DSN)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT time, regime, regime_reasons, suppress, bias_direction, threshold,
                       vix, vix_change_pct, vix_regime, sp500_change_pct, sp500_range_pct,
                       wti, wti_day_change_pct, tlt_change_pct, hyg_change_pct,
                       conflict_events, conflict_fatalities
                FROM macro_sweeps
                WHERE time > %s
                ORDER BY time DESC
                LIMIT 1
            """, (last_sim_time,))
            if cur.description is None:
                conn.close()
                return None
            columns = [d[0] for d in cur.description]
            row = cur.fetchone()
        conn.close()
        return dict(zip(columns, row)) if row else None
    except Exception as e:
        print(f"[FinSim] TSDB error: {e}")
        stats["errors"] += 1
        return None


def generate_persona_post(persona: dict, regime_prompt: str) -> str:
    """Have a financial persona generate market commentary via LLM."""
    _init_llm()

    system_prompt = (
        f"You are {persona['realname']}, {persona['profession']}. "
        f"{persona['bio']} "
        f"Personality: {persona['personality']} "
        f"Posting style: {persona['posting_style']} "
        f"Your areas of focus: {', '.join(persona['interested_topics'])}.\n\n"
        "You are posting on a financial discussion platform. "
        "Write a single post (2-4 sentences) reacting to the market update below. "
        "Stay in character. Be specific about data points. No hashtags."
    )

    try:
        resp = llm_client_oai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": regime_prompt},
            ],
            max_tokens=200,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[FinSim] LLM error for {persona['username']}: {e}")
        stats["errors"] += 1
        return ""


def run_simulation(sweep: dict) -> list:
    """Generate posts from all financial personas reacting to a regime sweep."""
    regime_prompt = generate_regime_prompt(sweep)
    posts = []

    for persona in FINANCIAL_PERSONAS:
        content = generate_persona_post(persona, regime_prompt)
        if content:
            posts.append({
                "username": persona["username"],
                "realname": persona["realname"],
                "profession": persona["profession"],
                "content": content,
            })
            print(f"[FinSim]   {persona['username']}: {content[:80]}...")

    return posts


async def ingest_simulation_results(sweep: dict, posts: list):
    _init_graphiti()

    sweep_time = sweep.get("time", datetime.now(timezone.utc))
    if not hasattr(sweep_time, "tzinfo") or sweep_time.tzinfo is None:
        sweep_time = sweep_time.replace(tzinfo=timezone.utc) if hasattr(sweep_time, "replace") else datetime.now(timezone.utc)

    regime = sweep.get("regime", "unknown")
    ts_str = sweep_time.strftime("%Y%m%d_%H%M%S") if hasattr(sweep_time, "strftime") else str(int(time.time()))

    narratives = [
        f"Financial simulation triggered by {regime} regime at {ts_str}.",
        f"Market conditions: VIX={sweep.get('vix')}, WTI=${sweep.get('wti')}, "
        f"S&P500={sweep.get('sp500_change_pct', 0):+.2f}%.",
        "",
        "Market participant commentary:",
    ]
    for post in posts:
        narratives.append(
            f"- {post['username']} ({post['profession']}): {post['content']}"
        )

    await graphiti.add_episode(
        name=f"finsim_{regime}_{ts_str}",
        episode_body="\n".join(narratives),
        source=EpisodeType.text,
        source_description="Financial simulation - market participant commentary on regime conditions",
        reference_time=sweep_time,
        group_id=GROUP_ID,
    )
    stats["episodes_ingested"] += 1
    print(f"[FinSim] Ingested episode: {len(posts)} posts, regime={regime}")


async def run_one_cycle():
    global last_sim_time

    sweep = get_latest_sweep()
    if not sweep:
        return

    sweep_time = sweep.get("time", datetime.now(timezone.utc))
    ts_str = sweep_time.strftime("%Y%m%d_%H%M%S") if hasattr(sweep_time, "strftime") else "unknown"
    regime = sweep.get("regime", "unknown")
    print(f"[FinSim] New sweep: {ts_str} regime={regime}")

    posts = run_simulation(sweep)
    stats["posts_generated"] += len(posts)
    print(f"[FinSim] Generated {len(posts)} posts")

    if posts:
        await ingest_simulation_results(sweep, posts)

    stats["simulations_run"] += 1
    stats["last_sim"] = datetime.now(timezone.utc).isoformat()
    last_sim_time = sweep_time


def poll_loop():
    print(f"[FinSim] Poll loop started (interval={POLL_INTERVAL}s)")
    while running:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_one_cycle())
        except Exception as e:
            print(f"[FinSim] Cycle error: {e}")
            import traceback
            traceback.print_exc()
            stats["errors"] += 1
        finally:
            loop.close()

        for _ in range(POLL_INTERVAL):
            if not running:
                break
            time.sleep(1)


def main():
    global running
    stats["started_at"] = datetime.now(timezone.utc).isoformat()

    print("[FinSim] Financial Simulation Service starting")
    print(f"[FinSim] Neo4j: {NEO4J_URI}")
    print(f"[FinSim] LLM: {LLM_MODEL} via {OPENAI_BASE_URL or 'default'}")
    print(f"[FinSim] Poll interval: {POLL_INTERVAL}s")
    print(f"[FinSim] Personas: {len(FINANCIAL_PERSONAS)}")

    def shutdown(sig, frame):
        global running
        print("\n[FinSim] Shutting down...")
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    poll_thread = Thread(target=poll_loop, daemon=True, name="FinSimPoll")
    poll_thread.start()

    while running:
        time.sleep(1)

    print("[FinSim] Stopped.")


if __name__ == "__main__":
    main()
