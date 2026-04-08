#!/usr/bin/env python3
"""
Test script: Ingest a sample Crucix sweep into Graphiti and search it back.
Run: python -m scripts.test_graphiti_ingestion
Or:  python scripts/test_graphiti_ingestion.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Neo4j connection (adjust for port-forward or cluster access)
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
GROUP_ID = "qid_intelligence"


async def main():
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 1. Ingest a sample Crucix sweep as narrative
        print("\n--- Ingesting sample Crucix sweep ---")
        narrative = (
            "Crucix macro intelligence sweep at 2026-04-08T03:39:22Z. "
            "Market regime classification: elevated. "
            "Regime reasons: VIX 25.8 (25-30 danger zone). "
            "VIX is at 25.8 (+5.1% change). "
            "S&P 500 change: +4.31%. "
            "WTI crude oil at $95.67 (-15.3% intraday). "
            "TLT (bonds) change: -0.16%. "
            "HYG (high-yield credit) change: +1.15%. "
            "ML score threshold at 0.60."
        )

        result = await graphiti.add_episode(
            name="crucix_sweep_20260408_033922",
            episode_body=narrative,
            source=EpisodeType.text,
            source_description="Crucix macro intelligence sweep - narrative",
            reference_time=datetime(2026, 4, 8, 3, 39, 22, tzinfo=timezone.utc),
            group_id=GROUP_ID,
        )
        print(f"Episode ingested: {result}")

        # 2. Search for related information
        print("\n--- Searching: 'VIX elevated regime' ---")
        edges = await graphiti.search(
            query="VIX elevated regime",
            num_results=5,
            group_ids=[GROUP_ID],
        )
        print(f"Found {len(edges)} results:")
        for edge in edges:
            print(f"  - {edge.fact}")

        print("\n--- Searching: 'WTI crude oil price' ---")
        edges2 = await graphiti.search(
            query="WTI crude oil price",
            num_results=5,
            group_ids=[GROUP_ID],
        )
        print(f"Found {len(edges2)} results:")
        for edge in edges2:
            print(f"  - {edge.fact}")

        print("\n--- Searching: 'market regime energy shock' ---")
        edges3 = await graphiti.search(
            query="market regime energy shock",
            num_results=5,
            group_ids=[GROUP_ID],
        )
        print(f"Found {len(edges3)} results:")
        for edge in edges3:
            print(f"  - {edge.fact}")

        print("\nTest complete!")

    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
