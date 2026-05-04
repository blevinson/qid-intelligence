#!/usr/bin/env python3
"""
SEC EDGAR 8-K filings ingest into crucix_sec_8k_filings.
Polls the EDGAR full-text search / current-filings Atom feed for 8-K forms,
parses item codes, joins CIK→ticker, and upserts into TimescaleDB hypertable.
Designed to run every 15 min as a k8s CronJob.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

EDGAR_FEED_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%228-K%22&dateRange=custom&startdt={start}&enddt={end}"
    "&forms=8-K&from={offset}&size={size}"
)
EDGAR_CURRENT_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?forms=8-K&from={offset}&size={size}"
)
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index?forms=8-K&from={offset}&size={size}"

EDGAR_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include&count={count}&search_text=&start={start}&output=atom"
)

USER_AGENT = "FinTorch crucix-ingest brantjlevinson@gmail.com"

HIGH_IMPACT_ITEMS = {
    "1.01",  # Material definitive agreement
    "2.01",  # Completion of acquisition
    "2.02",  # Results of operations (earnings)
    "4.02",  # Non-reliance / restatement
    "5.02",  # Director/officer departure
    "7.01",  # Reg FD disclosure (guidance/preannouncements)
    "8.01",  # Other material events
}

ITEM_PATTERN = re.compile(r"(?:Item\s+)?(\d+\.\d{2})", re.IGNORECASE)

TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'timescaledb.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', 'qid')}"
))

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crucix_sec_8k_filings (
  accession_no TEXT NOT NULL,
  cik          BIGINT NOT NULL,
  ticker       TEXT,
  filed_at     TIMESTAMPTZ NOT NULL,
  items        TEXT[] NOT NULL,
  has_material BOOLEAN NOT NULL,
  filing_url   TEXT NOT NULL,
  document_url TEXT,
  fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (accession_no, filed_at)
);

SELECT create_hypertable('crucix_sec_8k_filings', 'filed_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_8k_ticker_filed
  ON crucix_sec_8k_filings(ticker, filed_at DESC);

CREATE INDEX IF NOT EXISTS idx_8k_material
  ON crucix_sec_8k_filings(filed_at DESC) WHERE has_material;
"""


def get_watermark(conn):
    """Get the latest filed_at from the DB, or 7 days ago if empty."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(filed_at) FROM crucix_sec_8k_filings")
        row = cur.fetchone()
        if row and row[0]:
            return row[0] - timedelta(hours=1)
    return datetime.now(timezone.utc) - timedelta(days=7)


def load_cik_map(conn):
    """Load CIK→ticker mapping from sec_cik_ticker."""
    cik_map = {}
    with conn.cursor() as cur:
        cur.execute("SELECT cik, ticker FROM sec_cik_ticker")
        for cik, ticker in cur.fetchall():
            cik_map[cik] = ticker
    return cik_map


def parse_accession(raw):
    """Normalize accession number: strip dashes, return canonical form."""
    return raw.replace("-", "").strip()


def fetch_atom_page(start=0, count=40):
    """Fetch one page of the EDGAR current-filings Atom feed for 8-K."""
    url = EDGAR_ATOM_URL.format(start=start, count=count)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_atom_entries(xml_text):
    """Parse Atom XML into a list of filing dicts."""
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        title = title_el.text if title_el is not None else ""

        if "8-K" not in title:
            continue

        link_el = entry.find("atom:link", ATOM_NS)
        filing_url = link_el.attrib.get("href", "") if link_el is not None else ""

        updated_el = entry.find("atom:updated", ATOM_NS)
        updated_text = updated_el.text if updated_el is not None else ""

        summary_el = entry.find("atom:summary", ATOM_NS)
        summary = summary_el.text if summary_el is not None else ""

        accession_match = re.search(r"(\d{10}-\d{2}-\d{6})", filing_url)
        if not accession_match:
            continue
        accession_no = accession_match.group(1)

        cik_match = re.search(r"/cgi-bin/browse-edgar\?action=getcompany&CIK=(\d+)", summary + filing_url)
        if not cik_match:
            cik_match = re.search(r"CIK=(\d+)", filing_url)
        if not cik_match:
            cik_match = re.search(r"/data/(\d+)/", filing_url)
        cik = int(cik_match.group(1)) if cik_match else 0

        items = ITEM_PATTERN.findall(title + " " + summary)
        items = sorted(set(items))

        try:
            filed_at = datetime.fromisoformat(updated_text.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            filed_at = datetime.now(timezone.utc)

        doc_match = re.search(r'href="([^"]*\.htm)"', summary)
        document_url = doc_match.group(1) if doc_match else None
        if document_url and not document_url.startswith("http"):
            document_url = "https://www.sec.gov" + document_url

        has_material = bool(set(items) & HIGH_IMPACT_ITEMS)

        entries.append({
            "accession_no": accession_no,
            "cik": cik,
            "filed_at": filed_at,
            "items": items,
            "has_material": has_material,
            "filing_url": filing_url if filing_url.startswith("http") else f"https://www.sec.gov{filing_url}",
            "document_url": document_url,
        })
    return entries


def fetch_filings(watermark):
    """Walk Atom feed pages until we see filings older than watermark."""
    all_entries = []
    start = 0
    page_size = 40

    while True:
        print(f"[sec_8k_ingest] Fetching page start={start}")
        xml_text = fetch_atom_page(start=start, count=page_size)
        entries = parse_atom_entries(xml_text)

        if not entries:
            break

        all_entries.extend(entries)

        oldest = min(e["filed_at"] for e in entries)
        if oldest < watermark:
            break

        start += page_size

        if start > 400:
            print("[sec_8k_ingest] Safety cap at 400 entries, stopping pagination")
            break

    return all_entries


def upsert_filings(conn, filings, cik_map):
    """Upsert filings into crucix_sec_8k_filings."""
    if not filings:
        print("[sec_8k_ingest] No filings to upsert")
        return 0

    for f in filings:
        f["ticker"] = cik_map.get(f["cik"])

    # Deduplicate by (accession_no, filed_at) — EDGAR can return the same filing multiple times
    seen = {}
    for f in filings:
        seen[(f["accession_no"], f["filed_at"])] = f
    filings = list(seen.values())

    with conn.cursor() as cur:
        values = [
            (
                f["accession_no"],
                f["cik"],
                f["ticker"],
                f["filed_at"],
                f["items"],
                f["has_material"],
                f["filing_url"],
                f["document_url"],
            )
            for f in filings
        ]
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO crucix_sec_8k_filings
               (accession_no, cik, ticker, filed_at, items, has_material, filing_url, document_url, fetched_at)
               VALUES %s
               ON CONFLICT (accession_no, filed_at) DO UPDATE SET
                 ticker       = EXCLUDED.ticker,
                 items        = EXCLUDED.items,
                 has_material = EXCLUDED.has_material,
                 document_url = EXCLUDED.document_url,
                 fetched_at   = NOW()""",
            values,
            template="(%s, %s, %s, %s, %s::TEXT[], %s, %s, %s, NOW())",
        )
        conn.commit()
    return len(filings)


def main():
    conn = psycopg2.connect(TSDB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

        cik_map = load_cik_map(conn)
        print(f"[sec_8k_ingest] Loaded {len(cik_map)} CIK→ticker mappings")

        watermark = get_watermark(conn)
        print(f"[sec_8k_ingest] Watermark: {watermark.isoformat()}")

        filings = fetch_filings(watermark)
        print(f"[sec_8k_ingest] Fetched {len(filings)} 8-K entries from EDGAR")

        count = upsert_filings(conn, filings, cik_map)
        print(f"[sec_8k_ingest] Upserted {count} filings")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM crucix_sec_8k_filings")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM crucix_sec_8k_filings WHERE has_material")
            material = cur.fetchone()[0]
            print(f"[sec_8k_ingest] Table totals: {total} filings, {material} material")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
