#!/usr/bin/env python3
"""Lightweight persistent memory CLI for this workspace.

The system keeps a local SQLite index in `.memory/memory.db` for fast search.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _detect_workspace_root() -> Path:
    """Find the workspace root by walking upward from this script file."""
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / ".git").exists() or (parent / "AGENTS.md").exists():
            return parent
    return here.parents[4]


ROOT = _detect_workspace_root()
DB_PATH = ROOT / ".memory" / "memory.db"
WEIGHT_RELEVANCE = 0.60
WEIGHT_RECENCY = 0.20
WEIGHT_USAGE = 0.10
WEIGHT_TAG = 0.10


def utc_now() -> str:
    """Return the current UTC timestamp in a compact ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect() -> sqlite3.Connection:
    """Open the SQLite database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Return True when a SQLite table or virtual table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table','view')",
        (name,),
    ).fetchone()
    return row is not None


def init_db(conn: sqlite3.Connection) -> bool:
    """Create schema and return whether FTS5 indexing is available."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            hits INTEGER NOT NULL DEFAULT 0,
            last_seen_at TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS notes_created_at_idx ON notes(created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS notes_last_seen_at_idx ON notes(last_seen_at DESC)"
    )

    fts_available = True
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
            USING fts5(
                content,
                tags,
                source,
                content='notes',
                content_rowid='id',
                tokenize='porter unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        fts_available = False

    if fts_available:
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, content, tags, source)
                VALUES (new.id, new.content, new.tags, new.source);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, content, tags, source)
                VALUES ('delete', old.id, old.content, old.tags, old.source);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, content, tags, source)
                VALUES ('delete', old.id, old.content, old.tags, old.source);
                INSERT INTO notes_fts(rowid, content, tags, source)
                VALUES (new.id, new.content, new.tags, new.source);
            END;
            """
        )
        if _table_exists(conn, "notes_fts"):
            conn.execute("INSERT INTO notes_fts(notes_fts) VALUES ('rebuild')")

    conn.commit()
    return fts_available


def _normalize_tags(tags: str) -> str:
    """Normalize tags into a deduplicated comma-separated string."""
    parts = [p.strip().lower() for p in tags.split(",") if p.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    return ",".join(ordered)


def _content_hash(content: str) -> str:
    """Compute a stable hash for note deduplication."""
    normalized = " ".join(content.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def add_note(
    conn: sqlite3.Connection,
    *,
    content: str,
    tags: str,
    source: str,
) -> tuple[int | None, bool]:
    """Insert a note into the memory database.

    Returns (note_id, inserted) where inserted is False when deduplicated.
    """
    content = content.strip()
    if not content:
        raise ValueError("content cannot be empty")

    tags = _normalize_tags(tags)
    created_at = utc_now()
    digest = _content_hash(content)
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO notes(created_at, source, tags, content, content_hash)
        VALUES (?, ?, ?, ?, ?)
        """,
        (created_at, source.strip() or "manual", tags, content, digest),
    )
    conn.commit()

    inserted = cursor.rowcount > 0
    note_id: int | None = None
    if inserted:
        row = conn.execute("SELECT id FROM notes WHERE content_hash = ?", (digest,)).fetchone()
        note_id = int(row["id"]) if row else None
    return note_id, inserted


def _format_row(row: sqlite3.Row) -> str:
    """Format a search/recent row for CLI output."""
    tags = f" [{row['tags']}]" if row["tags"] else ""
    return f"{row['id']:>4} | {row['created_at']} | {row['source']}{tags}\n      {row['content']}"


def _format_search_row(row: sqlite3.Row, score: float) -> str:
    """Format a search row with recall metadata and ranking score."""
    base = _format_row(row)
    last_seen = row["last_seen_at"] or "never"
    hits = int(row["hits"] or 0)
    return f"{base}\n      hits={hits} last_seen={last_seen} score={score:.3f}"


def _query_tokens(query: str) -> set[str]:
    """Tokenize query text into lowercase words."""
    return {token for token in re.findall(r"[a-z0-9]+", query.lower()) if token}


def _fts_query_from_text(query: str) -> str:
    """Build a safe FTS5 query that handles punctuation-heavy input."""
    tokens = sorted(_query_tokens(query))
    if not tokens:
        return ""
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)


def _build_like_predicate(query: str) -> tuple[str, list[str]]:
    """Build a broad LIKE predicate from query tokens plus raw query."""
    tokens = sorted(_query_tokens(query))
    terms: list[str] = []
    if query.strip():
        terms.append(query.strip())
    for token in tokens:
        if token not in terms:
            terms.append(token)
    if not terms:
        terms = [query]

    clauses: list[str] = []
    params: list[str] = []
    for term in terms:
        clauses.append("(content LIKE ? OR tags LIKE ? OR source LIKE ?)")
        pattern = f"%{term}%"
        params.extend([pattern, pattern, pattern])
    return " OR ".join(clauses), params


def _tags_set(tags: str) -> set[str]:
    """Split tag CSV into a normalized set."""
    return {tag.strip().lower() for tag in tags.split(",") if tag.strip()}


def _iso_age_days(iso_value: str | None, now_ts: datetime) -> float:
    """Return age in days for an ISO timestamp; very large when missing/invalid."""
    if not iso_value:
        return 3650.0
    try:
        past = datetime.fromisoformat(iso_value)
    except ValueError:
        return 3650.0
    delta = now_ts - past
    return max(0.0, delta.total_seconds() / 86400.0)


def _recency_component(row: sqlite3.Row, now_ts: datetime) -> float:
    """Compute recency score combining creation time and last-seen time."""
    created_days = _iso_age_days(row["created_at"], now_ts)
    seen_days = _iso_age_days(row["last_seen_at"], now_ts)
    created_score = math.exp(-created_days / 14.0)
    seen_score = math.exp(-seen_days / 7.0)
    return max(created_score, seen_score)


def _usage_component(hits: int) -> float:
    """Compute usage score with capped log scaling."""
    if hits <= 0:
        return 0.0
    return min(1.0, math.log1p(hits) / math.log(11.0))


def _tag_component(query: str, tags: str) -> float:
    """Compute tag overlap score based on query token intersection."""
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tags_set(tags)
    if not t_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(t_tokens))
    return overlap / len(q_tokens)


def _relevance_component_from_bm25(bm25_score: float | None) -> float:
    """Map bm25 score to [0,1], where higher means more relevant."""
    if bm25_score is None:
        return 0.0
    return 1.0 / (1.0 + max(0.0, bm25_score))


def _relevance_component_like(query: str, row: sqlite3.Row) -> float:
    """Estimate relevance in LIKE fallback mode."""
    text = f"{row['content']} {row['tags']} {row['source']}".lower()
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return 0.0
    hits = sum(1 for token in q_tokens if token in text)
    return hits / len(q_tokens)


def _score_row(
    *,
    row: sqlite3.Row,
    query: str,
    now_ts: datetime,
    bm25_score: float | None,
) -> float:
    """Compute hybrid ranking score for a row."""
    relevance = (
        _relevance_component_from_bm25(bm25_score)
        if bm25_score is not None
        else _relevance_component_like(query, row)
    )
    recency = _recency_component(row, now_ts)
    usage = _usage_component(int(row["hits"] or 0))
    tag_match = _tag_component(query, row["tags"])
    return (
        WEIGHT_RELEVANCE * relevance
        + WEIGHT_RECENCY * recency
        + WEIGHT_USAGE * usage
        + WEIGHT_TAG * tag_match
    )


def _mark_recalled(conn: sqlite3.Connection, note_ids: list[int]) -> None:
    """Increment hits and update last_seen_at for recalled notes."""
    if not note_ids:
        return
    seen_at = utc_now()
    conn.executemany(
        """
        UPDATE notes
        SET hits = hits + 1, last_seen_at = ?
        WHERE id = ?
        """,
        [(seen_at, note_id) for note_id in note_ids],
    )
    conn.commit()


def _search_with_hybrid_ranking(
    conn: sqlite3.Connection,
    *,
    query: str,
    limit: int,
    fts_available: bool,
) -> list[tuple[sqlite3.Row, float]]:
    """Search notes and return top rows scored by hybrid ranking."""
    candidate_limit = max(limit * 5, 20)
    now_ts = datetime.now(timezone.utc)
    scored_rows: list[tuple[sqlite3.Row, float]] = []

    fts_query = _fts_query_from_text(query)
    if fts_available and fts_query:
        try:
            rows = conn.execute(
                """
                SELECT
                    n.id, n.created_at, n.source, n.tags, n.content, n.hits, n.last_seen_at,
                    bm25(notes_fts) AS bm25_score
                FROM notes_fts
                JOIN notes n ON n.id = notes_fts.rowid
                WHERE notes_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (fts_query, candidate_limit),
            ).fetchall()
            if rows:
                for row in rows:
                    score = _score_row(
                        row=row,
                        query=query,
                        now_ts=now_ts,
                        bm25_score=float(row["bm25_score"]),
                    )
                    scored_rows.append((row, score))
                scored_rows.sort(key=lambda item: item[1], reverse=True)
                return scored_rows[:limit]
        except sqlite3.OperationalError:
            pass

    where_clause, where_params = _build_like_predicate(query)
    rows = conn.execute(
        f"""
        SELECT id, created_at, source, tags, content, hits, last_seen_at
        FROM notes
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [*where_params, candidate_limit],
    ).fetchall()
    for row in rows:
        score = _score_row(row=row, query=query, now_ts=now_ts, bm25_score=None)
        scored_rows.append((row, score))
    scored_rows.sort(key=lambda item: item[1], reverse=True)
    return scored_rows[:limit]


def cmd_init(conn: sqlite3.Connection, _: argparse.Namespace) -> None:
    """Handle the init command."""
    fts_available = init_db(conn)
    print(f"initialized: {DB_PATH}")
    print(f"fts5: {'enabled' if fts_available else 'unavailable (LIKE fallback)'}")


def cmd_add(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """Handle the add command."""
    init_db(conn)
    note_id, inserted = add_note(
        conn,
        content=args.content,
        tags=args.tags or "",
        source=args.source or "manual",
    )
    if inserted:
        print(f"added note id={note_id}")
    else:
        print("skipped duplicate note")


def cmd_sync(conn: sqlite3.Connection, _: argparse.Namespace) -> None:
    """Handle the sync command in database-only mode."""
    init_db(conn)
    print("sync complete: database-only mode (no external files to index)")


def cmd_search(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """Handle the search command."""
    fts_available = init_db(conn)
    scored = _search_with_hybrid_ranking(
        conn,
        query=args.query,
        limit=args.limit,
        fts_available=fts_available,
    )
    if not scored:
        print("no matches")
        return
    note_ids = [int(row["id"]) for row, _ in scored]
    _mark_recalled(conn, note_ids)
    refreshed = conn.execute(
        f"""
        SELECT id, created_at, source, tags, content, hits, last_seen_at
        FROM notes
        WHERE id IN ({",".join(["?"] * len(note_ids))})
        """,
        note_ids,
    ).fetchall()
    refreshed_by_id = {int(row["id"]): row for row in refreshed}
    for row, score in scored:
        current = refreshed_by_id[int(row["id"])]
        print(_format_search_row(current, score))


def cmd_recent(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """Handle the recent command."""
    init_db(conn)
    rows = conn.execute(
        """
        SELECT id, created_at, source, tags, content
        FROM notes
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    if not rows:
        print("no notes yet")
        return
    for row in rows:
        print(_format_row(row))


def cmd_stats(conn: sqlite3.Connection, _: argparse.Namespace) -> None:
    """Handle the stats command."""
    fts_available = init_db(conn)
    row = conn.execute("SELECT COUNT(*) AS c FROM notes").fetchone()
    count = int(row["c"]) if row else 0
    recalled_row = conn.execute(
        """
        SELECT
            COUNT(*) AS recalled_count,
            COALESCE(AVG(hits), 0) AS avg_hits,
            MAX(last_seen_at) AS latest_seen
        FROM notes
        WHERE hits > 0
        """
    ).fetchone()
    recalled_count = int(recalled_row["recalled_count"]) if recalled_row else 0
    avg_hits = float(recalled_row["avg_hits"]) if recalled_row else 0.0
    latest_seen = recalled_row["latest_seen"] if recalled_row else None
    print(f"notes: {count}")
    print(f"recalled_notes: {recalled_count}")
    print(f"avg_hits_recalled: {avg_hits:.2f}")
    print(f"latest_last_seen_at: {latest_seen or 'never'}")
    print(f"db: {DB_PATH}")
    print(f"fts5: {'enabled' if fts_available else 'unavailable (LIKE fallback)'}")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI parser."""
    parser = argparse.ArgumentParser(description="Workspace memory CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="initialize memory database")
    init_parser.set_defaults(handler=cmd_init)

    add_parser = subparsers.add_parser("add", help="add a memory note")
    add_parser.add_argument("content", help="note content")
    add_parser.add_argument("--tags", default="", help="comma-separated tags")
    add_parser.add_argument("--source", default="manual", help="source label")
    add_parser.set_defaults(handler=cmd_add)

    sync_parser = subparsers.add_parser("sync", help="sync markdown notes into sqlite index")
    sync_parser.set_defaults(handler=cmd_sync)

    search_parser = subparsers.add_parser("search", help="search memory notes")
    search_parser.add_argument("query", help="search query")
    search_parser.add_argument("--limit", type=int, default=8, help="max results")
    search_parser.set_defaults(handler=cmd_search)

    recent_parser = subparsers.add_parser("recent", help="show recent notes")
    recent_parser.add_argument("--limit", type=int, default=10, help="max results")
    recent_parser.set_defaults(handler=cmd_recent)

    stats_parser = subparsers.add_parser("stats", help="show memory stats")
    stats_parser.set_defaults(handler=cmd_stats)
    return parser


def main() -> int:
    """Entry point for the memory CLI."""
    parser = build_parser()
    args = parser.parse_args()
    conn = connect()
    try:
        args.handler(conn, args)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
