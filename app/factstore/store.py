"""
Fact Store — Persistent Read/Write Interface
=============================================
Dual-backend storage: Postgres (Supabase, production) or SQLite (local dev).

The backend is selected automatically based on the DATABASE_URL environment
variable.  If it starts with "postgresql://", Postgres is used; otherwise
SQLite.

Design principles:
  - Per-stage checkpointing: each agent writes immediately, not only on
    job success.  A failure at stage N never discards completed work from
    stages 1..N-1 (AGENTS.md rule 5).
  - Every write is a real DB commit — not buffered until end-of-job.
  - Thread-safe: uses a per-thread connection pattern for SQLite,
    connection pooling for Postgres.

Reference: docs/ARTHA_ARCHITECTURE.md §4.4
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from app.factstore.schemas import (
    BaseFact,
    FactType,
    FundamentalRow,
    NarrativeFact,
    NewsSignal,
    QuantSignal,
    ReportCitation,
    RiskFlag,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseFact)

# Registry mapping FactType → Pydantic model class
_FACT_MODELS: dict[FactType, Type[BaseFact]] = {
    FactType.FUNDAMENTAL: FundamentalRow,
    FactType.QUANT_SIGNAL: QuantSignal,
    FactType.RISK_FLAG: RiskFlag,
    FactType.NEWS_SIGNAL: NewsSignal,
    FactType.NARRATIVE: NarrativeFact,
    FactType.REPORT_CITATION: ReportCitation,
}


# ── Backend-agnostic SQL ────────────────────────────────────────────────────────

_CREATE_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id     TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    fact_type   TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    data_json   TEXT NOT NULL
)
"""

_CREATE_TABLE_POSTGRES = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id     TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL,
    fact_type   TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    data_json   JSONB NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_facts_job_id ON facts (job_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_job_type ON facts (job_id, fact_type)",
    "CREATE INDEX IF NOT EXISTS idx_facts_type ON facts (fact_type)",
]


class FactStore:
    """
    Dual-backend fact store supporting both SQLite (dev) and Postgres (prod).

    Usage::

        # Auto-selects backend from DATABASE_URL env var
        store = FactStore()

        # Explicit SQLite
        store = FactStore(db_url="sqlite:///data_cache/factstore.db")

        # Explicit Postgres (Supabase)
        store = FactStore(db_url="postgresql://user:pass@host:5432/db")
    """

    def __init__(self, db_url: Optional[str] = None, db_path: Optional[str] = None) -> None:
        """
        Initialize the Fact Store.

        Parameters
        ----------
        db_url : str, optional
            Database URL.  If starts with "postgresql://", uses Postgres.
            Otherwise uses SQLite.  Defaults to DATABASE_URL env var,
            falling back to SQLite at data_cache/factstore.db.
        db_path : str, optional
            Legacy parameter — direct path to SQLite file.
            Takes precedence over db_url for backward compatibility.
        """
        if db_path is not None:
            # Legacy SQLite path (backward compat with existing tests)
            self._backend = "sqlite"
            self._db_path = db_path
            self._pg_dsn = None
        elif db_url is not None:
            if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
                self._backend = "postgres"
                self._pg_dsn = db_url
                self._db_path = None
            else:
                self._backend = "sqlite"
                self._db_path = db_url.replace("sqlite:///", "")
                self._pg_dsn = None
        else:
            # Auto-detect from environment
            env_url = os.getenv("DATABASE_URL", "").strip()
            if env_url and (env_url.startswith("postgresql://") or env_url.startswith("postgres://")):
                self._backend = "postgres"
                self._pg_dsn = env_url
                self._db_path = None
            else:
                self._backend = "sqlite"
                self._db_path = env_url.replace("sqlite:///", "") if env_url else "data_cache/factstore.db"
                self._pg_dsn = None

        # SQLite thread-local storage
        self._local = threading.local()

        # Postgres connection pool (lazy init)
        self._pg_pool = None

        # Ensure schema
        self._ensure_schema()
        logger.info("FactStore initialized with backend=%s", self._backend)

    # ── Connection management ───────────────────────────────────────────────

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _get_pg_conn(self):
        """Return a Postgres connection from the pool."""
        if self._pg_pool is None:
            try:
                import psycopg2
                from psycopg2 import pool as pg_pool

                self._pg_pool = pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self._pg_dsn,
                )
            except ImportError:
                raise RuntimeError(
                    "psycopg2 is required for Postgres backend. "
                    "Install with: pip install psycopg2-binary"
                )
        return self._pg_pool.getconn()

    def _put_pg_conn(self, conn):
        """Return a Postgres connection to the pool."""
        if self._pg_pool is not None:
            self._pg_pool.putconn(conn)

    def _ensure_schema(self) -> None:
        """Create the facts table if it doesn't exist."""
        if self._backend == "sqlite":
            conn = self._get_sqlite_conn()
            conn.execute(_CREATE_TABLE_SQLITE)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.commit()
        else:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_POSTGRES)
                    for idx_sql in _CREATE_INDEXES:
                        cur.execute(idx_sql)
                conn.commit()
            finally:
                self._put_pg_conn(conn)

    # ── Write ───────────────────────────────────────────────────────────────

    def put_fact(self, fact: BaseFact) -> str:
        """
        Persist a fact immediately (per-stage checkpointing).
        Returns the fact_id.
        """
        data_json = fact.model_dump_json()
        params = (
            fact.fact_id,
            fact.job_id,
            fact.fact_type.value,
            fact.source,
            fact.created_at.isoformat(),
            data_json,
        )

        if self._backend == "sqlite":
            conn = self._get_sqlite_conn()
            conn.execute(
                "INSERT OR REPLACE INTO facts (fact_id, job_id, fact_type, source, created_at, data_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                params,
            )
            conn.commit()
        else:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO facts (fact_id, job_id, fact_type, source, created_at, data_json) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (fact_id) DO UPDATE SET "
                        "job_id=EXCLUDED.job_id, fact_type=EXCLUDED.fact_type, "
                        "source=EXCLUDED.source, created_at=EXCLUDED.created_at, "
                        "data_json=EXCLUDED.data_json",
                        params,
                    )
                conn.commit()
            finally:
                self._put_pg_conn(conn)

        return fact.fact_id

    def put_facts(self, facts: list[BaseFact]) -> list[str]:
        """Bulk insert with a single commit for efficiency."""
        if self._backend == "sqlite":
            conn = self._get_sqlite_conn()
            ids = []
            for fact in facts:
                data_json = fact.model_dump_json()
                conn.execute(
                    "INSERT OR REPLACE INTO facts (fact_id, job_id, fact_type, source, created_at, data_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (fact.fact_id, fact.job_id, fact.fact_type.value,
                     fact.source, fact.created_at.isoformat(), data_json),
                )
                ids.append(fact.fact_id)
            conn.commit()
            return ids
        else:
            conn = self._get_pg_conn()
            try:
                ids = []
                with conn.cursor() as cur:
                    for fact in facts:
                        data_json = fact.model_dump_json()
                        cur.execute(
                            "INSERT INTO facts (fact_id, job_id, fact_type, source, created_at, data_json) "
                            "VALUES (%s, %s, %s, %s, %s, %s) "
                            "ON CONFLICT (fact_id) DO UPDATE SET "
                            "job_id=EXCLUDED.job_id, fact_type=EXCLUDED.fact_type, "
                            "source=EXCLUDED.source, created_at=EXCLUDED.created_at, "
                            "data_json=EXCLUDED.data_json",
                            (fact.fact_id, fact.job_id, fact.fact_type.value,
                             fact.source, fact.created_at.isoformat(), data_json),
                        )
                        ids.append(fact.fact_id)
                conn.commit()
                return ids
            finally:
                self._put_pg_conn(conn)

    # ── Read ────────────────────────────────────────────────────────────────

    def _execute_query(self, sql_sqlite: str, sql_pg: str, params: tuple) -> list[dict]:
        """Execute a read query against the active backend."""
        if self._backend == "sqlite":
            conn = self._get_sqlite_conn()
            rows = conn.execute(sql_sqlite, params).fetchall()
            return [dict(r) for r in rows]
        else:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql_pg, params)
                    cols = [desc[0] for desc in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
            finally:
                self._put_pg_conn(conn)

    def get_fact(self, fact_id: str) -> Optional[BaseFact]:
        """Retrieve a single fact by its ID."""
        sql = "SELECT fact_type, data_json FROM facts WHERE fact_id = "
        rows = self._execute_query(
            sql + "?", sql + "%s", (fact_id,)
        )
        if not rows:
            return None
        r = rows[0]
        data_json = r["data_json"] if isinstance(r["data_json"], str) else json.dumps(r["data_json"])
        return self._deserialize(r["fact_type"], data_json)

    def get_facts_by_type(self, job_id: str, fact_type: FactType) -> list[BaseFact]:
        """Get all facts of a given type for a job."""
        sql = "SELECT fact_type, data_json FROM facts WHERE job_id = {p} AND fact_type = {p} ORDER BY created_at"
        rows = self._execute_query(
            sql.format(p="?"),
            sql.format(p="%s"),
            (job_id, fact_type.value),
        )
        return [self._deserialize(
            r["fact_type"],
            r["data_json"] if isinstance(r["data_json"], str) else json.dumps(r["data_json"]),
        ) for r in rows]

    def get_facts_for_symbol(self, job_id: str, symbol: str) -> list[BaseFact]:
        """Get all facts mentioning a symbol within a job."""
        if self._backend == "sqlite":
            sql = "SELECT fact_type, data_json FROM facts WHERE job_id = ? AND data_json LIKE ? ORDER BY created_at"
            params = (job_id, f"%{symbol}%")
        else:
            # Postgres JSONB contains operator is more efficient
            sql = "SELECT fact_type, data_json FROM facts WHERE job_id = %s AND data_json::text LIKE %s ORDER BY created_at"
            params = (job_id, f"%{symbol}%")

        rows = self._execute_query(sql, sql, params)
        return [self._deserialize(
            r["fact_type"],
            r["data_json"] if isinstance(r["data_json"], str) else json.dumps(r["data_json"]),
        ) for r in rows]

    def get_job_facts(self, job_id: str) -> list[BaseFact]:
        """Get ALL facts for a job, ordered by creation time."""
        sql = "SELECT fact_type, data_json FROM facts WHERE job_id = {p} ORDER BY created_at"
        rows = self._execute_query(sql.format(p="?"), sql.format(p="%s"), (job_id,))
        return [self._deserialize(
            r["fact_type"],
            r["data_json"] if isinstance(r["data_json"], str) else json.dumps(r["data_json"]),
        ) for r in rows]

    def get_validated_signals(self, job_id: str) -> list[QuantSignal]:
        """Get only validated QuantSignals for a job."""
        all_signals = self.get_facts_by_type(job_id, FactType.QUANT_SIGNAL)
        return [s for s in all_signals if isinstance(s, QuantSignal) and s.validated]

    def count_facts(self, job_id: str) -> dict[str, int]:
        """Count facts by type for a job."""
        sql = "SELECT fact_type, COUNT(*) as cnt FROM facts WHERE job_id = {p} GROUP BY fact_type"
        rows = self._execute_query(sql.format(p="?"), sql.format(p="%s"), (job_id,))
        return {r["fact_type"]: r["cnt"] for r in rows}

    # ── Delete ──────────────────────────────────────────────────────────────

    def _execute_modify(self, sql_sqlite: str, sql_pg: str, params: tuple) -> int:
        """Execute a modifying query and return rowcount."""
        if self._backend == "sqlite":
            conn = self._get_sqlite_conn()
            cursor = conn.execute(sql_sqlite, params)
            conn.commit()
            return cursor.rowcount
        else:
            conn = self._get_pg_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql_pg, params)
                    rowcount = cur.rowcount
                conn.commit()
                return rowcount
            finally:
                self._put_pg_conn(conn)

    def delete_job_facts(self, job_id: str) -> int:
        """Delete all facts for a job.  Returns the count deleted."""
        return self._execute_modify(
            "DELETE FROM facts WHERE job_id = ?",
            "DELETE FROM facts WHERE job_id = %s",
            (job_id,),
        )

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a single fact.  Returns True if it existed."""
        return self._execute_modify(
            "DELETE FROM facts WHERE fact_id = ?",
            "DELETE FROM facts WHERE fact_id = %s",
            (fact_id,),
        ) > 0

    # ── Internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _deserialize(fact_type_str: str, data_json: str) -> BaseFact:
        """Reconstruct a typed Pydantic model from stored JSON."""
        ft = FactType(fact_type_str)
        model_cls = _FACT_MODELS.get(ft)
        if model_cls is None:
            raise ValueError(f"Unknown fact type: {fact_type_str}")
        return model_cls.model_validate_json(data_json)

    def close(self) -> None:
        """Close connections."""
        if self._backend == "sqlite":
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
                self._local.conn = None
        elif self._pg_pool is not None:
            self._pg_pool.closeall()
            self._pg_pool = None
