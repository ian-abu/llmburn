"""Save each LLM call as one row in SQLite and read totals back out."""
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

GROUP_BY_COLUMNS = ("user_id", "plan", "feature", "model")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    user_id       TEXT,
    plan          TEXT,
    feature       TEXT,
    model         TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls (ts);
CREATE INDEX IF NOT EXISTS idx_calls_user_ts ON calls (user_id, ts);
"""


@dataclass
class Call:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached_tokens: int = 0
    user_id: Optional[str] = None
    plan: Optional[str] = None
    feature: Optional[str] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Total:
    key: Optional[str]
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _parse_path(db):
    """Accept 'sqlite:///file.db', a plain path, or ':memory:'."""
    return db.removeprefix("sqlite:///")


class Storage:
    def __init__(self, db="llmburn.db"):
        self.path = _parse_path(db)
        # One shared connection guarded by a lock, so Flask/FastAPI worker
        # threads can all write safely.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def save(self, call: Call):
        with self._lock:
            self._conn.execute(
                "INSERT INTO calls (ts, user_id, plan, feature, model,"
                " input_tokens, output_tokens, cached_tokens, cost_usd)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    call.ts.astimezone(timezone.utc).isoformat(),
                    call.user_id,
                    call.plan,
                    call.feature,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.cached_tokens,
                    call.cost_usd,
                ),
            )
            self._conn.commit()

    def totals(self, by="user_id", since: Optional[datetime] = None, limit=20):
        """Cost totals grouped by one column, most expensive first."""
        if by not in GROUP_BY_COLUMNS:
            # Column names can't be SQL parameters, so whitelist them.
            raise ValueError(f"by must be one of {GROUP_BY_COLUMNS}")
        sql = (
            f"SELECT {by}, COUNT(*), SUM(input_tokens), SUM(output_tokens),"
            " SUM(cost_usd) FROM calls"
        )
        params = []
        if since is not None:
            sql += " WHERE ts >= ?"
            params.append(since.astimezone(timezone.utc).isoformat())
        sql += f" GROUP BY {by} ORDER BY SUM(cost_usd) DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Total(key, n, i, o, round(c, 6)) for key, n, i, o, c in rows]

    def spent(self, user_id, since: datetime) -> float:
        """Total USD one user has spent since a point in time."""
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE user_id = ? AND ts >= ?",
                (user_id, since.astimezone(timezone.utc).isoformat()),
            ).fetchone()
        return total

    def spent_total(self, since: datetime) -> float:
        """Total USD the whole app has spent since a point in time."""
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE ts >= ?",
                (since.astimezone(timezone.utc).isoformat(),),
            ).fetchone()
        return total

    def close(self):
        with self._lock:
            self._conn.close()
