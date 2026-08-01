"""
SQLite persistence layer for paper reviews, batch jobs, and statistics.

Tables:
  papers     — uploaded papers metadata
  reviews    — one review per paper/model combination
  batch_jobs — batch review job tracking
  batch_items— individual items within a batch job
"""

import sqlite3
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "reviews.db"

os.makedirs(DB_DIR, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            title       TEXT DEFAULT '',
            text        TEXT NOT NULL,
            raw_text    TEXT DEFAULT '',
            is_clean    INTEGER DEFAULT 0,
            upload_time TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id          INTEGER NOT NULL,
            model_type        TEXT NOT NULL DEFAULT 'api',
            novelty           REAL DEFAULT 0,
            soundness         REAL DEFAULT 0,
            presentation      REAL DEFAULT 0,
            overall           REAL DEFAULT 0,
            decision          TEXT DEFAULT 'Unknown',
            review_text       TEXT DEFAULT '',
            defense_active    INTEGER DEFAULT 0,
            defense_triggered INTEGER DEFAULT 0,
            review_time       TEXT NOT NULL,
            source_scores     TEXT DEFAULT '{}',
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS batch_jobs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            job_type       TEXT NOT NULL DEFAULT 'local',
            status         TEXT NOT NULL DEFAULT 'pending',
            total          INTEGER DEFAULT 0,
            completed      INTEGER DEFAULT 0,
            created_time   TEXT NOT NULL,
            completed_time TEXT
        );

        CREATE TABLE IF NOT EXISTS batch_items (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id  INTEGER NOT NULL,
            paper_id  INTEGER NOT NULL,
            review_id INTEGER,
            status    TEXT NOT NULL DEFAULT 'pending',
            error     TEXT,
            FOREIGN KEY (batch_id) REFERENCES batch_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_reviews_paper ON reviews(paper_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_time ON reviews(review_time);
        CREATE INDEX IF NOT EXISTS idx_items_batch ON batch_items(batch_id);
    """)
    conn.commit()
    conn.close()


# ---- Papers ----

def add_paper(filename: str, text: str, raw_text: str = "",
              title: str = "", is_clean: bool = False) -> int:
    conn = get_db()
    c = conn.execute(
        "INSERT INTO papers (filename, title, text, raw_text, is_clean, upload_time) VALUES (?,?,?,?,?,?)",
        (filename, title, text, raw_text, 1 if is_clean else 0, _now())
    )
    conn.commit()
    pid = c.lastrowid
    conn.close()
    return pid


def get_paper(paper_id: int) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_papers(limit: int = 50, offset: int = 0,
                with_review_only: bool = False) -> List[Dict]:
    conn = get_db()
    if with_review_only:
        rows = conn.execute(
            """SELECT DISTINCT p.* FROM papers p
               INNER JOIN reviews r ON r.paper_id = p.id
               ORDER BY p.upload_time DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM papers ORDER BY upload_time DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_papers() -> int:
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()
    conn.close()
    return row["c"]


# ---- Reviews ----

def add_review(paper_id: int, model_type: str, novelty: float = 0,
               soundness: float = 0, presentation: float = 0,
               overall: float = 0, decision: str = "Unknown",
               review_text: str = "", defense_active: bool = False,
               defense_triggered: bool = False,
               source_scores: Optional[Dict] = None) -> int:
    conn = get_db()
    c = conn.execute(
        """INSERT INTO reviews (paper_id, model_type, novelty, soundness,
           presentation, overall, decision, review_text, defense_active,
           defense_triggered, review_time, source_scores)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (paper_id, model_type, novelty, soundness, presentation, overall,
         decision, review_text, 1 if defense_active else 0,
         1 if defense_triggered else 0, _now(),
         json.dumps(source_scores or {}, ensure_ascii=False))
    )
    conn.commit()
    rid = c.lastrowid
    conn.close()
    return rid


def get_review(review_id: int) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_reviews_for_paper(paper_id: int) -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM reviews WHERE paper_id=? ORDER BY review_time DESC",
        (paper_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_review(paper_id: int) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM reviews WHERE paper_id=? ORDER BY review_time DESC LIMIT 1",
        (paper_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---- Batch Jobs ----

def create_batch(job_type: str, total: int = 0) -> int:
    conn = get_db()
    c = conn.execute(
        "INSERT INTO batch_jobs (job_type, status, total, completed, created_time) VALUES (?,?,?,?,?)",
        (job_type, "pending", total, 0, _now())
    )
    conn.commit()
    jid = c.lastrowid
    conn.close()
    return jid


def update_batch_status(batch_id: int, status: str, completed: int = None):
    conn = get_db()
    if completed is not None:
        if status in ("completed", "failed"):
            conn.execute(
                "UPDATE batch_jobs SET status=?, completed=?, completed_time=? WHERE id=?",
                (status, completed, _now(), batch_id)
            )
        else:
            conn.execute(
                "UPDATE batch_jobs SET status=?, completed=? WHERE id=?",
                (status, completed, batch_id)
            )
    else:
        conn.execute(
            "UPDATE batch_jobs SET status=? WHERE id=?",
            (status, batch_id)
        )
    conn.commit()
    conn.close()


def get_batch(batch_id: int) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM batch_jobs WHERE id=?", (batch_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_batch_item(batch_id: int, paper_id: int, status: str = "pending") -> int:
    conn = get_db()
    c = conn.execute(
        "INSERT INTO batch_items (batch_id, paper_id, status) VALUES (?,?,?)",
        (batch_id, paper_id, status)
    )
    conn.commit()
    iid = c.lastrowid
    conn.close()
    return iid


def update_batch_item(item_id: int, status: str, review_id: int = None,
                      error: str = None):
    conn = get_db()
    conn.execute(
        "UPDATE batch_items SET status=?, review_id=?, error=? WHERE id=?",
        (status, review_id, error, item_id)
    )
    conn.commit()
    conn.close()


def get_batch_items(batch_id: int) -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT bi.*, p.filename, p.title
           FROM batch_items bi
           LEFT JOIN papers p ON p.id = bi.paper_id
           WHERE bi.batch_id=?
           ORDER BY bi.id""",
        (batch_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Stats ----

def get_stats() -> Dict:
    conn = get_db()
    total_papers = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()["c"]
    total_reviews = conn.execute("SELECT COUNT(*) as c FROM reviews").fetchone()["c"]
    api_count = conn.execute(
        "SELECT COUNT(*) as c FROM reviews WHERE model_type='api'"
    ).fetchone()["c"]
    local_count = conn.execute(
        "SELECT COUNT(*) as c FROM reviews WHERE model_type IN ('local','local-batch')"
    ).fetchone()["c"]

    # Decision distribution
    decisions = conn.execute(
        """SELECT decision, COUNT(*) as c FROM reviews
           GROUP BY decision ORDER BY c DESC"""
    ).fetchall()
    decision_map = {r["decision"]: r["c"] for r in decisions}

    # Average scores by model type
    avg_scores = conn.execute(
        """SELECT model_type,
                  AVG(novelty) as avg_novelty,
                  AVG(soundness) as avg_soundness,
                  AVG(presentation) as avg_presentation,
                  AVG(overall) as avg_overall,
                  COUNT(*) as c
           FROM reviews
           WHERE overall > 0
           GROUP BY model_type"""
    ).fetchall()

    # Score distribution (overall)
    score_dist = conn.execute(
        """SELECT CAST(ROUND(overall) AS INTEGER) as score, COUNT(*) as c
           FROM reviews WHERE overall > 0
           GROUP BY score ORDER BY score"""
    ).fetchall()
    score_map = {r["score"]: r["c"] for r in score_dist}

    # Defense stats
    defense_total = conn.execute(
        "SELECT COUNT(*) as c FROM reviews WHERE defense_active=1"
    ).fetchone()["c"]
    defense_triggered = conn.execute(
        "SELECT COUNT(*) as c FROM reviews WHERE defense_triggered=1"
    ).fetchone()["c"]

    # Recent trend (last 10 reviews)
    recent = conn.execute(
        """SELECT overall, decision, model_type, review_time
           FROM reviews WHERE overall > 0
           ORDER BY review_time DESC LIMIT 10"""
    ).fetchall()

    conn.close()

    return {
        "total_papers": total_papers,
        "total_reviews": total_reviews,
        "api_reviews": api_count,
        "local_reviews": local_count,
        "decisions": decision_map,
        "average_scores": [dict(r) for r in avg_scores],
        "score_distribution": score_map,
        "defense_active": defense_total,
        "defense_triggered": defense_triggered,
        "recent": [dict(r) for r in recent],
    }


# Initialize DB on import
init_db()
