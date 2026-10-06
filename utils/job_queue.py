import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone


DB_PATH = os.getenv("STUDYVERSE_JOB_DB", "studyverse_jobs.db")
POLL_INTERVAL_SECONDS = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "1.0"))
TRANSIENT_RETRY_SECONDS = float(os.getenv("JOB_TRANSIENT_RETRY_SECONDS", "8.0"))

_worker_started = False
_worker_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_job_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mcq_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                not_before REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcq_jobs_status_created "
            "ON mcq_jobs(status, created_at)"
        )


def recover_interrupted_jobs():
    """Requeue jobs that were processing when the Gunicorn worker died."""
    init_job_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE mcq_jobs
            SET status = 'queued',
                error = 'Previous worker stopped before this job finished; retrying automatically.',
                not_before = 0,
                updated_at = ?
            WHERE status = 'processing'
            """,
            (_now(),),
        )
        recovered = cursor.rowcount or 0

    if recovered:
        print(f"♻️ Requeued {recovered} interrupted MCQ job(s)")


def enqueue_job(payload):
    init_job_db()
    job_id = f"quiz_{uuid.uuid4().hex[:12]}"
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO mcq_jobs
            (id, status, payload_json, created_at, updated_at)
            VALUES (?, 'queued', ?, ?, ?)
            """,
            (job_id, json.dumps(payload), now, now),
        )
    return job_id


def get_job(job_id):
    init_job_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM mcq_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

        if not row:
            return None

        result = {
            "job_id": row["id"],
            "status": row["status"],
            "attempts": row["attempts"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
        }

        if row["result_json"]:
            result["result"] = json.loads(row["result_json"])

        if row["status"] == "queued":
            position = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM mcq_jobs
                WHERE status = 'queued'
                  AND created_at <= ?
                """,
                (row["created_at"],),
            ).fetchone()["count"]
            result["queue_position"] = position

        return result


def _claim_next_job():
    init_job_db()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT *
            FROM mcq_jobs
            WHERE status = 'queued'
              AND not_before <= ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (time.time(),),
        ).fetchone()

        if not row:
            conn.commit()
            return None

        updated = _now()
        conn.execute(
            """
            UPDATE mcq_jobs
            SET status = 'processing',
                attempts = attempts + 1,
                updated_at = ?
            WHERE id = ? AND status = 'queued'
            """,
            (updated, row["id"]),
        )
        conn.commit()

        return {
            "id": row["id"],
            "payload": json.loads(row["payload_json"]),
            "attempts": row["attempts"] + 1,
        }
    finally:
        conn.close()


def _complete_job(job_id, result):
    with _connect() as conn:
        conn.execute(
            """
            UPDATE mcq_jobs
            SET status = 'completed',
                result_json = ?,
                error = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(result), _now(), job_id),
        )


def _fail_job(job_id, error):
    with _connect() as conn:
        conn.execute(
            """
            UPDATE mcq_jobs
            SET status = 'failed',
                error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (str(error)[:2000], _now(), job_id),
        )


def _requeue_job(job_id, error):
    with _connect() as conn:
        conn.execute(
            """
            UPDATE mcq_jobs
            SET status = 'queued',
                error = ?,
                not_before = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                str(error)[:2000],
                time.time() + TRANSIENT_RETRY_SECONDS,
                _now(),
                job_id,
            ),
        )


def _looks_transient(exc):
    message = str(exc).lower()
    transient_markers = (
        "no ai provider returned usable text",
        "rate limit",
        "rate_limit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )
    return any(marker in message for marker in transient_markers)


def start_job_worker(processor):
    global _worker_started

    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True

    init_job_db()
    recover_interrupted_jobs()

    def worker_loop():
        print("🧵 Studyverse MCQ queue worker started")
        while True:
            job = _claim_next_job()
            if not job:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            try:
                result = processor(job["payload"])
                _complete_job(job["id"], result)
                print(f"✅ Queue job completed: {job['id']}")
            except Exception as exc:
                if _looks_transient(exc):
                    _requeue_job(job["id"], exc)
                    print(
                        f"⏳ Queue job requeued after temporary AI failure: "
                        f"{job['id']}"
                    )
                else:
                    _fail_job(job["id"], exc)
                    print(f"❌ Queue job failed: {job['id']} — {exc}")

    thread = threading.Thread(
        target=worker_loop,
        name="studyverse-mcq-worker",
        daemon=True,
    )
    thread.start()
