import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import redis


REDIS_URL = os.getenv("REDIS_URL")
POLL_INTERVAL_SECONDS = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "1.0"))
TRANSIENT_RETRY_SECONDS = float(os.getenv("JOB_TRANSIENT_RETRY_SECONDS", "8.0"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))

_worker_started = False
_worker_lock = threading.Lock()
_redis_client = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _redis():
    global _redis_client
    if _redis_client is None:
        if not REDIS_URL:
            raise RuntimeError("REDIS_URL is not configured")
        _redis_client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            health_check_interval=30,
        )
        _redis_client.ping()
    return _redis_client


def _job_key(job_id):
    return f"studyverse:mcq:job:{job_id}"


def _queue_key():
    return "studyverse:mcq:queue"


def enqueue_job(payload):
    client = _redis()
    job_id = f"quiz_{uuid.uuid4().hex[:12]}"
    now = _now()

    job = {
        "job_id": job_id,
        "status": "queued",
        "payload": payload,
        "result": None,
        "error": None,
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
        "not_before": 0,
    }

    pipe = client.pipeline()
    pipe.set(_job_key(job_id), json.dumps(job), ex=JOB_TTL_SECONDS)
    pipe.rpush(_queue_key(), job_id)
    pipe.execute()
    return job_id


def get_job(job_id):
    client = _redis()
    raw = client.get(_job_key(job_id))
    if not raw:
        return None

    job = json.loads(raw)
    result = {
        "job_id": job["job_id"],
        "status": job["status"],
        "attempts": job.get("attempts", 0),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job.get("error"),
    }

    if job.get("result") is not None:
        result["result"] = job["result"]

    if job["status"] == "queued":
        queue_ids = client.lrange(_queue_key(), 0, -1)
        try:
            result["queue_position"] = queue_ids.index(job_id) + 1
        except ValueError:
            result["queue_position"] = 1

    return result


def _save_job(job):
    _redis().set(
        _job_key(job["job_id"]),
        json.dumps(job),
        ex=JOB_TTL_SECONDS,
    )


def _claim_next_job():
    client = _redis()

    while True:
        job_id = client.lpop(_queue_key())
        if not job_id:
            return None

        raw = client.get(_job_key(job_id))
        if not raw:
            continue

        job = json.loads(raw)

        if job.get("status") != "queued":
            continue

        not_before = float(job.get("not_before", 0) or 0)
        if not_before > time.time():
            client.rpush(_queue_key(), job_id)
            time.sleep(min(1.0, max(0.05, not_before - time.time())))
            continue

        job["status"] = "processing"
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["updated_at"] = _now()
        _save_job(job)

        return {
            "id": job_id,
            "payload": job["payload"],
            "attempts": job["attempts"],
        }


def _complete_job(job_id, result):
    raw = _redis().get(_job_key(job_id))
    if not raw:
        return
    job = json.loads(raw)
    job["status"] = "completed"
    job["result"] = result
    job["error"] = None
    job["updated_at"] = _now()
    _save_job(job)


def _fail_job(job_id, error):
    raw = _redis().get(_job_key(job_id))
    if not raw:
        return
    job = json.loads(raw)
    job["status"] = "failed"
    job["error"] = str(error)[:2000]
    job["updated_at"] = _now()
    _save_job(job)


def _requeue_job(job_id, error):
    client = _redis()
    raw = client.get(_job_key(job_id))
    if not raw:
        return

    job = json.loads(raw)
    job["status"] = "queued"
    job["error"] = str(error)[:2000]
    job["not_before"] = time.time() + TRANSIENT_RETRY_SECONDS
    job["updated_at"] = _now()
    _save_job(job)
    client.rpush(_queue_key(), job_id)


def recover_interrupted_jobs():
    client = _redis()
    pattern = "studyverse:mcq:job:*"
    recovered = 0

    for key in client.scan_iter(match=pattern, count=100):
        raw = client.get(key)
        if not raw:
            continue
        job = json.loads(raw)

        if job.get("status") == "processing":
            job["status"] = "queued"
            job["error"] = (
                "Previous worker stopped before this job finished; retrying automatically."
            )
            job["not_before"] = 0
            job["updated_at"] = _now()
            _save_job(job)
            client.rpush(_queue_key(), job["job_id"])
            recovered += 1

    if recovered:
        print(f"♻️ Requeued {recovered} interrupted MCQ job(s)")


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

    # Fail fast during startup if Redis is not configured or unreachable.
    _redis()
    recover_interrupted_jobs()

    def worker_loop():
        print("🧵 Studyverse Redis MCQ queue worker started")
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
