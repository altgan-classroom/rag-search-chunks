"""Celery worker for bulk/async ingestion. Replaces the baseline worker (REVIEW I1, I4, I18, S7, S8).

  * payloads are plain JSON (paths + tenant + ACL); credentials never travel through the broker
  * retries ONLY transient failures (network/IO), with exponential backoff + jitter; a corrupt or empty document
    fails once and is reported, instead of being re-OCR'd three times
  * acks_late + reject_on_worker_lost: a crashed worker's task is redelivered; ingestion is idempotent so that is safe
  * the embedder is loaded once per worker process
"""
from __future__ import annotations

import os

from celery import Celery
from celery.signals import worker_process_init

from ..config.settings import get_env, load_profile
from ..llm.gateway import LLMGateway
from ..store.base import make_store
from .pipeline import Ingestor


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


REDIS = os.environ.get("ORCA_REDIS_URL", "redis://localhost:6379/0")
# Tunable without code changes. With a Redis broker a SIGKILLed worker's in-flight tasks are redelivered only after
# `visibility_timeout`, so production keeps it long (it MUST exceed the longest task or tasks run twice), while a demo
# with tiny files shortens all four (ORCA_CELERY_*), e.g. limit 120 s, visibility 150 s.
VISIBILITY_S = _int("ORCA_CELERY_VISIBILITY_S", 4 * 3600)
TIME_LIMIT_S = _int("ORCA_CELERY_TIME_LIMIT_S", 2400)
SOFT_LIMIT_S = _int("ORCA_CELERY_SOFT_LIMIT_S", 1800)
RETRY_BACKOFF_S = _int("ORCA_CELERY_RETRY_BACKOFF_S", 10)
MAX_RETRIES = _int("ORCA_CELERY_MAX_RETRIES", 4)
if VISIBILITY_S <= TIME_LIMIT_S or SOFT_LIMIT_S >= TIME_LIMIT_S:
    raise ValueError("need ORCA_CELERY_SOFT_LIMIT_S < ORCA_CELERY_TIME_LIMIT_S < ORCA_CELERY_VISIBILITY_S")
app = Celery("orca", broker=REDIS, backend=REDIS)
app.conf.update(
    task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1, task_track_started=True,
    broker_transport_options={"visibility_timeout": VISIBILITY_S},
    task_soft_time_limit=SOFT_LIMIT_S, task_time_limit=TIME_LIMIT_S, result_expires=24 * 3600, worker_max_tasks_per_child=200,
    task_serializer="json", result_serializer="json", accept_content=["json"],
)
_ing: Ingestor | None = None
TRANSIENT = (ConnectionError, TimeoutError, OSError)


@worker_process_init.connect
def _warm(**_):
    global _ing
    env, prof = get_env(), load_profile()
    _ing = Ingestor(make_store(env, prof), prof, LLMGateway(mock=env.llm.mock))


def _ingestor() -> Ingestor:
    global _ing
    if _ing is None:
        _warm()
    return _ing  # type: ignore[return-value]


@app.task(bind=True, autoretry_for=TRANSIENT, retry_backoff=RETRY_BACKOFF_S, retry_backoff_max=300, retry_jitter=True, max_retries=MAX_RETRIES)
def ingest_file(self, path: str, tenant: str = "default", acl: list[str] | None = None, force: bool = False) -> dict:
    rep = _ingestor().ingest_file(path, tenant=tenant, acl=acl, force=force)
    if rep.status == "failed" and rep.transient:
        raise ConnectionError(rep.error)  # a dependency is down: let Celery retry with backoff (a bad document is NOT retried)
    return rep.dict()
