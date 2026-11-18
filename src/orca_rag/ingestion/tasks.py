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

REDIS = os.environ.get("ORCA_REDIS_URL", "redis://localhost:6379/0")
app = Celery("orca", broker=REDIS, backend=REDIS)
app.conf.update(
    task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1, task_track_started=True,
    broker_transport_options={"visibility_timeout": 4 * 3600},  # must exceed task_time_limit or tasks run twice
    task_soft_time_limit=1800, task_time_limit=2400, result_expires=24 * 3600, worker_max_tasks_per_child=200,
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


@app.task(bind=True, autoretry_for=TRANSIENT, retry_backoff=10, retry_backoff_max=300, retry_jitter=True, max_retries=4)
def ingest_file(self, path: str, tenant: str = "default", acl: list[str] | None = None, force: bool = False) -> dict:
    rep = _ingestor().ingest_file(path, tenant=tenant, acl=acl, force=force)
    if rep.status == "failed" and any(t.__name__ in rep.error for t in TRANSIENT):
        raise ConnectionError(rep.error)  # transient: let Celery retry with backoff
    return rep.dict()
