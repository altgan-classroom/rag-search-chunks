import pytest

celery = pytest.importorskip("celery")


def test_task_is_configured_for_safe_redelivery_and_json_only():
    from orca_rag.ingestion import tasks
    c = tasks.app.conf
    assert c.task_acks_late and c.task_reject_on_worker_lost and c.accept_content == ["json"]
    assert c.broker_transport_options["visibility_timeout"] > c.task_time_limit  # else Redis redelivers running tasks
    assert tasks.ingest_file.autoretry_for == tasks.TRANSIENT and tasks.ingest_file.max_retries == tasks.MAX_RETRIES


def test_permanent_failure_is_not_retried_and_is_reported(monkeypatch):
    from orca_rag.ingestion import tasks
    from orca_rag.ingestion.pipeline import IngestReport

    class FakeIngestor:
        def ingest_file(self, path, **kw):
            return IngestReport(source_key="x.pdf", status="failed", error="ValueError: no indexable text")

    monkeypatch.setattr(tasks, "_ing", FakeIngestor())
    out = tasks.ingest_file.apply(args=["x.pdf"]).get()  # eager: runs inline
    assert out["status"] == "failed" and "no indexable text" in out["error"]


def test_transient_failure_raises_for_retry(monkeypatch):
    from orca_rag.ingestion import tasks
    from orca_rag.ingestion.pipeline import IngestReport

    class Flaky:
        def ingest_file(self, path, **kw):
            return IngestReport(source_key="x.pdf", status="failed", error="ResponseHandlingException: qdrant down", transient=True)

    monkeypatch.setattr(tasks, "_ing", Flaky())
    res = tasks.ingest_file.apply(args=["x.pdf"])
    assert res.failed() or res.state in ("RETRY", "FAILURE")


def test_bad_visibility_configuration_is_rejected(monkeypatch):
    import importlib
    from orca_rag.ingestion import tasks
    monkeypatch.setenv("ORCA_CELERY_VISIBILITY_S", "100")  # shorter than the 2400 s task limit: Redis would run tasks twice
    with pytest.raises(ValueError, match="VISIBILITY"):
        importlib.reload(tasks)
    monkeypatch.delenv("ORCA_CELERY_VISIBILITY_S")
    importlib.reload(tasks)


def test_lab_style_short_timeouts_are_accepted(monkeypatch):
    import importlib
    from orca_rag.ingestion import tasks
    for k, v in {"ORCA_CELERY_TIME_LIMIT_S": "120", "ORCA_CELERY_SOFT_LIMIT_S": "100", "ORCA_CELERY_VISIBILITY_S": "150"}.items():
        monkeypatch.setenv(k, v)
    importlib.reload(tasks)
    assert tasks.app.conf.broker_transport_options["visibility_timeout"] == 150
    for k in ("ORCA_CELERY_TIME_LIMIT_S", "ORCA_CELERY_SOFT_LIMIT_S", "ORCA_CELERY_VISIBILITY_S"):
        monkeypatch.delenv(k)
    importlib.reload(tasks)
