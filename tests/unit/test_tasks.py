import pytest

celery = pytest.importorskip("celery")


def test_task_is_configured_for_safe_redelivery_and_json_only():
    from orca_rag.ingestion import tasks
    c = tasks.app.conf
    assert c.task_acks_late and c.task_reject_on_worker_lost and c.accept_content == ["json"]
    assert c.broker_transport_options["visibility_timeout"] > c.task_time_limit  # else Redis redelivers running tasks
    assert tasks.ingest_file.autoretry_for == tasks.TRANSIENT and tasks.ingest_file.max_retries == 4


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
            return IngestReport(source_key="x.pdf", status="failed", error="ConnectionError: qdrant down")

    monkeypatch.setattr(tasks, "_ing", Flaky())
    res = tasks.ingest_file.apply(args=["x.pdf"])
    assert res.failed() or res.state in ("RETRY", "FAILURE")
