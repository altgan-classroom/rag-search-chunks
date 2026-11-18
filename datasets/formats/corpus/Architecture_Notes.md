# Architecture notes

## Services
The ingest service writes documents to object storage and publishes a job to the queue.

## Data retention
Audit logs are retained for 400 days and then deleted by the nightly compaction job.
Customer exports are retained for 30 days.
