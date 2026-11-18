# RAG ingestion lab: put 5 files into a vector database, then look inside it

**Time:** 10 minutes for Part 1, plus 10 to 15 for Part 2 (optional).
**What you will see:** documents of five different types become searchable vectors in Qdrant, and you check the result
three ways: container logs, a verification command, and the Qdrant dashboard. Then you break things on purpose to see
what the design does about it.

## Before you start
* **Docker Desktop is running** (the whale icon says "Engine running") and `docker compose version` works.
* **Git** is installed.
* Docker has **4 GB of memory or more** (Docker Desktop, Settings, Resources).
* Internet is needed for step 2 only (building the image). Do step 2 at home if you can: the build downloads about 1 GB.

Every command below is one line. Copy it, run it from the project folder, read what it prints.

---

## Part 1: ingest and verify (about 10 minutes)

### 1. Get the project
```
git clone <REPO URL from the chat>
cd rag-ingest-docs
```
Look around: `data/raw` holds the 5 files you will ingest (a Word file, a PowerPoint, an Excel sheet, a CSV and a PDF).

### 2. Build the image
```
docker compose build
```
This packages the ingestion code **and** the two embedding models (the image is about 1 GB; 1 to 5 minutes the first time).
Once it is built, ingestion needs no internet.

### 3. Start the vector database
```
docker compose up -d qdrant
```
It listens on `localhost:6333`. Check it is up: open **http://localhost:6333/dashboard** in a browser. You should see an
empty "Collections" page.

### 4. Run the ingestion container
```
docker compose run --rm ingest
```
This starts a container, parses the 5 files, splits them into chunks, turns each chunk into a vector, writes them to Qdrant,
prints a report and exits. **Read the log.** You should see one line per file:
```
ingested  Employee_Handbook.docx      pages=  1 skip=  0 chunks=   1
ingested  Q3_Board_Update.pptx        pages=  4 skip=  0 chunks=   4
ingested  Safety_Manual.pdf           pages=  1 skip=  0 chunks=   1
ingested  Sales_Pipeline_2026.xlsx    pages=  3 skip=  0 chunks=   3
ingested  Supplier_Contacts.csv       pages=  1 skip=  0 chunks=   1
profile=advanced collection=orca_461716c917 total=1s stats={'points': 25, 'status': 'green'}
```
Notice: the slide deck became **4 chunks** (one per slide) and the workbook **3** (one per sheet), so an answer can cite
"slide 3". A line `Warning: RapidOCR is missing` is harmless (it is about scanned pages; there are none here).

### 5. Verify what is in the vector database
```
docker compose run --rm verify
```
Expected:
```
collection  orca_461716c917  status=green  points_count=25
vectors     dense: size=384 distance=COSINE, sparse: ['sparse']
by kind     child=10 (searchable, have vectors)  parent=10 (context text, no vectors)  doc=5 (one manifest per file)
sample point ...
  dense     dim=384 first6=[-0.008, -0.041, 0.031, ...] L2norm=1.000
  sparse    non-zero terms=21 ...
  payload   tenant=lab acl=['public'] doc=Q3_Board_Update page=3 section='Revenue by region'
search      'What was the Q3 revenue for APAC?' as tenant=lab
  1.000  Q3_Board_Update  p.3-3  'Revenue by region | Region | Q3 revenue ...'
```
Check yourself:
* **25 points = 10 + 10 + 5.** Only the 10 *child* points carry vectors. *Parent* points hold the larger section of text that
  is handed to the model for context, and each *doc* point is a small record of what was ingested.
* Each child has **two vectors**: a **dense** one (384 numbers, meaning) and a **sparse** one (exact words, BM25).
* The best hit for the APAC question is **slide 3**, cited as `p.3-3`.

### 6. Look at it in the dashboard
Open **http://localhost:6333/dashboard**, click your collection, then **Points**. Open one point and find its `payload`
(tenant, page, section, text) and its vectors. The **Info** tab shows vector size 384 and cosine distance.

### 7. Run it again
```
docker compose run --rm ingest
```
Every line now says `unchanged` and the point count is still 25. **Ingestion is idempotent**: the same files never create
duplicates. Try it: copy one more file of yours into `data/raw` (PDF, DOCX, PPTX, XLSX, CSV, HTML, MD or TXT; keep it to 5 or
6 files in total), run the command again, and see only the new file say `ingested`.

**You are done with Part 1.** You ingested 5 documents, read the logs, and checked the vectors three ways.

---

## Part 2: break it on purpose (optional)

A production system is judged by how it behaves when things go wrong. Here ingestion runs through a **queue** (Redis) and
**workers**, the way it would at scale.

### 8. A corrupt file must not stop the good ones
```
docker compose run --rm challenge
```
The folder `challenge/` holds one good HTML file and one corrupt PDF. Expected: the HTML line says `ingested`, the PDF line says
`failed ... FileDataError`, the command exits with an error code, and the good file is in the database. A bad document fails
once, alone, and is not retried (retrying cannot fix it).

### 9. Start the queue and one worker
```
docker compose --profile queue up -d redis worker
```
```
docker compose run --rm enqueue
```
`enqueue` puts one job per file on the queue; the worker takes them. Expected: `done in 0.6s status={'unchanged': 5}`.

### 10. Take the database down while jobs are running
```
docker compose stop qdrant
```
```
docker compose run --rm enqueue /data/raw --force
```
```
docker compose logs worker --tail 10
```
The worker log shows `Retry in 4s ...`: the jobs are not lost and not marked as done; they wait and retry with growing delays.
Now bring the database back:
```
docker compose start qdrant
```
```
docker compose logs worker --tail 20
```
Within a minute every job shows `succeeded`. Run `docker compose run --rm verify`: the same counts as before, nothing
duplicated, nothing missing.

### 11. Scale out
Create 120 small distinct documents and process them with one worker, then two, then four (do the first one now):
```
docker compose --profile queue up -d --scale worker=1 redis worker
```
```
docker compose run --rm enqueue /data/load --synthetic 120 --wait
```
Note the time. Then add workers and repeat with `--force` so the work is the same each time:
```
docker compose --profile queue up -d --scale worker=2 redis worker
```
```
docker compose run --rm enqueue /data/load --synthetic 120 --force --wait
```
The last line lists **which worker handled how many jobs**. Reference numbers from a 4-CPU Docker setup: 1 worker 71 s,
2 workers 37 s, 4 workers 23 s. Your numbers depend on the CPUs Docker may use (Settings, Resources); once workers outnumber your
CPUs, more workers stop helping. Each worker is capped to one CPU thread on purpose: without that cap a single worker already
uses every core and extra workers only compete with each other (we measured it get *slower*).

### 12. Kill the workers hard
```
docker compose run --rm enqueue /data/load --force
```
```
docker compose kill worker
```
```
docker compose --profile queue up -d --scale worker=2 redis worker
```
```
docker compose exec redis redis-cli llen celery
```
```
docker compose exec redis redis-cli hlen unacked
```
Run the last two a few times over three minutes. `llen celery` is the **waiting jobs**: it drains to 0 in about a minute.
`hlen unacked` is **jobs that were running when the worker died**: they stay for about 150 seconds (the lab's setting), then the queue
hands them to a worker again. Finally `docker compose run --rm verify` shows the same counts as before: **no lost jobs, no duplicates.**
(In production that wait is set to hours, because it must be longer than your slowest job.)

**What you just saw:** bad input is isolated, a dependency outage is retried, extra workers add throughput until the CPU is full,
and a crashed worker's jobs come back. That is why ingestion is built as idempotent jobs on a queue, not a script.

---

## Clean up (please do this)
```
docker compose --profile queue down -v --rmi local
```
This removes this lab's containers, its Qdrant data and queue, and the image it built. It only touches things named
`rag-ingest-docs`. Check:
```
docker compose ps -a
```
should list nothing. To delete the project folder, remove it like any folder.

## If something goes wrong
| symptom | fix |
|---|---|
| `Cannot connect to the Docker daemon` | start Docker Desktop and wait for "Engine running" |
| `port is already allocated` (6333) | stop whatever uses 6333 (another Qdrant?), or change the port in `docker-compose.yml` |
| build is very slow or fails downloading | retry; on slow wifi do it at home. A corporate proxy needs Docker proxy settings |
| `verify` says `NOT FOUND` | run step 4 first (and `docker compose up -d qdrant`) |
| a worker is killed with exit 137 | Docker has too little memory: raise it to 4 GB, or scale to fewer workers |
| a command waits for `qdrant` and it is stopped | `docker compose start qdrant` |
| a file in `data/load` or `data/raw` is not picked up | supported: PDF, DOCX, PPTX, XLSX, CSV, HTML, EML, MD, TXT |
