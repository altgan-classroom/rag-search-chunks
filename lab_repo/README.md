# RAG ingestion lab: put 5 files into a vector database, then look inside it

**Time:** about 15 minutes for Part 1 (in the session); Part 2 is a take-home task (30 to 45 minutes).
**What you will see:** documents of five different types become searchable vectors in Qdrant, and you check the result
three ways: container logs, a verification command, and the Qdrant dashboard. Then you break things on purpose to see
what the design does about it.

## Before you start
* **Docker Desktop is running** (the whale icon says "Engine running") and `docker compose version` works.
* **Git** is installed.
* Docker has **4 GB of memory or more** (Docker Desktop, Settings, Resources).
* Internet is needed for step 2 only (building the image: about 1 GB, 2 to 5 minutes or more on slow wifi). **Start step 2 first** and read on while it builds.

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
profile=advanced collection=rag_461716c917 total=1s stats={'points': 25, 'status': 'green'}
```
Notice: the slide deck became **4 chunks** (one per slide) and the workbook **3** (one per sheet), so an answer can cite
"slide 3". A line `Warning: RapidOCR is missing` is harmless (it is about scanned pages; there are none here).

### 5. Verify what is in the vector database
```
docker compose run --rm verify
```
Expected:
```
collection  rag_461716c917  status=green  points_count=25
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

## Part 2: break it on purpose (take-home task)

Part 2 is for you to do after the session, at your own pace (30 to 45 minutes). Ingestion runs through a **queue** (Redis) and
**workers**, and you cause four failures yourself and check what the system does:

1. a **corrupt file** next to good ones;
2. the **database going down** while jobs are waiting;
3. **more workers** (1, 2, then 4): how much faster?
4. **killing the workers** in the middle of work.

Everything you need is in **`docs/TAKE_HOME_PART2.pdf`**: the commands, what you should see, how to verify each result,
a troubleshooting table and reflection questions. Predict first, then run.

---

## Clean up (after Part 2, or now if you are skipping it)
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
| build is very slow or fails downloading | retry; on slow wifi it can take 10 minutes or more, so keep it running while you read on, or ask for the saved image file. A corporate proxy needs Docker proxy settings |
| `verify` says `NOT FOUND` | run step 4 first (and `docker compose up -d qdrant`) |
| a worker is killed with exit 137 | Docker has too little memory: raise it to 4 GB, or scale to fewer workers |
| a command waits for `qdrant` and it is stopped | `docker compose start qdrant` |
| a file in `data/load` or `data/raw` is not picked up | supported: PDF, DOCX, PPTX, XLSX, CSV, HTML, EML, MD, TXT |
