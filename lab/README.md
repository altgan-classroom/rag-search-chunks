# RAG lab: search and chunks (Sessions 1 and 2)

A small, self-contained lab. You need **Docker Desktop** and nothing else (no API key, no Python install). Everything runs in containers.

```
git clone https://github.com/altgan-classroom/rag-search-chunks.git
cd rag-search-chunks/lab
docker compose build            # once: 2 to 5 minutes (code + two small models are baked into the image)
docker compose up -d qdrant     # the vector database, dashboard at http://localhost:6333/dashboard
docker compose run --rm s2 ingest      # load six invented insurance policies
docker compose run --rm s2 ask q3      # same question, ranked by meaning, by keywords, and by both
```

The step-by-step guide, with what each command does and why, is in the **Visual Lab** page (tab "6 · Hands-on"), shared in the session chat.

## What is in this folder
| Path | What it is |
|---|---|
| `docker-compose.yml`, `Dockerfile` | the lab environment (run every command from this folder) |
| `data/session2/` | six invented insurance policies (the Session 2 corpus) |
| `data/session2_extra/` | eight more, for the optional "bigger corpus" step (`--corpus plus`) |
| `data/raw/`, `challenge/` | the five sample files and a broken PDF from the Session 1 lab |
| `configs/` | chunking recipes (`s2_arena`, `s2_fixed`, `s2_tiny`, `s2_parent`) and the Session 1 recipe (`advanced`) |
| `scripts/s2.py` | the helper behind `s2 ingest`, `s2 ask` and `s2 collections` |
| `cleanup.sh`, `cleanup.bat` | remove everything the lab created from Docker |

Session 1 commands (`docker compose run --rm ingest`, `verify`, and so on) work from this folder too. The Docker project is still called `rag-ingested-docs` on purpose, so if you did Session 1 you reuse the same database and image.

The code in `../src` is the same RAG pipeline the course builds on; the lab image is built from it.
