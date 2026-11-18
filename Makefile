.PHONY: install test lint parse ingest eval ladder loo sweep serve data up down
PY ?= .venv/bin/python
export PYTHONPATH := src

install:        ## create venv + install (no torch)
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[ingest,dev]"
test:           ## fast, keyless, offline
	$(PY) -m pytest tests -q -m "not integration"
lint:
	.venv/bin/ruff check src tests
data:           ## real corpus (pinned, sha256-verified) + synthetic corpus/gold
	$(PY) datasets/ccs/download.py && $(PY) datasets/ccs/build_gold.py && $(PY) -m orca_rag.datasets.synthetic --out datasets/synthetic
parse:          ## OCR/layout once, in parallel, into the parse cache
	$(PY) -m orca_rag.cli parse data/raw/ccs --workers 4
eval:           ## baseline vs advanced on a dataset:  make eval D=datasets/synthetic
	$(PY) -m orca_rag.cli eval $(D) --profiles baseline,advanced,full
ladder:         ## cumulative ablation ladder
	$(PY) -m orca_rag.cli ablate $(D) --mode ladder --ci
loo:            ## leave-one-out from the full recipe
	$(PY) -m orca_rag.cli ablate $(D) --mode loo --ci
serve:          ## local API, embedded Qdrant, anonymous (dev only)
	ORCA_AUTH__ALLOW_ANONYMOUS=true $(PY) -m uvicorn orca_rag.api.app:create_app --factory --port 8080
up:
	docker compose -f deployment/docker-compose.yml --env-file deployment/.env up -d --build
down:
	docker compose -f deployment/docker-compose.yml down
