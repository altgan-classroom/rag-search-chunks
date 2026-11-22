#!/bin/sh
# Cleans up the RAG ingestion lab. Removes ONLY what this lab created (Docker project "rag-ingested-docs"):
#   containers, the database data and queue (volumes), the image built for the lab (about 1 GB), and files generated in data/load.
# It does not touch any other Docker project, image or volume, and it never deletes your own files.
# Run from anywhere:   sh cleanup.sh        (add -y to skip the question)
cd "$(dirname "$0")" || exit 1

echo "This will remove everything the lab created (project: rag-ingested-docs):"
echo "  - containers: qdrant, redis, workers and one-off containers"
echo "  - the database data and the queue (Docker volumes)"
echo "  - the image built for the lab (about 1 GB; you will need 'docker compose build' to run the lab again)"
echo "  - generated files in data/load (README.txt stays)"
if [ "$1" != "-y" ]; then
  printf "Continue? [y/N] "
  read ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "Cancelled. Nothing was removed."; exit 0 ;; esac
fi

docker compose --profile queue down -v --rmi local --remove-orphans
find data/load -type f ! -name README.txt -delete 2>/dev/null

echo
echo "Checking what is left for this project (nothing should be listed below):"
docker ps -a --filter "label=com.docker.compose.project=rag-ingested-docs" --format "  container: {{.Names}} ({{.Status}})"
docker volume ls --filter "label=com.docker.compose.project=rag-ingested-docs" --format "  volume: {{.Name}}"
docker images rag-ingested-docs --format "  image: {{.Repository}}:{{.Tag}}"
echo "Done. (Optional, frees more space but affects ALL Docker projects: docker builder prune)"
