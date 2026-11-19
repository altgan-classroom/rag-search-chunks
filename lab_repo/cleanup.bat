@echo off
rem Cleans up the RAG ingestion lab. Removes ONLY what this lab created (Docker project "rag-ingested-docs"):
rem containers, the database data and queue (volumes), the image built for the lab (about 1 GB), and files generated in data\load.
rem It does not touch any other Docker project, image or volume, and it never deletes your own files.
rem Run:  double-click cleanup.bat, or in a terminal:  cleanup.bat        (add -y to skip the question)
cd /d "%~dp0"

echo This will remove everything the lab created (project: rag-ingested-docs):
echo   - containers: qdrant, redis, workers and one-off containers
echo   - the database data and the queue (Docker volumes)
echo   - the image built for the lab (about 1 GB; you will need "docker compose build" to run the lab again)
echo   - generated files in data\load (README.txt stays)
if /i "%~1"=="-y" goto run
set /p ans=Continue? [y/N] 
if /i "%ans%"=="y" goto run
if /i "%ans%"=="yes" goto run
echo Cancelled. Nothing was removed.
goto end

:run
docker compose --profile queue down -v --rmi local --remove-orphans
for %%f in (data\load\*) do if /i not "%%~nxf"=="README.txt" del /q "%%f"
echo.
echo Checking what is left for this project (nothing should be listed below):
docker ps -a --filter "label=com.docker.compose.project=rag-ingested-docs" --format "  container: {{.Names}} ({{.Status}})"
docker volume ls --filter "label=com.docker.compose.project=rag-ingested-docs" --format "  volume: {{.Name}}"
docker images rag-ingested-docs --format "  image: {{.Repository}}:{{.Tag}}"
echo Done. (Optional, frees more space but affects ALL Docker projects: docker builder prune)

:end
pause
