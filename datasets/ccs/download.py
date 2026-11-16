"""Fetch the pinned CCS corpus subset from the Hugging Face dataset and verify sha256 (Apache-2.0)."""
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = json.loads((HERE / "sources.json").read_text())
out = HERE.parents[1] / "data" / "raw" / "ccs"
out.mkdir(parents=True, exist_ok=True)
bad = 0
for m in src["files"]:
    p = out / m["file"]
    if not p.exists():
        url = f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/" + urllib.parse.quote(m["file"])
        print("download", m["file"])
        urllib.request.urlretrieve(url, p)
    if hashlib.sha256(p.read_bytes()).hexdigest() != m["sha256"]:
        print("SHA MISMATCH", m["file"])
        bad += 1
sys.exit(bad)
