"""Document parsing -> per-page markdown with quality signals.

PDF: pymupdf4llm gives layout-aware markdown (headings, real markdown tables, reading order) and OCRs
text-less pages with Tesseract when available. We then:
  * strip running headers/footers that repeat across pages (they pollute embeddings and BM25),
  * score each page's text quality and exclude garbage OCR instead of indexing it (baseline REVIEW I12),
  * never write anything back to the source store (baseline REVIEW I10).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from .models import Page, ParsedDocument, sha256_file

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[A-Za-z]{3,}")
_TOK = re.compile(r"\S+")
SUPPORTED = {".pdf", ".md", ".txt", ".html", ".htm"}


_NUM = re.compile(r"\$?-?\d[\d,.]*%?")
_ID = re.compile(r"[A-Za-z]+[-_]?\d[\w\-]*|\d+[-/]\d[\d\-/,.]*|[A-Za-z]{2,}-[A-Za-z0-9\-]+")
_TWO = set("an as at be by do go he if in is it me my no of on or so to up us we ft in mm cm km ml mg kg ph id ok vs tv"
           " co ca ct al ar az ga il in ks ky la md mi mn mo ms mt nc nd ne nh nj nm nv ny oh ok or pa ri sc sd tn tx ut va vt wa wi wv wy".split())


def _good(t: str) -> bool:
    t = t.strip(".,;:()[]\"'")
    if not t:
        return False
    return bool(_NUM.fullmatch(t) or re.fullmatch(r"[A-Za-z]{3,}", t) or t.lower() in _TWO or _ID.fullmatch(t))


def text_quality(md: str) -> float:
    """Share of tokens that look like real content: words, numbers/currency/percentages, identifiers (MB-107).
    Table scaffolding is ignored. OCR garbage (stray symbols, invented 2-letter 'words') scores near 0, while a
    numeric table scores high; counting only alphabetic words wrongly rejected production tables."""
    md = re.sub(r"\|[-: |]+\|?|[|*_#`]", " ", md)
    toks = _TOK.findall(md)
    return (sum(map(_good, toks)) / len(toks)) if toks else 0.0


def _norm_line(line: str) -> str:
    return re.sub(r"\d+|[ivxlc]+\b", "#", line.strip().lower())


def strip_running_lines(pages: list[Page], min_pages: int = 4, frac: float = 0.35) -> int:
    """Remove lines that recur (modulo numbers) on >= `frac` of pages. Returns lines removed."""
    live = [p for p in pages if p.markdown.strip()]
    if len(live) < min_pages:
        return 0
    freq: Counter[str] = Counter()
    for p in live:
        seen = {_norm_line(l) for l in p.markdown.splitlines() if 3 < len(l.strip()) < 120}
        freq.update(seen)
    boiler = {k for k, c in freq.items() if c / len(live) >= frac and not k.startswith(("|", "#"))}
    removed = 0
    for p in live:
        keep = []
        for l in p.markdown.splitlines():
            if _norm_line(l) in boiler and 3 < len(l.strip()) < 120:
                removed += 1
                continue
            keep.append(l)
        p.markdown = "\n".join(keep)
    return removed


def _clean(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = md.replace("<br>", " ")
    md = re.sub(r"\*\*|__", "", md)  # bold markers add noise to tokens/BM25
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def parse_pdf_raw(path: str, *, ocr: bool = True, max_pages: int | None = None,
                  page_numbers: list[int] | None = None) -> tuple[list[Page], dict]:
    """Expensive step (layout analysis + OCR). Output is cacheable; gating/cleanup is applied afterwards by `_finish`."""
    import pymupdf
    import pymupdf4llm

    with pymupdf.open(path) as d:
        n = d.page_count
        meta = {"title": (d.metadata or {}).get("title", ""), "page_count": n, "format": (d.metadata or {}).get("format", "")}
    idx = page_numbers or list(range(min(n, max_pages) if max_pages else n))
    kwargs = {} if ocr else {"use_ocr": False}
    chunks = pymupdf4llm.to_markdown(path, page_chunks=True, pages=idx, show_progress=False, **kwargs)
    pages = [Page(number=int(c["metadata"].get("page_number", i + 1)), markdown=_clean(c["text"]))
             for i, c in zip(idx, chunks)]
    return pages, meta


def parse_pdf(path: str, **kw) -> ParsedDocument:
    pages, meta = parse_pdf_raw(path, **kw)
    return _finish(path, pages, meta)


def _finish(path: str, pages: list[Page], meta: dict) -> ParsedDocument:
    warnings: list[str] = []
    removed = strip_running_lines(pages)
    if removed:
        warnings.append(f"removed {removed} running header/footer lines")
    for p in pages:
        p.markdown = _clean(p.markdown)
        p.quality = text_quality(p.markdown)
        if len(p.markdown) < 40 or p.quality < 0.35:
            p.skipped = True
    skipped = sum(p.skipped for p in pages)
    if skipped:
        warnings.append(f"excluded {skipped}/{len(pages)} low-quality/empty pages")
    pth = Path(path)
    return ParsedDocument(name=pth.stem, source_key=pth.name, sha256=sha256_file(path), pages=pages,
                          meta=meta, warnings=warnings)


def parse_text(path: str) -> ParsedDocument:
    raw = Path(path).read_text(errors="replace")
    if path.lower().endswith((".html", ".htm")):
        raw = re.sub(r"<(script|style).*?</\1>", "", raw, flags=re.S | re.I)
        raw = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "\n\n" + "#" * int(m.group(1)) + " " + m.group(2) + "\n\n", raw, flags=re.S | re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
    return _finish(path, [Page(1, _clean(raw))], {"page_count": 1})


def parse_file(path: str, **kw) -> ParsedDocument:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED:  # reject up front instead of failing + retrying inside a worker (baseline I11)
        raise ValueError(f"unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")
    return parse_pdf(path, **kw) if ext == ".pdf" else parse_text(path)


RAW_CACHE_VERSION = "r1"


def parse_cached(path: str, cache_dir: str | Path | None, **kw) -> ParsedDocument:
    """Cache the *raw* layout/OCR output per (content sha, options). Header/footer stripping, quality gating and
    cleanup are cheap and re-applied on every load, so improving them never forces a re-OCR, and re-chunking or
    re-embedding for a new recipe skips OCR entirely (the dominant cost on scanned PDFs)."""
    import hashlib
    import json

    if cache_dir is None or Path(path).suffix.lower() != ".pdf":
        return parse_file(path, **kw)
    sha = sha256_file(path)
    opt = hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:8]
    f = Path(cache_dir) / f"{sha[:16]}_{RAW_CACHE_VERSION}_{opt}.json"
    if f.exists():
        d = json.loads(f.read_text())
        pages = [Page(number=p["number"], markdown=p["markdown"]) for p in d["pages"]]
        meta = d["meta"]
    else:
        pages, meta = parse_pdf_raw(path, **kw)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"meta": meta, "pages": [{"number": p.number, "markdown": p.markdown} for p in pages]}))
    return _finish(path, pages, meta)
