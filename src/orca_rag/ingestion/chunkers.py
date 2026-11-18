"""Chunking strategies. Selected by `profile.chunking.strategy`; all return `list[Chunk]`.

fixed_words  - the baseline template's recipe (word windows by character budget, newlines flattened,
               page = page of first word). Kept so the comparison is honest.
structure    - heading/paragraph/table aware packing with token budgets, sentence-level overlap,
               tables kept whole (or split by rows with the header repeated), exact page ranges,
               heading-path metadata, and optional parent-child (index small, return large).
semantic     - sentence embeddings; split where adjacent-window similarity drops (topic shifts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config.settings import ChunkingCfg
from .models import Chunk, ParsedDocument

_TOK = re.compile(r"\w+|[^\w\s]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_HEAD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def count_tokens(s: str) -> int:
    """Cheap, deterministic, offline token estimate (words + punctuation). Not a model tokenizer; chunk
    sizes are *relative* budgets, and using one tokenizer everywhere keeps sweeps comparable."""
    return len(_TOK.findall(s))


# =============================================================================== fixed_words (baseline)
def chunk_fixed_words(doc: ParsedDocument, cfg: ChunkingCfg) -> list[Chunk]:
    out: list[Chunk] = []
    cur: list[str] = []
    cur_pages: list[int] = []
    size = 0
    words_in: list[tuple[str, int]] = [(w, p.number) for p in doc.pages if not p.skipped for w in p.markdown.split()]

    def flush():
        if cur:
            out.append(Chunk(ordinal=len(out), text=" ".join(cur), page_start=cur_pages[0], page_end=cur_pages[0],
                             tokens=count_tokens(" ".join(cur))))

    for w, pg in words_in:
        if size + len(w) + 1 > cfg.size and cur:
            flush()
            keep: list[str] = []
            keep_pages: list[int] = []
            acc = 0
            for ww, pp in zip(reversed(cur), reversed(cur_pages)):
                acc += len(ww) + 1
                keep.insert(0, ww)
                keep_pages.insert(0, pp)
                if acc >= cfg.overlap:
                    break
            cur, cur_pages, size = keep, keep_pages, acc
        cur.append(w)
        cur_pages.append(pg)
        size += len(w) + 1
    flush()
    return out


# =============================================================================== structure-aware
@dataclass
class Block:
    text: str
    page: int
    kind: str  # heading | para | table | list
    section: str
    tokens: int


def _blocks(doc: ParsedDocument) -> list[Block]:
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    for pg in doc.pages:
        if pg.skipped:
            continue
        lines = pg.markdown.split("\n")
        buf: list[str] = []
        kind = "para"

        def emit():
            nonlocal buf, kind
            txt = "\n".join(buf).strip()
            if txt:
                sec = " > ".join(t for _, t in stack)
                blocks.append(Block(txt, pg.number, kind, sec, count_tokens(txt)))
            buf, kind = [], "para"

        for ln in lines:
            m = _HEAD.match(ln)
            if m:
                emit()
                lvl, title = len(m.group(1)), re.sub(r"[*_`]", "", m.group(2)).strip()
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, title))
                blocks.append(Block(title, pg.number, "heading", " > ".join(t for _, t in stack), count_tokens(title)))
                continue
            is_tbl = ln.lstrip().startswith("|")
            if not ln.strip():
                emit()
                continue
            if is_tbl and kind != "table":
                emit()
                kind = "table"
            elif not is_tbl and kind == "table":
                emit()
            buf.append(ln)
        emit()
    return blocks


def _split_table(b: Block, budget: int) -> list[Block]:
    rows = b.text.split("\n")
    header = rows[:2] if len(rows) > 2 and set(rows[1].replace("|", "").strip()) <= set("-: ") else rows[:1]
    body = rows[len(header):]
    out, cur, cur_t = [], [], count_tokens("\n".join(header))
    for r in body:
        rt = count_tokens(r)
        if cur and cur_t + rt > budget:
            out.append("\n".join(header + cur))
            cur, cur_t = [], count_tokens("\n".join(header))
        cur.append(r)
        cur_t += rt
    if cur:
        out.append("\n".join(header + cur))
    return [Block(t, b.page, "table", b.section, count_tokens(t)) for t in out]


def _split_para(b: Block, budget: int) -> list[Block]:
    sents = _SENT.split(b.text)
    out, cur, cur_t = [], [], 0
    for s in sents:
        st = count_tokens(s)
        if st > budget:  # pathological sentence: hard split on words
            words = s.split()
            for i in range(0, len(words), budget):
                out.append(" ".join(words[i : i + budget]))
            continue
        if cur and cur_t + st > budget:
            out.append(" ".join(cur))
            cur, cur_t = [], 0
        cur.append(s)
        cur_t += st
    if cur:
        out.append(" ".join(cur))
    return [Block(t, b.page, b.kind, b.section, count_tokens(t)) for t in out]


def _pack(blocks: list[Block], size: int, overlap: int, keep_tables: bool,
          hard_pages: bool = False) -> list[tuple[list[Block], bool]]:
    """Greedy pack blocks into groups <= size tokens. A heading is never left stranded at a group's end.
    `hard_pages`: never let a group span two pages (slides, sheets), so a citation names exactly one slide or sheet."""
    norm: list[Block] = []
    for b in blocks:
        if b.tokens <= size:
            norm.append(b)
        elif b.kind == "table":
            norm.extend(_split_table(b, size) if keep_tables else _split_para(b, size))
        else:
            norm.extend(_split_para(b, size))
    groups: list[tuple[list[Block], bool]] = []
    cur: list[Block] = []
    cur_t = 0
    for b in norm:
        if hard_pages and cur and b.page != cur[-1].page:
            groups.append((cur, any(x.kind == "table" for x in cur)))
            cur, cur_t = [], 0
        if cur and cur_t + b.tokens > size:
            while cur and cur[-1].kind == "heading":  # carry dangling heading forward
                carried = [cur.pop()]
                break
            else:
                carried = []
            if cur:
                groups.append((cur, any(x.kind == "table" for x in cur)))
            tail: list[Block] = []
            if overlap and cur:
                t = 0
                for x in reversed(cur):
                    if x.kind == "table" or t + x.tokens > overlap:
                        break
                    tail.insert(0, x)
                    t += x.tokens
            cur = tail + carried
            cur_t = sum(x.tokens for x in cur)
        cur.append(b)
        cur_t += b.tokens
        if b.kind == "table" and not keep_tables:
            pass
    if cur:
        groups.append((cur, any(x.kind == "table" for x in cur)))
    return groups


def _to_chunk(ordinal: int, blocks: list[Block], is_table: bool, parent: int | None = None) -> Chunk:
    text = "\n\n".join(b.text for b in blocks)
    # section of the first *content* block (a leading heading already names its own section)
    sec = next((b.section for b in blocks if b.kind != "heading"), blocks[0].section)
    return Chunk(ordinal=ordinal, text=text, page_start=min(b.page for b in blocks), page_end=max(b.page for b in blocks),
                 section=sec, is_table=is_table, parent_ordinal=parent, tokens=count_tokens(text))


def chunk_structure(doc: ParsedDocument, cfg: ChunkingCfg) -> tuple[list[Chunk], list[Chunk]]:
    """Returns (children, parents). Without parent_child, parents is empty and children are the final chunks."""
    blocks = _blocks(doc)
    if not blocks:
        return [], []
    hard = bool(doc.meta.get("hard_pages"))
    if not cfg.parent_child:
        return [_to_chunk(i, g, t) for i, (g, t) in enumerate(_pack(blocks, cfg.size, cfg.overlap, cfg.keep_tables_whole, hard))], []
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for pi, (pg, pt) in enumerate(_pack(blocks, cfg.parent_size, 0, cfg.keep_tables_whole, hard)):
        parents.append(_to_chunk(pi, pg, pt))
        for g, t in _pack(pg, cfg.size, cfg.overlap, cfg.keep_tables_whole, hard):
            children.append(_to_chunk(len(children), g, t, parent=pi))
    return children, parents


# =============================================================================== semantic
def chunk_semantic(doc: ParsedDocument, cfg: ChunkingCfg, embedder, percentile: float = 85.0) -> list[Chunk]:
    import numpy as np

    sents: list[tuple[str, int, str]] = []
    for b in _blocks(doc):
        if b.kind == "heading":
            continue
        for s in ([b.text] if b.kind == "table" else _SENT.split(b.text)):
            if s.strip():
                sents.append((s.strip(), b.page, b.section))
    if not sents:
        return []
    vecs = np.array(embedder.embed_docs([s for s, _, _ in sents]), dtype=np.float32)
    w = np.array([vecs[max(0, i - 1) : i + 2].mean(axis=0) for i in range(len(vecs))])
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-9
    dist = 1 - (w[:-1] * w[1:]).sum(axis=1) if len(w) > 1 else np.array([])
    cut = np.percentile(dist, percentile) if len(dist) else 1.0
    out: list[Chunk] = []
    cur: list[tuple[str, int, str]] = []
    t = 0
    for i, s in enumerate(sents):
        st = count_tokens(s[0])
        boundary = i > 0 and ((dist[i - 1] >= cut and t >= cfg.size // 3) or t + st > cfg.size)
        if cur and boundary:
            out.append(_sem_chunk(len(out), cur))
            cur, t = [], 0
        cur.append(s)
        t += st
    if cur:
        out.append(_sem_chunk(len(out), cur))
    return out


def _sem_chunk(i: int, cur) -> Chunk:
    text = " ".join(s for s, _, _ in cur)
    return Chunk(ordinal=i, text=text, page_start=min(p for _, p, _ in cur), page_end=max(p for _, p, _ in cur),
                 section=cur[0][2], is_table=all(s.lstrip().startswith("|") for s, _, _ in cur), tokens=count_tokens(text))


def chunk_document(doc: ParsedDocument, cfg: ChunkingCfg, embedder=None) -> tuple[list[Chunk], list[Chunk]]:
    if cfg.strategy == "fixed_words":
        return chunk_fixed_words(doc, cfg), []
    if cfg.strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        return chunk_semantic(doc, cfg, embedder), []
    return chunk_structure(doc, cfg)
