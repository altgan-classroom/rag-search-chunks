"""Parsers for non-PDF formats, registered with the parser registry in `parsers.py`.

Every parser returns the same thing as the PDF parser: per-page markdown (headings as `#`, tables as markdown tables) so
the structure-aware chunker, table handling and citations work identically for every format.

What "page" means (it is what a citation shows):
  PDF   page number                     DOCX  page break count when the file records it (else 1)
  PPTX  slide number                    XLSX  sheet index (the sheet name is the heading, so it is in the section path)
        (PPTX and XLSX set meta["hard_pages"]: the chunker never merges two slides or two sheets into one chunk)
  CSV   1 (rows are chunked as a table, header repeated)      HTML / EML / MD / TXT  1

Born-digital formats skip the OCR-noise quality gate and the 40-character minimum: a slide that says "Q3 revenue: $4.2M"
is real content, not a blank page.
"""
from __future__ import annotations

import csv
import datetime as dt
import email
import email.policy
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from .models import Page, ParsedDocument
from .parsers import Parser, _clean, _finish, parse_pdf_raw, register

MAX_ROWS = 5000  # rows kept per sheet / CSV; larger tables are truncated with a warning, never silently


# ------------------------------------------------------------------------------------------- helpers
def md_table(rows: list[list[str]]) -> str:
    """Markdown table from rows (first non-empty row = header). Ragged rows are padded; pipes/newlines escaped."""
    rows = [[re.sub(r"\s*\n\s*", " ", str(c)).replace("|", "\\|").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    while w > 1 and not any(len(r) >= w and r[w - 1] for r in rows):  # drop trailing empty columns
        w -= 1
    rows = [(r + [""] * w)[:w] for r in rows]
    head = [c or f"col{i + 1}" for i, c in enumerate(rows[0])]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * w) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.6g}"
    if isinstance(v, dt.datetime):
        return v.date().isoformat() if v.time() == dt.time(0) else v.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    return str(v)


def _doc(path: str, pages: list[Page], meta: dict, warnings: list[str], *, gate: bool = False) -> ParsedDocument:
    d = _finish(path, pages, meta, gate=gate)
    d.warnings = warnings + d.warnings
    return d


# ------------------------------------------------------------------------------------------- DOCX
class DocxParser:
    exts = (".docx",)
    cache_raw = False

    def parse(self, path: str, **_) -> ParsedDocument:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = docx.Document(path)
        pages: dict[int, list[str]] = {1: []}
        cur = 1
        for el in d.element.body.iterchildren():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = Paragraph(el, d)
                if el.xpath(".//w:lastRenderedPageBreak"):  # Word records where it paginated; absent in generated files
                    cur += 1
                    pages.setdefault(cur, [])
                txt = p.text.strip()
                style = (p.style.name or "") if p.style is not None else ""
                m = re.match(r"Heading (\d)", style)
                if txt and (m or style == "Title"):
                    pages[cur].append("#" * (int(m.group(1)) if m else 1) + " " + txt)
                elif txt and ("List" in style):
                    pages[cur].append("- " + txt)
                elif txt:
                    pages[cur].append(txt)
                if el.xpath('.//w:br[@w:type="page"]'):
                    cur += 1
                    pages.setdefault(cur, [])
            elif tag == "tbl":
                t = Table(el, d)
                rows = []
                for r in t.rows:
                    seen, cells = set(), []
                    for c in r.cells:
                        if id(c._tc) not in seen:  # merged cells repeat; keep one
                            seen.add(id(c._tc))
                            cells.append(c.text)
                    rows.append(cells)
                pages[cur].append(md_table(rows))
        out = [Page(n, _clean("\n\n".join(x))) for n, x in sorted(pages.items()) if any(x)]
        title = (d.core_properties.title or "").strip()
        return _doc(path, out, {"title": title, "page_count": len(out), "format": "docx"}, [])


# ------------------------------------------------------------------------------------------- PPTX
class PptxParser:
    exts = (".pptx",)
    cache_raw = False

    def _shape(self, sh, out: list[str], title_id) -> None:
        from pptx.enum.shapes import MSO_SHAPE_TYPE as T

        if sh.shape_type == T.GROUP:
            for s in sh.shapes:
                self._shape(s, out, title_id)
            return
        if getattr(sh, "shape_id", None) == title_id:
            return
        if getattr(sh, "has_table", False) and sh.has_table:
            out.append(md_table([[c.text for c in r.cells] for r in sh.table.rows]))
        elif getattr(sh, "has_chart", False) and sh.has_chart:
            out.append(self._chart(sh.chart))
        elif getattr(sh, "has_text_frame", False) and sh.has_text_frame:
            lines = []
            for p in sh.text_frame.paragraphs:
                t = "".join(r.text for r in p.runs).strip()
                if t:
                    lines.append(("  " * p.level + "- " + t) if p.level else t)
            if lines:
                out.append("\n".join(lines))
        elif sh.shape_type == T.PICTURE:
            alt = (sh._element.xpath(".//p:cNvPr/@descr") or [""])[0]
            if alt:
                out.append(f"[image: {alt}]")

    @staticmethod
    def _chart(ch) -> str:
        try:
            plot = ch.plots[0]
            cats = [str(c) for c in plot.categories]
            rows = [["", *[s.name or f"series {i + 1}" for i, s in enumerate(plot.series)]]]
            for i, c in enumerate(cats):
                rows.append([c, *[_cell(list(s.values)[i]) for s in plot.series]])
            title = ch.chart_title.text_frame.text if ch.has_title else "Chart"
            return f"Chart: {title}\n\n" + md_table(rows)
        except Exception:  # unusual chart types: say so instead of failing the whole deck
            return "[chart: data not extractable]"

    def parse(self, path: str, **_) -> ParsedDocument:
        from pptx import Presentation

        prs = Presentation(path)
        pages = []
        for i, slide in enumerate(prs.slides, 1):
            ts = slide.shapes.title
            title = (ts.text_frame.text.strip() if ts is not None and ts.has_text_frame else "") or f"Slide {i}"
            parts = [f"## {title}"]
            order = sorted(slide.shapes, key=lambda s: ((s.top or 0), (s.left or 0)))
            for sh in order:
                self._shape(sh, parts, ts.shape_id if ts is not None else None)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                n = slide.notes_slide.notes_text_frame.text.strip()
                if n:
                    parts.append("Speaker notes: " + n)
            pages.append(Page(i, _clean("\n\n".join(parts))))
        t = (prs.core_properties.title or "").strip()
        return _doc(path, pages, {"title": t, "page_count": len(pages), "format": "pptx", "hard_pages": True}, [])


# ------------------------------------------------------------------------------------------- XLSX / CSV
class XlsxParser:
    exts = (".xlsx", ".xlsm")
    cache_raw = False

    def parse(self, path: str, **_) -> ParsedDocument:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        pages, warns = [], []
        for idx, ws in enumerate(wb.worksheets, 1):
            if ws.sheet_state != "visible":
                continue
            rows = []
            for n, r in enumerate(ws.iter_rows(values_only=True)):
                if n >= MAX_ROWS:
                    warns.append(f"sheet {ws.title!r}: truncated to {MAX_ROWS} rows")
                    break
                rows.append([_cell(v) for v in r])
            body = md_table(rows)
            if body:
                pages.append(Page(idx, f"## Sheet: {ws.title}\n\n{body}"))
        wb.close()
        return _doc(path, pages, {"title": "", "page_count": len(pages), "format": "xlsx", "hard_pages": True}, warns)


class CsvParser:
    exts = (".csv", ".tsv")
    cache_raw = False

    def parse(self, path: str, **_) -> ParsedDocument:
        raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        if path.lower().endswith(".tsv"):
            delim = "\t"
        else:
            try:
                delim = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|").delimiter
            except csv.Error:
                delim = ","
        rows, warns = [], []
        for n, r in enumerate(csv.reader(raw.splitlines(), delimiter=delim)):
            if n >= MAX_ROWS:
                warns.append(f"truncated to {MAX_ROWS} rows")
                break
            rows.append(r)
        return _doc(path, [Page(1, md_table(rows))], {"page_count": 1, "format": "csv"}, warns)


# ------------------------------------------------------------------------------------------- HTML / EML
class _Md(HTMLParser):
    SKIP = {"script", "style", "head", "noscript", "nav", "footer", "svg", "template"}
    BLOCK = {"p", "div", "section", "article", "br", "ul", "ol", "blockquote", "pre", "main", "header"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.head: str | None = None

    def _w(self, s):
        (self.cell if self.cell is not None else self.out).append(s)

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._w("\n\n" + "#" * int(tag[1]) + " ")
            self.head = tag
        elif tag == "li":
            self._w("\n- ")
        elif tag in self.BLOCK:
            self._w("\n\n" if tag != "br" else "\n")
        elif tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if self.head == tag:
            self._w("\n\n")
            self.head = None
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.out.append("\n\n" + md_table(self.table) + "\n\n")
            self.table = None
        elif tag in self.BLOCK or tag == "li":
            self._w("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skip:
            self._w(data)


def html_to_markdown(raw: str) -> tuple[str, str]:
    p = _Md()
    p.feed(raw)
    p.close()
    text = "".join(p.out)
    text = re.sub(r"[ \t]+", " ", text)
    return _clean(re.sub(r"\n[ \t]+", "\n", text)), p.title.strip()


class HtmlParser:
    exts = (".html", ".htm")
    cache_raw = False

    def parse(self, path: str, **_) -> ParsedDocument:
        md, title = html_to_markdown(Path(path).read_text(errors="replace"))
        return _doc(path, [Page(1, md)], {"title": title, "page_count": 1, "format": "html"}, [], gate=True)


class EmlParser:
    exts = (".eml",)
    cache_raw = False

    def parse(self, path: str, **_) -> ParsedDocument:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part is not None else ""
        if part is not None and part.get_content_type() == "text/html":
            body, _t = html_to_markdown(body)
        subject = str(msg.get("subject", "")).strip()
        head = [f"# {subject}" if subject else "# (no subject)"]
        head += [f"{h.capitalize()}: {msg[h]}" for h in ("from", "to", "cc", "date") if msg[h]]
        atts = [a.get_filename() or "attachment" for a in msg.iter_attachments()]
        warns = [f"{len(atts)} attachment(s) not indexed: {', '.join(atts)}"] if atts else []
        return _doc(path, [Page(1, _clean("\n".join(head) + "\n\n" + body))],
                    {"title": subject, "page_count": 1, "format": "eml"}, warns, gate=True)


# ------------------------------------------------------------------------------------------- images (OCR)
class ImageParser:
    """Scans and screenshots. Needs an OCR engine (Tesseract or RapidOCR); without one we say so instead of
    silently indexing nothing."""
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    cache_raw = False

    def parse(self, path: str, **_) -> ParsedDocument:
        import pymupdf

        with pymupdf.open(path) as im:
            pdf = im.convert_to_pdf()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / (Path(path).stem + ".pdf")
            tmp.write_bytes(pdf)
            pages, meta = parse_pdf_raw(str(tmp), ocr=True)
        if not any(p.markdown.strip() for p in pages):
            raise ValueError("image has no readable text: OCR is required (install Tesseract: "
                             "`brew install tesseract` / `winget install UB-Mannheim.TesseractOCR`) or the image has no text")
        for p in pages:
            p.ocr = True
        return _doc(path, pages, {**meta, "format": "image"}, ["text recovered by OCR"], gate=True)


for _p in (DocxParser(), PptxParser(), XlsxParser(), CsvParser(), HtmlParser(), EmlParser(), ImageParser()):
    register(_p)  # type: ignore[arg-type]

__all__ = ["md_table", "html_to_markdown", "Parser"]
