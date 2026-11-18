"""Mixed-format corpus with facts planted in known places, plus gold questions that name the expected document AND page.

  python -m orca_rag.datasets.formats --out datasets/formats

One fictional company ("Helio Robotics"), eleven files, one format each. Every file holds several facts so retrieval has to
pick the right passage, and the question for each format targets a place a naive text dump would lose: a table cell in a
DOCX, slide 3 and the speaker notes of a PPTX, the second sheet of an XLSX, a CSV row, an HTML table behind navigation
noise, an e-mail body, text inside an image (OCR), and so on. `status="synthetic"`: facts are invented, so the corpus is
a *correctness* test of the format pipeline, not a quality benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ..eval.schema import GoldQuestion, GoldRef, GoldSet

KINDS = {  # file -> (format label, needs OCR)
    "Employee_Handbook.docx": "docx", "Q3_Board_Update.pptx": "pptx", "Sales_Pipeline_2026.xlsx": "xlsx",
    "Supplier_Contacts.csv": "csv", "Release_Notes.html": "html", "Incident_4471.eml": "eml",
    "Safety_Manual.pdf": "pdf", "Architecture_Notes.md": "md", "Onboarding.txt": "txt",
    "Whiteboard_Photo.png": "image",
}


def _docx(p: Path):
    import docx
    d = docx.Document()
    d.add_heading("Helio Robotics Employee Handbook", 0)
    d.add_heading("Leave policy", 1)
    d.add_paragraph("Full-time employees accrue 1.75 days of paid leave per month. Unused leave up to 12 days carries over "
                    "to the next calendar year; anything above that expires on 31 March.")
    d.add_paragraph("Parental leave is 26 weeks at full pay for the primary caregiver.")
    d.add_heading("Travel and expenses", 1)
    d.add_paragraph("Claims must be filed within 30 days of travel. Daily meal limits are shown below.")
    t = d.add_table(rows=1, cols=3)
    for c, h in zip(t.rows[0].cells, ["Trip type", "Meal limit per day (USD)", "Approval needed"]):
        c.text = h
    for row in [("Domestic", "45", "No"), ("International", "70", "No"), ("Client hosting", "120", "Manager")]:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = v
    d.add_heading("Remote work", 1)
    d.add_paragraph("Employees may work remotely up to 3 days per week with manager agreement.")
    d.save(p)


def _pptx(p: Path):
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Q3 Board Update"
    s.placeholders[1].text = "Helio Robotics, October 2026"
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Highlights"
    s.placeholders[1].text = "Customer count reached 212\nChurn fell to 2.1 percent\nHired 14 engineers"
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Revenue by region"
    rows = [("Region", "Q3 revenue (USD M)", "YoY growth"), ("EMEA", "4.2", "12%"), ("APAC", "3.1", "27%"), ("Americas", "6.8", "9%")]
    tb = s.shapes.add_table(4, 3, Inches(0.5), Inches(1.6), Inches(8), Inches(2)).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            tb.cell(i, j).text = v
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Roadmap"
    cd = CategoryChartData()
    cd.categories = ["Alpha", "Beta", "GA"]
    cd.add_series("Planned units", (120, 340, 900))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1.6), Inches(7), Inches(3.5), cd)
    s.notes_slide.notes_text_frame.text = "Presenter: the Helio X2 general availability date is 14 March 2027; do not announce earlier."
    prs.save(p)


def _xlsx(p: Path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Metric", "Value"])
    ws.append(["Open deals", 5])
    ws.append(["Weighted pipeline (USD)", 1830000])
    ws2 = wb.create_sheet("Pipeline")
    ws2.append(["Deal", "Customer", "Value (USD)", "Stage", "Close date"])
    for r in [("Falcon", "Arrowline Logistics", 480000, "Negotiation", "2026-12-15"),
              ("Heron", "Bluepeak Foods", 215000, "Proposal", "2027-01-20"),
              ("Osprey", "Cinder Mining", 760000, "Verbal commit", "2026-11-30"),
              ("Kestrel", "Dunmore Retail", 95000, "Discovery", "2027-02-28"),
              ("Merlin", "Eastgate Energy", 280000, "Proposal", "2027-01-31")]:
        ws2.append(list(r))
    ws3 = wb.create_sheet("Archive")
    ws3.append(["Deal", "Outcome"])
    ws3.append(["Condor", "Lost to incumbent"])
    wb.save(p)


def _csv(p: Path):
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Supplier", "Contact", "Phone", "Lead time (days)", "Part family"])
        for r in [("Norden Fasteners", "Ingrid Solberg", "+47 555 0142", 21, "bolts"), ("Kiyo Precision", "Haruto Mori", "+81 555 0199", 35, "gears"),
                  ("Tamsin Plastics", "Wen Zhao", "+44 555 0118", 14, "housings"), ("Ardent Cable", "Lucia Romero", "+34 555 0163", 28, "harness")]:
            w.writerow(r)


def _html(p: Path):
    p.write_text("""<!doctype html><html><head><title>Helio Platform Release Notes</title><style>.x{color:red}</style>
<script>var tracking = "do-not-index-this";</script></head><body>
<nav><a href="/">Home</a> <a href="/docs">Docs</a> <a href="/pricing">Pricing</a></nav>
<h1>Release notes</h1><p>Summary of platform releases.</p>
<h2>Versions</h2>
<table><tr><th>Version</th><th>Date</th><th>Highlights</th></tr>
<tr><td>3.4</td><td>2026-06-02</td><td>Dark mode for the console</td></tr>
<tr><td>3.5</td><td>2026-08-19</td><td>Batch export API (up to 10,000 rows per call)</td></tr>
<tr><td>3.6</td><td>2026-10-07</td><td>Single sign-on with SAML</td></tr></table>
<h2>Known issues</h2><ul><li>Exports over 50 MB may time out</li><li>SAML logout is not propagated to the mobile app</li></ul>
<footer>Copyright Helio Robotics</footer></body></html>""")


def _eml(p: Path):
    p.write_text("""From: Maya Patel <maya.patel@helio.example>
To: platform-oncall@helio.example
Date: Tue, 21 Oct 2026 09:14:00 +0000
Subject: Incident 4471 post-mortem
Content-Type: text/plain; charset=utf-8

Team,

Incident 4471 caused 47 minutes of failed logins on 20 October. The root cause was an expired TLS certificate on gateway-7.
Monitoring did not page because the certificate-expiry check only covered gateways 1 to 6.

Actions: add gateway-7 to the expiry check (owner: Tomas), and move to automatic renewal by the end of November.

Maya
""")


def _pdf(p: Path):
    import pymupdf
    d = pymupdf.open()
    pg = d.new_page()
    y = 72
    for line in ["Helio Warehouse Safety Manual", "", "1. Forklift operation",
                 "The maximum rated load of the Model F-30 forklift is 2,500 kg.",
                 "Operators must hold a valid certificate renewed every 24 months.", "",
                 "2. Aisle rules", "Pedestrians keep to the marked green walkway at all times.",
                 "Speed limit inside the warehouse is 8 km/h."]:
        pg.insert_text((72, y), line, fontsize=11)
        y += 18
    d.save(p)
    d.close()


def _md(p: Path):
    p.write_text("""# Architecture notes

## Services
The ingest service writes documents to object storage and publishes a job to the queue.

## Data retention
Audit logs are retained for 400 days and then deleted by the nightly compaction job.
Customer exports are retained for 30 days.
""")


def _txt(p: Path):
    p.write_text("""Onboarding checklist for new hires

Day 1: collect laptop from IT, sign the acceptable use policy.
Day 2: complete security training. The training portal password resets every 90 days.
Week 1: meet your buddy and shadow one on-call shift.
""")


def _png(p: Path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1100, 420), "white")
    dr = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except OSError:
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", 40)
        except OSError:
            f = ImageFont.load_default(size=40)
    for i, line in enumerate(["Warehouse door code", "Rotate the door code every 30 days.", "Owner: Priya Raman",
                              "Next rotation: first Monday of the month"]):
        dr.text((40, 40 + i * 80), line, fill="black", font=f)
    img.save(p)


BUILD = {"Employee_Handbook.docx": _docx, "Q3_Board_Update.pptx": _pptx, "Sales_Pipeline_2026.xlsx": _xlsx,
         "Supplier_Contacts.csv": _csv, "Release_Notes.html": _html, "Incident_4471.eml": _eml,
         "Safety_Manual.pdf": _pdf, "Architecture_Notes.md": _md, "Onboarding.txt": _txt, "Whiteboard_Photo.png": _png}

# (id, question, category, doc stem, [pages], must_contain, what is being tested)
Q = [
    ("docx-table", "What is the daily meal limit for international travel?", "table", "Employee_Handbook", [1], [["70"]], "DOCX table cell"),
    ("docx-para", "How many days of paid leave do employees accrue each month?", "numeric", "Employee_Handbook", [1], [["1.75"]], "DOCX heading + paragraph"),
    ("pptx-table", "What was the Q3 revenue for APAC?", "table", "Q3_Board_Update", [3], [["3.1"]], "PPTX table on slide 3"),
    ("pptx-notes", "When is the general availability date of the Helio X2?", "factoid", "Q3_Board_Update", [4], [["14 march 2027"]], "PPTX speaker notes (slide 4)"),
    ("pptx-chart", "How many planned units are there for the Beta stage?", "numeric", "Q3_Board_Update", [4], [["340"]], "PPTX chart data"),
    ("xlsx-sheet2", "What is the value of the Falcon deal?", "table", "Sales_Pipeline_2026", [2], [["480000"]], "XLSX second sheet"),
    ("xlsx-stage", "Which deal is at the verbal commit stage?", "table", "Sales_Pipeline_2026", [2], [["osprey"]], "XLSX row lookup"),
    ("csv-row", "Who is the contact at Norden Fasteners?", "table", "Supplier_Contacts", [1], [["ingrid solberg"]], "CSV row"),
    ("html-table", "Which version introduced the batch export API?", "table", "Release_Notes", [1], [["3.5"]], "HTML table behind nav/script noise"),
    ("eml-body", "What was the root cause of incident 4471?", "factoid", "Incident_4471", [1], [["expired tls certificate"]], "e-mail body"),
    ("pdf-fact", "What is the maximum rated load of the F-30 forklift?", "numeric", "Safety_Manual", [1], [["2,500 kg", "2500 kg"]], "PDF text"),
    ("md-fact", "How long are audit logs retained?", "numeric", "Architecture_Notes", [1], [["400 days"]], "Markdown"),
    ("txt-fact", "How often does the training portal password reset?", "numeric", "Onboarding", [1], [["90 days"]], "plain text"),
    ("img-ocr", "Who owns the warehouse door code rotation?", "factoid", "Whiteboard_Photo", [1], [["priya raman"]], "text inside an image (OCR)"),
    ("para-leave", "What happens to leftover holiday allowance at year end?", "paraphrase", "Employee_Handbook", [1], [["12 days"]], "paraphrase, no shared keywords"),
    ("none", "What is the CEO's favourite colour?", "unanswerable", "", [], [], "abstain"),
]


def build(out: str | Path) -> Path:
    out = Path(out)
    corpus = out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILD.items():
        fn(corpus / name)
    files = [{"file": f"corpus/{n}", "tenant": "default", "acl": ["public"]} for n in BUILD]
    (out / "manifest.json").write_text(json.dumps({"name": "helio-formats", "files": files}, indent=1))
    qs = []
    for qid, q, cat, doc, pages, must, note in Q:
        ans = bool(doc)
        qs.append(GoldQuestion(id=qid, question=q, category=cat, answerable=ans, gold=[GoldRef(doc=doc, pages=pages)] if ans else [],
                               must_contain=must, notes=note))
    GoldSet(name="helio-formats", description="Facts planted in one place per file format; tests the parsers, not retrieval quality",
            questions=qs).save(out / "gold.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/formats")
    a = ap.parse_args()
    print("wrote", build(a.out), file=sys.stderr)
