"""Gold questions for three open arXiv papers (RAG, Self-RAG, CRAG). Model-authored, needs_review, and every expected
string is machine-verified on its gold page text (the build fails otherwise)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from orca_rag.eval.metrics import norm  # noqa: E402
from orca_rag.eval.schema import CorpusFile, GoldQuestion, GoldRef, GoldSet  # noqa: E402
from orca_rag.ingestion.parsers import parse_cached  # noqa: E402

RAW, CACHE = HERE.parents[1] / "data" / "raw" / "open", HERE.parents[1] / "data" / "parsed"
RAG, SELF, CRAG = "arxiv-2005.11401-RAG-Lewis", "arxiv-2310.11511-Self-RAG", "arxiv-2401.15884-CRAG"
Q: list[GoldQuestion] = []


def add(id_, q, cat, refs=(), must=(), answerable=True, **kw):
    Q.append(GoldQuestion(id=id_, question=q, category=cat, answerable=answerable, source="model_authored", status="needs_review",
                          gold=[GoldRef(doc=d, pages=list(p)) for d, p in refs], must_contain=[list(g) for g in must], **kw))


add("rag-index", "How many Wikipedia chunks make up the non-parametric memory index in the RAG paper?", "numeric", [(RAG, [4, 19])], [["21M", "21 million"]])
add("rag-gen", "Which seq2seq model serves as the generator in RAG?", "factoid", [(RAG, [2])], [["BART"]])
add("rag-variants", "What are the two RAG model formulations?", "factoid", [(RAG, [3])], [["RAG-Sequence"], ["RAG-Token"]])
add("rag-chunk", "How long is each Wikipedia chunk in the RAG paper's index?", "numeric", [(RAG, [4])], [["100-word", "100 word"]])
add("rag-hnsw", "Which approximate search method does RAG use for fast document retrieval?", "factoid", [(RAG, [4])], [["Navigable Small World"]])
add("self-sizes", "Which two parameter sizes of Self-RAG are evaluated?", "numeric", [(SELF, [1])], [["7B"], ["13B"]])
add("self-tokens", "Into which two categories are Self-RAG's reflection tokens divided?", "factoid", [(SELF, [1])], [["retrieval"], ["critique"]])
add("self-critic", "Why does Self-RAG use an offline critic model when creating training data?", "paraphrase", [(SELF, [2])], [["reducing overhead"]])
add("self-issup", "What values can the ISSUP critique token take in Self-RAG?", "definition", [(SELF, [4])], [["fully supported"]])
add("self-isrel", "What does the ISREL token decide in Self-RAG?", "definition", [(SELF, [4])], [["provides useful information"]])
add("crag-init", "Which model initialises the retrieval evaluator in CRAG?", "factoid", [(CRAG, [4])], [["T5-large"]])
add("crag-acc", "What accuracy does the CRAG retrieval evaluator reach in the paper's evaluation table?", "numeric", [(CRAG, [9])], [["84.3"]])
add("crag-algo", "Which algorithm does CRAG use to strip redundant content from retrieved documents?", "factoid", [(CRAG, [1, 2])], [["decompose-then-recompose"]])
add("crag-web", "What does CRAG fall back on when the retrieved documents are judged incorrect or ambiguous?", "paraphrase", [(CRAG, [2])], [["web search"]])
add("crag-actions", "What are the three retrieval actions CRAG can trigger?", "factoid", [(CRAG, [2])], [["Correct"], ["Incorrect"], ["Ambiguous"]])
add("cmp-reflect", "Which approach relies on special reflection tokens: RAG, Self-RAG or CRAG?", "comparison", [(SELF, [1])], [["Self-RAG", "SELF-RAG", "SELFRAG"]])
add("cmp-evaluator", "Which of these papers introduces a lightweight retrieval evaluator?", "comparison", [(CRAG, [1])], [["CRAG", "Corrective"]])
for i, qt in enumerate(["What is the price per million tokens of the Self-RAG critic model API?",
                        "Which vector database version did the original RAG paper deploy in production?",
                        "How many GPUs did the CRAG authors use for web search caching?"]):
    add(f"unans-{i}", qt, "unanswerable", answerable=False)


def verify():
    errs, cache = [], {}
    for q in Q:
        for r in q.gold:
            cache.setdefault(r.doc, parse_cached(str(RAW / f"{r.doc}.pdf"), CACHE))
        text = norm(" ".join(p.markdown for r in q.gold for p in cache[r.doc].pages if not r.pages or p.number in r.pages))
        for grp in q.must_contain:
            if not any(norm(a) in text for a in grp):
                errs.append(f"{q.id}: none of {grp} on gold pages")
    return errs


if __name__ == "__main__":
    errs = verify()
    if errs:
        print("\n".join(errs))
        sys.exit(f"{len(errs)} verification error(s)")
    GoldSet(name="open-arxiv", description="Open arXiv papers on RAG (domain-shift check). Model-authored, machine-verified, needs review.",
            questions=Q).save(HERE / "gold.json")
    src = json.loads((HERE / "sources.json").read_text())
    (HERE / "manifest.json").write_text(json.dumps({"name": "open-arxiv", "files": [CorpusFile(file=f"../../data/raw/open/{s}.pdf").model_dump() for s in src]}, indent=1))
    print(f"{len(Q)} questions verified on {len(src)} papers")
