"""Builds datasets/ccs/{gold.json,manifest.json} for the real CO2-storage corpus (Illinois Basin - Decatur Project).

Questions are MODEL-AUTHORED from the actual document text and marked status="needs_review": a human should confirm
each one (see docs/DOMAIN_EVAL.md for the review protocol). Every `must_contain` string is machine-verified to occur
on the gold page(s) of the gold document, so a typo or wrong page fails the build instead of poisoning the eval.

    python datasets/ccs/download.py          # fetch the pinned corpus (Apache-2.0 HF dataset, sha256-verified)
    python datasets/ccs/build_gold.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from orca_rag.eval.metrics import norm  # noqa: E402
from orca_rag.eval.schema import CorpusFile, GoldQuestion, GoldRef, GoldSet  # noqa: E402
from orca_rag.ingestion.parsers import parse_cached  # noqa: E402

RAW = HERE.parents[1] / "data" / "raw" / "ccs"
CACHE = HERE.parents[1] / "data" / "parsed"
CCS1 = "1211523415_20191113_CCS1_Well_Summary_Sheet"
VW1 = "1211523460_20200916_VW1_Well_Summary_Sheet"
GM1 = "1211523438_20140708_GM1_Well_Summary_Sheet"
GM1A = "1211523438_20140708_GM1_Well_Summary_Sheet-Alt"
IBDP = "IBDP_FactSheet_2015_Final_sm"
FIN = "Financial-Assurance-fact-sheet2"
TERMS = "IBDP - CO2 DataShare Terms Final"
MIT = "1211523415_20130700_CCS1_Periodic_Report_Mechanical_Integrity_bb"
F12 = "1211523415_20121005_CCS1_Ambient_Falloff_Test_Report"
F13 = "1211523415_20130607_CCS1_Ambient_Falloff_Test_Report"

Q: list[GoldQuestion] = []


def add(id_, q, cat, refs=(), must=(), answerable=True, **kw):
    Q.append(GoldQuestion(id=id_, question=q, category=cat, answerable=answerable, source="model_authored", status="needs_review",
                          gold=[GoldRef(doc=d, pages=list(p)) for d, p in refs], must_contain=[list(g) for g in must], **kw))


# ---- factoid / numeric (CCS1, VW1 sheets)
add("ccs1-cost", "What is the total well cost of injection well CCS1?", "factoid", [(CCS1, [1])], [["6.6 million"]])
add("ccs1-rig", "Which drilling company drilled the CCS1 injection well?", "factoid", [(CCS1, [1])], [["Les Wilson"]])
add("ccs1-pmax", "What is the maximum wellhead pressure allowed by the CCS1 permit?", "numeric", [(CCS1, [2])], [["1950 psig", "1,950 psig"]])
add("ccs1-rate", "What injection rate range does the CCS1 permit allow?", "numeric", [(CCS1, [2])], [["14.1 tonnes per hour"], ["49.7"]])
add("ccs1-daily", "What is the maximum amount of CO2 that may be injected per day into CCS1?", "numeric", [(CCS1, [2])], [["1,200 tonnes per day"]])
add("ccs1-perf", "Between which depths is the CCS1 well perforated?", "table", [(CCS1, [2])], [["6,982"], ["7,012"]])
add("ccs1-sigma", "How often must a temperature survey and time-lapse sigma log be run on CCS1?", "factoid", [(CCS1, [2])], [["24 months"]])
add("ccs1-annulus", "What minimum annulus pressure must be maintained on CCS1?", "numeric", [(CCS1, [2])], [["400 psi"]])
add("ccs1-gauge", "At what measured depth is the CCS1 pressure/temperature gauge mounted?", "factoid", [(CCS1, [1])], [["6,325 ft"]])
add("ccs1-packer", "What type of packer is installed in CCS1?", "factoid", [(CCS1, [1])], [["Q-Max", "QUANTUM MAX"]])
add("ccs1-cement", "What cement system was used between 4,170 and 7,219 ft in CCS1?", "factoid", [(CCS1, [1])], [["EverCRETE"]])
add("ccs1-dts", "How often along the fibre does the CCS1 distributed temperature sensing cable measure temperature?", "numeric", [(CCS1, [2])], [["1.624 ft"]])
add("vw1-cost", "What was the total well cost of verification well VW1?", "factoid", [(VW1, [1])], [["4.4 million"]])
add("vw1-westbay", "Which multilevel monitoring system was installed in VW1 between 2010 and 2017?", "factoid", [(VW1, [2])], [["Westbay"]])
add("vw1-packers", "How many inflatable packers did the Westbay system in VW1 use for zonal isolation?", "numeric", [(VW1, [2])], [["28 inflatable packers", "28"]])
add("vw1-mit", "How often is the internal and external mechanical integrity test required for VW1?", "factoid", [(VW1, [2])], [["5 years"]])
add("vw1-eauclaire", "What is the formation top depth of the Eau Claire in VW1?", "table", [(VW1, [1])], [["5,010"]])
add("cmp-cost", "Which well cost more to drill, CCS1 or VW1?", "comparison", [(CCS1, [1]), (VW1, [1])], [["CCS1"]])
# ---- conflicting versions (two GM1 sheets disagree)
add("gm1-cost", "What was the total well cost of geophysical monitoring well GM1?", "conflicting", [(GM1, [1]), (GM1A, [1])],
    [["504,000", "4.0 million"]], notes="the two GM1 sheets disagree (~$504,000 vs ~4.0 million): ideal answers flag the conflict")
add("gm1-sources", "How many source points were acquired in the 2013 3D VSP at GM1?", "factoid", [(GM1A, [1])], [["557"]])
add("gm1-fluid", "What drilling fluid was used in the final section of GM1?", "factoid", [(GM1, [2]), (GM1A, [1])], [["WBM", "water based mud"]])
add("gm1-cement", "How many sacks of Franklin cement were run in the final section of GM1?", "numeric", [(GM1, [2]), (GM1A, [1])], [["920"]])
# ---- IBDP fact sheet / policy docs
add("ibdp-start", "When did injection start at the Illinois Basin - Decatur Project?", "factoid", [(IBDP, [1, 2])], [["November 17, 2011"]])
add("ibdp-volume", "What total volume of CO2 was injected at IBDP?", "numeric", [(IBDP, [1])], [["999,215 tonnes"]])
add("ibdp-seal", "At what depth is the top of the Eau Claire seal at IBDP according to the fact sheet?", "factoid", [(IBDP, [2])], [["5,047 ft"]])
add("ibdp-whp", "What wellhead injection pressure did the IBDP fact sheet report?", "numeric", [(IBDP, [2])], [["1,350 psi"]])
add("ibdp-pipe", "How long is the delivery pipeline at IBDP?", "factoid", [(IBDP, [2])], [["1.2 mile"]])
add("ibdp-holder", "Who holds the UIC injection permit for IBDP?", "factoid", [(IBDP, [2])], [["Archer Daniels Midland"]])
add("ibdp-thick", "What is the thickness of the Mount Simon reservoir in the IBDP fact sheet?", "numeric", [(IBDP, [2])], [["1,506 ft"]])
add("def-mva", "What does MVA stand for in the context of IBDP?", "definition", [(IBDP, [1])], [["Monitoring, Verification, and Accounting"]])
add("fin-instr", "Which financial instruments does USEPA recommend for Class VI financial responsibility?", "factoid", [(FIN, [2])],
    [["trust fund"], ["surety bond"], ["letter of credit"], ["escrow"]])
add("fin-cfr", "Under which federal regulation did USEPA publish the Class VI requirements in 2010?", "factoid", [(FIN, [1])], [["40 CFR 146"]])
add("terms-sell", "May the IBDP CO2 injection datasets be sold?", "factoid", [(TERMS, [1, 3])], [["may not sell"]])
add("terms-cc", "Which license is the IBDP CO2 injection datasets license based on?", "factoid", [(TERMS, [1])], [["CC BY 4.0"]])
add("mit-date", "On what date was the RST run in CCS #1 for the July 2013 mechanical integrity report?", "factoid", [(MIT, [1])], [["July 12, 2013"]])
add("mit-depth", "To what depth had CO2 in the formation migrated according to the July 2013 RST analysis?", "numeric", [(MIT, [2])], [["6750"]])
add("def-rst", "What does the RST used for mechanical integrity monitoring in CCS #1 stand for?", "definition", [(MIT, [1])], [["Reservoir Saturation Tool"]])
# ---- 2012 vs 2013 falloff tests (long documents, near-identical structure)
add("f12-trans", "What was the calculated transmissibility in the 2012 ambient falloff test of CCS No. 1?", "temporal", [(F12, [5, 10])], [["9,531"]])
add("f13-trans", "What was the calculated transmissibility in the 2013 ambient falloff test of CCS No. 1?", "temporal", [(F13, [5, 10])],
    [["60,668", "60,939"]], notes="the 2013 report's summary (60,668) and its own table (60,939) disagree; either is accepted")
add("f13-radius", "What was the radius of investigation in the 2013 ambient falloff test?", "temporal", [(F13, [5, 10])], [["12,177"]])
add("f12-dates", "Over what dates was the 2012 ambient pressure falloff test conducted?", "temporal", [(F12, [5])], [["October 5"]])
add("f12-skin", "What skin value was calculated in the 2012 falloff test?", "table", [(F12, [10])], [["-5.95"]])
add("f13-perm", "What native formation permeability was calculated in the 2013 falloff test?", "table", [(F13, [10])], [["945.1 md"]])
add("f13-gauge", "What kind of gauge was used to record pressures in the 2013 falloff test?", "temporal", [(F13, [5])], [["slickline"]])
add("f13-grad", "What pressure was measured at 7,025 feet in the static gradient stop of the 2013 falloff test?", "table", [(F13, [10])], [["3,247.28"]])
add("cmp-trans", "Which year's ambient falloff test showed higher transmissibility, 2012 or 2013?", "comparison", [(F12, [5, 10]), (F13, [5, 10])], [["2013"]])
# ---- follow-ups (need history-aware rewriting)
H1 = [("user", "Tell me about the CCS1 injection well."), ("assistant", "CCS1 is the injection well of the Midwest Geological Sequestration Consortium.")]
add("fu-cost", "What is its total well cost?", "follow_up", [(CCS1, [1])], [["6.6 million"]], history=H1)
add("fu-pmax", "And what is its maximum wellhead pressure?", "follow_up", [(CCS1, [2])], [["1,950"]], history=H1)
H2 = [("user", "What can you tell me about VW1?"), ("assistant", "VW1 is the verification well.")]
add("fu-vw1", "How often is its mechanical integrity test required?", "follow_up", [(VW1, [2])], [["5 years"]], history=H2)
# ---- unanswerable (verified absent from the 17-document subset by search, see needs_review)
for i, qt in enumerate(["What was the price per tonne of CO2 purchased from ADM?", "How many people were employed on the IBDP in 2015?",
                        "What is the brand of safety helmets used on the CCS1 drilling rig?", "What is the H2S content of the injected CO2 stream?",
                        "What was the total well cost of verification well VW2?", "Who is the chief executive officer of Schlumberger Carbon Services?"]):
    add(f"unans-{i}", qt, "unanswerable", answerable=False)


def verify():
    errs, cache = [], {}
    for q in Q:
        for r in q.gold:
            if r.doc not in cache:
                cache[r.doc] = parse_cached(str(RAW / f"{r.doc}.pdf"), CACHE)
            doc = cache[r.doc]
            pages = [p for p in doc.pages if (not r.pages or p.number in r.pages)]
            if not pages:
                errs.append(f"{q.id}: doc {r.doc} has no page(s) {r.pages}")
        if q.must_contain:
            for grp in q.must_contain:
                ok = any(norm(alt) in norm(" ".join(p.markdown for r in q.gold for p in cache[r.doc].pages
                                                    if not r.pages or p.number in r.pages)) for alt in grp)
                if not ok:
                    errs.append(f"{q.id}: none of {grp} found on gold pages")
    return errs


if __name__ == "__main__":
    errs = verify()
    if errs:
        print("\n".join(errs))
        sys.exit(f"{len(errs)} gold verification error(s)")
    gs = GoldSet(name="ccs-real", description="Real CO2-storage corpus (IBDP). Model-authored, machine-verified against page text, needs human review.", questions=Q)
    gs.save(HERE / "gold.json")
    man = json.loads((HERE / "sources.json").read_text())
    files = [CorpusFile(file=f"../../data/raw/ccs/{m['file']}") for m in man["files"]]
    (HERE / "manifest.json").write_text(json.dumps({"name": "ccs-real", "files": [f.model_dump() for f in files]}, indent=1))
    print(f"{len(Q)} questions {gs.by_category()}; {len(files)} documents; all gold strings verified on their pages")
