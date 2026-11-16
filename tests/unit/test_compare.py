from orca_rag.eval.compare import paired_delta


def R(vals, cat="factoid"):
    return {f"q{i}": {"category": cat, "metrics": {"m": v}} for i, v in enumerate(vals)}


def test_consistent_small_gain_is_detected_by_pairing():
    a = R([0.5] * 40)
    b = R([0.55] * 40)       # every question improves a little: unpaired CIs would overlap, paired CI excludes 0
    assert paired_delta(a, b, "m")["verdict"] == "better"


def test_noise_is_reported_as_no_effect():
    a = R([1, 0] * 20)
    b = R([0, 1] * 20)       # same mean, individual flips
    r = paired_delta(a, b, "m")
    assert r["verdict"] == "no measurable effect" and abs(r["mean"]) < 1e-9 and r["wins"] == r["losses"] == 20


def test_regression_detected_and_missing_metric_handled():
    assert paired_delta(R([1] * 30), R([0.9] * 30), "m")["verdict"] == "worse"
    assert paired_delta(R([1] * 3), R([1] * 3), "absent")["verdict"] == "n/a"


def test_category_filter():
    a = {**R([0.5] * 10, "a"), **{f"z{i}": {"category": "b", "metrics": {"m": 0.5}} for i in range(10)}}
    b = {**R([0.9] * 10, "a"), **{f"z{i}": {"category": "b", "metrics": {"m": 0.5}} for i in range(10)}}
    assert paired_delta(a, b, "m", categories={"a"})["verdict"] == "better"
    assert paired_delta(a, b, "m", categories={"b"})["verdict"] == "no measurable effect"
