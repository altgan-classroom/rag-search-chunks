"""Qdrant parameter benchmark on a real server (docker): HNSW m / ef, quantization, and filtered search with/without
payload indexes. Vectors are a Gaussian mixture (clustered like real embeddings); ground truth is exact brute force.

    docker run -d --name qd -p 6333:6333 qdrant/qdrant:v1.20.2
    python -m orca_rag.eval.vector_bench --n 100000 --dim 384
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from qdrant_client import QdrantClient, models as qm


def make_data(n: int, dim: int, nq: int, clusters: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    assign = rng.integers(0, clusters, n)
    x = centers[assign] + 0.35 * rng.normal(size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    qa = rng.integers(0, clusters, nq)
    q = centers[qa] + 0.35 * rng.normal(size=(nq, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tenant = rng.integers(0, 100, n)  # 1% selectivity per tenant
    return x, q, tenant


def truth(x, q, k, mask=None):
    xs = x if mask is None else x[mask]
    ids = np.arange(len(x)) if mask is None else np.nonzero(mask)[0]
    out = []
    for qi in q:
        sims = xs @ qi
        top = np.argpartition(-sims, k)[:k]
        out.append(set(ids[top].tolist()))
    return out


def build(client, name, x, tenant, *, m=16, ef_construct=128, quant=None, payload_index=True):
    if client.collection_exists(name):
        client.delete_collection(name)
    qc = None
    if quant == "scalar":
        qc = qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, quantile=0.99, always_ram=True))
    elif quant == "binary":
        qc = qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True))
    client.create_collection(name, vectors_config=qm.VectorParams(size=x.shape[1], distance=qm.Distance.COSINE),
                             hnsw_config=qm.HnswConfigDiff(m=m, ef_construct=ef_construct), quantization_config=qc)
    if payload_index:
        client.create_payload_index(name, "tenant", qm.PayloadSchemaType.INTEGER)
    t0 = time.perf_counter()
    for i in range(0, len(x), 2000):
        client.upsert(name, points=[qm.PointStruct(id=int(j), vector=x[j].tolist(), payload={"tenant": int(tenant[j])})
                                    for j in range(i, min(i + 2000, len(x)))], wait=False)
    # wait for the optimiser to finish indexing
    while client.get_collection(name).status != qm.CollectionStatus.GREEN:
        time.sleep(1)
    return time.perf_counter() - t0


def measure(client, name, q, gt, k=10, ef=None, exact=False, flt=None, quant=None, rescore=True, oversample=2.0):
    sp = qm.SearchParams(hnsw_ef=ef, exact=exact)
    if quant:
        sp.quantization = qm.QuantizationSearchParams(rescore=rescore, oversampling=oversample)
    lat, rec = [], []
    for qi, g in zip(q, gt):
        t0 = time.perf_counter()
        res = client.query_points(name, query=qi.tolist(), limit=k, search_params=sp, query_filter=flt).points
        lat.append((time.perf_counter() - t0) * 1000)
        rec.append(len({p.id for p in res} & g) / k)
    lat.sort()
    return statistics.mean(rec), statistics.median(lat), lat[int(0.95 * (len(lat) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--nq", type=int, default=200)
    a = ap.parse_args()
    client = QdrantClient(url=a.url, timeout=120)
    x, q, tenant = make_data(a.n, a.dim, a.nq)
    gt = truth(x, q, 10)
    mask = tenant == 7
    gt_f = truth(x, q, 10, mask)
    rows = []

    def row(label, rec, p50, p95, extra=""):
        rows.append(f"| {label} | {rec:.3f} | {p50:.1f} | {p95:.1f} | {extra} |")
        print(rows[-1], flush=True)

    print(f"n={a.n} dim={a.dim} queries={a.nq}\n| config | recall@10 | p50 ms | p95 ms | note |\n|---|---:|---:|---:|---|")
    for m in (8, 16, 32):
        bt = build(client, "b", x, tenant, m=m)
        for ef in (32, 64, 128, 256):
            r, p50, p95 = measure(client, "b", q, gt, ef=ef)
            row(f"HNSW m={m} ef={ef}", r, p50, p95, f"build {bt:.0f}s" if ef == 32 else "")
        if m == 16:
            r, p50, p95 = measure(client, "b", q, gt, exact=True)
            row("exact (brute force)", r, p50, p95)
    for quant in (None, "scalar", "binary"):
        build(client, "b", x, tenant, quant=quant)
        info = client.get_collection("b")
        r, p50, p95 = measure(client, "b", q, gt, ef=128, quant=quant)
        row(f"quantization={quant or 'none'} ef=128 (rescore)", r, p50, p95, f"segments={len(info.payload_schema) if False else ''}")
    # filtered search (tenant selectivity 1%): payload index on/off
    for pi in (True, False):
        build(client, "b", x, tenant, payload_index=pi)
        flt = qm.Filter(must=[qm.FieldCondition(key="tenant", match=qm.MatchValue(value=7))])
        r, p50, p95 = measure(client, "b", q, gt_f, ef=128, flt=flt)
        row(f"filter tenant=7 (1%), payload_index={pi}", r, p50, p95)
    client.delete_collection("b")


if __name__ == "__main__":
    main()
