from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()
REQUESTS = Counter("orca_requests_total", "Chat requests", ["endpoint", "outcome"], registry=REGISTRY)
LATENCY = Histogram("orca_request_seconds", "End-to-end chat latency", ["endpoint"], registry=REGISTRY,
                    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))
TTFT = Histogram("orca_first_token_seconds", "Time to first streamed token", registry=REGISTRY,
                 buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
CACHE = Counter("orca_cache_total", "Semantic cache lookups", ["result"], registry=REGISTRY)
ABSTAIN = Counter("orca_abstentions_total", "Answers that abstained", registry=REGISTRY)
UNGROUNDED = Counter("orca_ungrounded_total", "Answers failing the grounding check", registry=REGISTRY)


def render() -> bytes:
    return generate_latest(REGISTRY)
