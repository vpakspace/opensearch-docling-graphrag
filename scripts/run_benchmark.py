#!/usr/bin/env python3
"""Run benchmark — 30 questions x 6 modes = 180 evaluations.

Evaluates all search modes (bm25, vector, graph, hybrid, enhanced, cognitive)
against the 30-question benchmark set (Doc1 RU + Doc2 EN).

Evaluation: keyword overlap judge (no external API needed).
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("benchmark")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from neo4j import GraphDatabase

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.opensearch_store import OpenSearchStore
from opensearch_graphrag.service import PipelineService

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cfg = get_settings()
store = OpenSearchStore(settings=cfg)
driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
service = PipelineService(store=store, neo4j_driver=driver, settings=cfg)

# Verify connectivity
health = service.health()
logger.info("Health: %s", health)
stats = service.graph_stats()
logger.info("Graph: %s", stats)

# ---------------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------------

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")
with open(os.path.join(BENCH_DIR, "questions.json")) as f:
    questions = json.load(f)

doc1 = sum(1 for q in questions if q["id"] <= 15)
doc2 = sum(1 for q in questions if q["id"] > 15)
print(f"\nLoaded {len(questions)} questions (Doc1 RU: {doc1}, Doc2 EN: {doc2})")

# ---------------------------------------------------------------------------
# Cross-language concept map (RU ↔ EN)
# ---------------------------------------------------------------------------

CONCEPT_MAP = {
    "онтология": "ontology",
    "граф": "graph",
    "извлечение": "extraction",
    "сущности": "entities",
    "триплеты": "triplets",
    "схема": "schema",
    "валидация": "validation",
    "верификация": "verification",
    "пайплайн": "pipeline",
    "хранение": "storage",
    "временных": "temporal",
    "компоненты": "components",
    "архитектура": "architecture",
    "улучшения": "improvements",
    "построение": "construction",
    "фреймворк": "framework",
    "мультимодальных": "multimodal",
    "интеграция": "integration",
    "эволюция": "evolution",
    "изменения": "changes",
    "качество": "quality",
    "стратегия": "strategy",
    "сравнение": "comparison",
    "локальные": "local",
    "методы": "methods",
    "обзор": "overview",
}


# ---------------------------------------------------------------------------
# Keyword overlap judge
# ---------------------------------------------------------------------------

_GLOBAL_RE = re.compile(
    r"(все|перечисли|list all|describe all|overview|резюмируй|summarize)",
    re.IGNORECASE,
)


def evaluate_answer(question: dict, answer: str) -> bool:
    """Evaluate answer using keyword overlap with cross-language matching."""
    if not answer or len(answer) < 20:
        return False

    keywords = question.get("keywords", [])
    if not keywords:
        return len(answer) > 50  # no keywords → accept any non-trivial answer

    answer_lower = answer.lower()

    # Count keyword matches (direct + cross-language)
    matched = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in answer_lower:
            matched += 1
            continue
        # Cross-language: check if concept equivalent appears
        concept_en = CONCEPT_MAP.get(kw_lower)
        if concept_en and concept_en in answer_lower:
            matched += 1
            continue
        # Reverse map: EN → RU
        for ru, en in CONCEPT_MAP.items():
            if en == kw_lower and ru in answer_lower:
                matched += 1
                break

    overlap = matched / len(keywords) if keywords else 0.0

    # Global/enumeration questions need higher threshold
    q_text = question.get("question_ru", question.get("question", ""))
    threshold = 0.5 if _GLOBAL_RE.search(q_text) else 0.3

    # Also check reference_answer if available
    ref = question.get("reference_answer", "")
    if ref and overlap < threshold:
        ref_lower = ref.lower()
        ref_words = set(re.findall(r"\b\w{4,}\b", ref_lower))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer_lower))
        ref_overlap = len(ref_words & answer_words) / len(ref_words) if ref_words else 0
        if ref_overlap >= 0.3:
            return True

    return overlap >= threshold


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

MODES = ["bm25", "vector", "graph", "hybrid", "enhanced", "cognitive"]
results: dict[str, list[dict]] = {}

for mode in MODES:
    mode_results = []
    print(f"\n{'─' * 60}")
    print(f"  Mode: {mode}")
    print(f"{'─' * 60}")

    for q in questions:
        qid = q["id"]
        # Use Russian for Doc1 (id <= 15), English for Doc2 (id > 15)
        question_text = q.get("question_ru", q["question"]) if qid <= 15 else q["question"]
        qtype = q.get("type", "unknown")

        t0 = time.time()
        try:
            qa = service.query(question_text, mode=mode)
            answer = qa.answer
            confidence = qa.confidence
        except Exception as e:
            answer = f"ERROR: {e}"
            confidence = 0.0
        latency = time.time() - t0

        passed = evaluate_answer(q, answer)
        mark = "PASS" if passed else "FAIL"
        print(f"  Q{qid:2d} [{qtype:10s}] {mark}  ({latency:.1f}s)  {answer[:80]}...")

        mode_results.append({
            "id": qid,
            "question": question_text,
            "type": qtype,
            "answer": answer,
            "confidence": confidence,
            "latency": round(latency, 2),
            "passed": passed,
        })

    results[mode] = mode_results
    passed_count = sum(1 for r in mode_results if r["passed"])
    print(f"\n  {mode}: {passed_count}/{len(mode_results)} ({100 * passed_count // len(mode_results)}%)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print(f"BENCHMARK — 30 questions x {len(MODES)} modes = {30 * len(MODES)} evaluations")
print(f"LLM: {cfg.ollama.llm_model}  |  Embeddings: {cfg.ollama.embed_model}")
print(f"{'=' * 70}")

for mode in MODES:
    mr = results[mode]
    passed = sum(1 for r in mr if r["passed"])
    total = len(mr)
    doc1_pass = sum(1 for r in mr if r["id"] <= 15 and r["passed"])
    doc2_pass = sum(1 for r in mr if r["id"] > 15 and r["passed"])
    avg_lat = sum(r["latency"] for r in mr) / total if total else 0

    details = " ".join(f"Q{r['id']}:{'P' if r['passed'] else 'F'}" for r in sorted(mr, key=lambda x: x["id"]))
    print(f"\n  {mode:12s}: {passed}/{total} ({100 * passed // total}%)  avg {avg_lat:.1f}s")
    print(f"    Doc1 (RU): {doc1_pass}/15  Doc2 (EN): {doc2_pass}/15")
    print(f"    {details}")

total_passed = sum(sum(1 for r in mr if r["passed"]) for mr in results.values())
total_all = sum(len(mr) for mr in results.values())
doc1_total = sum(sum(1 for r in mr if r["id"] <= 15 and r["passed"]) for mr in results.values())
doc2_total = sum(sum(1 for r in mr if r["id"] > 15 and r["passed"]) for mr in results.values())

print(f"\n{'=' * 70}")
print(f"  {'OVERALL':12s}: {total_passed}/{total_all} ({100 * total_passed // total_all}%)")
print(f"    Doc1 (RU): {doc1_total}/{15 * len(MODES)}  Doc2 (EN): {doc2_total}/{15 * len(MODES)}")

# By question type
type_stats: dict[str, dict[str, int]] = {}
for mr in results.values():
    for r in mr:
        t = r.get("type", "unknown")
        if t not in type_stats:
            type_stats[t] = {"passed": 0, "total": 0}
        type_stats[t]["total"] += 1
        if r["passed"]:
            type_stats[t]["passed"] += 1

print("\n  By question type:")
for t, s in sorted(type_stats.items()):
    print(f"    {t:12s}: {s['passed']}/{s['total']} ({100 * s['passed'] // s['total']}%)")

# Persistent failures
print("\n  Persistent failures (0/{} modes):".format(len(MODES)))
for q in questions:
    passes = 0
    for mr in results.values():
        for r in mr:
            if r["id"] == q["id"] and r["passed"]:
                passes += 1
    if passes == 0:
        print(f"    Q{q['id']} ({q.get('type', '?')}): {q.get('question_ru', q['question'])[:70]}...")

# Latency percentiles
all_latencies = sorted(r["latency"] for mr in results.values() for r in mr)
if all_latencies:
    def _percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    p50 = _percentile(all_latencies, 50)
    p95 = _percentile(all_latencies, 95)
    p99 = _percentile(all_latencies, 99)
    print(f"\n  Latency: p50={p50:.1f}s  p95={p95:.1f}s  p99={p99:.1f}s")

# Save results
out_path = os.path.join(BENCH_DIR, "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}")

# Save history
history_dir = os.path.join(BENCH_DIR, "history")
os.makedirs(history_dir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
history_path = os.path.join(history_dir, f"run_{ts}.json")
history_data = {
    "timestamp": ts,
    "llm_model": cfg.ollama.llm_model,
    "embed_model": cfg.ollama.embed_model,
    "modes": MODES,
    "total_passed": total_passed,
    "total_all": total_all,
    "accuracy": round(total_passed / total_all, 4) if total_all else 0,
    "latency_p50": p50 if all_latencies else 0,
    "latency_p95": p95 if all_latencies else 0,
    "latency_p99": p99 if all_latencies else 0,
    "results": results,
}
with open(history_path, "w") as f:
    json.dump(history_data, f, ensure_ascii=False, indent=2)
print(f"History saved to {history_path}")

driver.close()
