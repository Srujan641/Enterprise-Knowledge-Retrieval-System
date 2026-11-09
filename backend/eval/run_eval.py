"""Run the benchmark through the real pipeline and write a report.

Point DATA_DIR and FAISS_INDEX_PATH at a temporary folder so real documents are untouched.
"""
import argparse
import csv
import json
import math
import re
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from agents.rag_agent import get_answer  # noqa: E402
from ingestion.document_processor import process_document  # noqa: E402
from retrieval.vector_store import vector_store_manager as store  # noqa: E402
from utils import budget  # noqa: E402
from utils.budget import BudgetExceeded  # noqa: E402
from utils.llm import active_provider, get_llm  # noqa: E402

DATA = Path(__file__).parent / "data"
ABSTAIN = re.compile(
    r"(couldn.?t|could not|can.?t|cannot|unable to|did not|didn.?t|failed to) (find|locate|determine|identify|confirm|answer)"
    r"|no (relevant |specific |such )?(information|mention|details?|data|record)"
    r"|not (found|available|mentioned|specified|stated|provided|contain|included)"
    r"|(does|do)(n.?t| not) (contain|mention|specify|state|say|provide|include|list|indicate|have)",
    re.IGNORECASE,
)


def tokens(text):
    text = "".join(" " if c in string.punctuation or c in "–—‘’“”" else c for c in text.lower())
    return [t for t in text.split() if t not in ("a", "an", "the")]


def contains(gold, text):
    needle, haystack = tokens(gold), tokens(text)
    return bool(needle) and any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def wilson(successes, total, z=1.96):
    if not total:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0


def build_index():
    if store.vector_store is not None:
        return
    chunks = []
    for path in sorted((DATA / "docs").iterdir()):
        chunks += process_document(str(path), path.suffix, path.name, f"general/{path.name}", "general")
    store.vector_store = FAISS.from_documents(chunks, store.embeddings)
    store.index_path.mkdir(parents=True, exist_ok=True)
    store.vector_store.save_local(str(store.index_path))


def retrieval_scores(question):
    if question["category"] == "unanswerable":
        return {}
    started = time.perf_counter()
    docs = store.search(question["question"], top_k=5, role="general")
    elapsed = time.perf_counter() - started
    gold = set(question["gold_sources"])

    def relevant(doc):
        if doc.metadata["source"] not in gold:
            return False
        return not question["evidence"] or any(contains(g, doc.page_content) for g in question["gold"])

    ranks = [i for i, doc in enumerate(docs, 1) if relevant(doc)]
    found = {doc.metadata["source"] for doc in docs if relevant(doc)}
    return {
        "hit_at_1": bool(ranks and ranks[0] == 1),
        "hit_at_5": bool(ranks),
        "all_sources_at_5": found == gold,
        "reciprocal_rank": 1 / ranks[0] if ranks else 0.0,
        "search_seconds": elapsed,
    }


def answer_one(question):
    last_error = None
    for attempt in range(4):
        try:
            started = time.perf_counter()
            result = get_answer(question["question"], role="general")
            return result, time.perf_counter() - started, None
        except BudgetExceeded:
            raise
        except Exception as e:  # Rate limits are not billed, so these can be retried.
            last_error = f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(6 * (attempt + 1))
    return {"answer": "", "sources": []}, 0.0, last_error


def score(question, result):
    answer = result["answer"]
    abstained = bool(ABSTAIN.search(answer))
    if question["category"] == "unanswerable":
        return abstained, abstained
    matched = any(contains(g, answer) for g in question["gold"])
    return matched and not abstained, abstained


def access_control_probe(count=50):
    """Index restricted documents, then try to retrieve them as a general user."""
    secrets = [
        Document(
            page_content=f"Confidential payroll record {i}: employee code ZX{i:03} earns bonus tier {i % 7}.",
            metadata={"source": f"restricted_{i}.txt", "doc_path": f"HR/restricted_{i}.txt", "doc_hash": "x",
                      "role": "HR", "page": 0, "section": "", "chunk_index": 0},
        )
        for i in range(count)
    ]
    store.vector_store.add_documents(secrets)
    leaks = returned_to_hr = 0
    for i in range(count):
        query = f"What bonus tier does employee code ZX{i:03} earn in the confidential payroll record?"
        leaks += sum(d.metadata.get("role") != "general" for d in store.search(query, top_k=5, role="general"))
        returned_to_hr += any(d.metadata["source"] == f"restricted_{i}.txt" for d in store.search(query, top_k=5, role="HR"))
    return {"attempts": count, "leaked_chunks": leaks, "hr_can_still_read": returned_to_hr}


def rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0


def summarize(records, spend, security, model):
    categories = {}
    for category in ("fact", "table", "multi_hop", "unanswerable"):
        rows = [r for r in records if r["category"] == category]
        correct = sum(r["correct"] for r in rows)
        low, high = wilson(correct, len(rows))
        categories[category] = {"questions": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0,
                                "ci95_low": low, "ci95_high": high}
    answerable = [r for r in records if r["category"] != "unanswerable"]
    unanswerable = [r for r in records if r["category"] == "unanswerable"]
    correct = sum(r["correct"] for r in records)
    low, high = wilson(correct, len(records))
    cited = [r for r in answerable if r["sources"]]
    latencies = [r["latency_seconds"] for r in records if not r["error"]]
    return {
        "model": model,
        "questions": len(records),
        "correct": correct,
        "overall_accuracy": correct / len(records),
        "overall_ci95_low": low,
        "overall_ci95_high": high,
        "categories": categories,
        "retrieval": {
            "questions": len(answerable),
            "hit_rate_at_1": rate(answerable, "hit_at_1"),
            "hit_rate_at_5": rate(answerable, "hit_at_5"),
            "all_gold_sources_at_5_multi_hop": rate([r for r in answerable if r["category"] == "multi_hop"], "all_sources_at_5"),
            "mrr": sum(r["reciprocal_rank"] for r in answerable) / len(answerable) if answerable else 0,
            "search_latency_p50_ms": percentile([r["search_seconds"] for r in answerable], 0.5) * 1000,
        },
        "abstention": {
            "correct_refusal_rate": rate(unanswerable, "abstained"),
            "false_refusal_rate": rate(answerable, "abstained"),
        },
        "citations": {
            "any_correct_source_rate": sum(bool(set(r["sources"]) & set(r["gold_sources"])) for r in cited) / len(cited) if cited else 0,
            "precision": sum(len(set(r["sources"]) & set(r["gold_sources"])) / len(r["sources"]) for r in cited) / len(cited) if cited else 0,
        },
        "latency_seconds": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "cost": {"total_usd": spend, "per_question_usd": spend / len(records)},
        "errors": sum(bool(r["error"]) for r in records),
        "access_control": security,
    }


def write_report(summary, records, out):
    pct = lambda v: f"{v * 100:.1f}%"
    names = {"fact": "Single fact (SQuAD v2)", "table": "PDF tables (generated)", "multi_hop": "Multi-hop (HotpotQA)",
             "unanswerable": "No answer exists (SQuAD v2)"}
    lines = [
        "# RAG benchmark results", "",
        f"Model: `{summary['model']}`. Scored automatically against known answers, in one run, with no tuning on these questions.", "",
        f"**Overall accuracy: {pct(summary['overall_accuracy'])}** ({summary['correct']}/{summary['questions']}), "
        f"95% CI {pct(summary['overall_ci95_low'])} to {pct(summary['overall_ci95_high'])}", "",
        "| Question type | Questions | Correct | Accuracy | 95% CI |", "|---|---|---|---|---|",
    ]
    for key, row in summary["categories"].items():
        lines.append(f"| {names[key]} | {row['questions']} | {row['correct']} | {pct(row['accuracy'])} | {pct(row['ci95_low'])} to {pct(row['ci95_high'])} |")
    r, a, c, l, s, x = summary["retrieval"], summary["abstention"], summary["citations"], summary["latency_seconds"], summary["access_control"], summary["cost"]
    lines += [
        "", "| Other metric | Result |", "|---|---|",
        f"| Search hit rate @1 | {pct(r['hit_rate_at_1'])} |",
        f"| Search hit rate @5 | {pct(r['hit_rate_at_5'])} |",
        f"| Both supporting documents in top 5 (multi-hop) | {pct(r['all_gold_sources_at_5_multi_hop'])} |",
        f"| MRR | {r['mrr']:.3f} |",
        f"| Search latency, p50 | {r['search_latency_p50_ms']:.0f} ms |",
        f"| Correct refusal when no answer exists | {pct(a['correct_refusal_rate'])} |",
        f"| False refusal when an answer exists | {pct(a['false_refusal_rate'])} |",
        f"| Answers citing a correct source | {pct(c['any_correct_source_rate'])} |",
        f"| Citation precision | {pct(c['precision'])} |",
        f"| Answer time, p50 / p95 (4 at once) | {l['p50']:.1f}s / {l['p95']:.1f}s |",
        f"| Total cost / per question | ${x['total_usd']:.4f} / ${x['per_question_usd']:.5f} |",
        f"| Restricted chunks leaked to a general user | {s['leaked_chunks']} in {s['attempts']} attempts |",
        f"| Restricted documents still found by HR | {s['hr_can_still_read']}/{s['attempts']} |",
        f"| Requests that errored | {summary['errors']} |",
        "", "Every question, expected answer and system answer is in `results.csv`.", "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "expected", "rag_answer", "correct", "abstained", "sources", "latency_seconds", "error"])
        for r in records:
            writer.writerow([r["id"], r["category"], r["question"], " | ".join(r["gold"]) or "(no answer exists)", r["rag_answer"],
                             r["correct"], r["abstained"], " | ".join(r["sources"]), f"{r['latency_seconds']:.2f}", r["error"] or ""])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=0, help="Use only the first N questions of each type")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    questions = [json.loads(line) for line in open(DATA / "questions.jsonl", encoding="utf-8")]
    if args.per_category:
        seen = {}
        questions = [q for q in questions if seen.setdefault(q["category"], []).append(1) or len(seen[q["category"]]) <= args.per_category]

    started = budget.spent()
    build_index()
    print(f"index ready: {store.vector_store.index.ntotal} chunks; {len(questions)} questions; spent so far ${started:.4f}", flush=True)
    scored = {q["id"]: retrieval_scores(q) for q in questions}

    records, lock = [], threading.Lock()

    def process(question):
        result, latency, error = answer_one(question)
        correct, abstained = score(question, result)
        record = {**question, **scored[question["id"]], "rag_answer": result["answer"], "sources": result["sources"],
                  "correct": bool(correct), "abstained": abstained, "latency_seconds": latency, "error": error}
        with lock:
            records.append(record)
            if len(records) % 25 == 0:
                print(f"{len(records)}/{len(questions)} done, ${budget.spent() - started:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(process, q) for q in questions]:
            future.result()

    records.sort(key=lambda r: r["id"])
    summary = summarize(records, budget.spent() - started, access_control_probe(),
                        f"{active_provider()}/{get_llm().model_name}")
    write_report(summary, records, out)
    print((out / "report.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
