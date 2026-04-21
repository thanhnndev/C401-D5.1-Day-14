import asyncio
import json
import os
import time
from typing import Any, Dict, Tuple

from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator

# Real components
expert_evaluator = RetrievalEvaluator()
multi_model_judge = LLMJudge()

DEFAULT_RELEASE_THRESHOLDS = {
    "min_avg_score": 3.0,
    "min_hit_rate": 0.7,
    "max_hallucination_rate": 0.15,
    "max_latency": 15.0, # Tăng thêm vì RAG + Rerank + LLM
}

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None: return default
        return float(value)
    except Exception:
        return default

def _load_release_thresholds() -> Dict[str, float]:
    thresholds = dict(DEFAULT_RELEASE_THRESHOLDS)
    raw_json = os.getenv("RELEASE_THRESHOLDS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for k in thresholds:
                    if k in parsed:
                        thresholds[k] = _safe_float(parsed[k], thresholds[k])
        except Exception:
            pass
    return thresholds

def evaluate_release_gate(
    v1_metrics: Dict[str, Any],
    v2_metrics: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    v1_score = _safe_float(v1_metrics.get("avg_score"))
    v2_score = _safe_float(v2_metrics.get("avg_score"))
    v1_hit = _safe_float(v1_metrics.get("hit_rate"))
    v2_hit = _safe_float(v2_metrics.get("hit_rate"))
    v1_latency = _safe_float(v1_metrics.get("avg_latency"))
    v2_latency = _safe_float(v2_metrics.get("avg_latency"))
    
    checks = [
        {
            "name": "score_improved",
            "passed": v2_score >= v1_score,
            "detail": f"avg_score v2 ({v2_score:.4f}) >= v1 ({v1_score:.4f})",
        },
        {
            "name": "hit_rate_not_degraded",
            "passed": v2_hit >= v1_hit,
            "detail": f"hit_rate v2 ({v2_hit:.4f}) >= v1 ({v1_hit:.4f})",
        },
        {
            "name": "threshold_min_avg_score",
            "passed": v2_score >= thresholds["min_avg_score"],
            "detail": f"avg_score v2 ({v2_score:.4f}) >= min_avg_score ({thresholds['min_avg_score']:.4f})",
        },
    ]

    failed = [c for c in checks if not c["passed"]]
    decision = "APPROVE" if not failed else "BLOCK RELEASE"

    return {
        "decision": decision,
        "passed": len(failed) == 0,
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": failed,
        "reasoning": "All gates passed." if not failed else "; ".join(c["detail"] for c in failed),
    }

async def run_benchmark_with_results(agent_version: str, use_rerank: bool = False):
    print(f"🚀 Khởi động Benchmark cho {agent_version} (use_rerank={use_rerank})...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    max_cases_env = os.getenv("BENCHMARK_MAX_CASES", "5").strip() # Mặc định 5 cases cho demo
    if max_cases_env:
        max_cases = max(1, int(max_cases_env))
        dataset = dataset[:max_cases]
        print(f"ℹ️ Giới hạn benchmark theo BENCHMARK_MAX_CASES={max_cases}")

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None

    agent = MainAgent()
    # Patch agent.query để truyền use_rerank
    original_query = agent.query
    async def patched_query(question: str):
        return await original_query(question, use_rerank=use_rerank)
    agent.query = patched_query

    runner = BenchmarkRunner(agent, expert_evaluator, multi_model_judge)
    results = await runner.run_all(dataset, batch_size=2)
    agg = runner.calculate_metrics(results)

    total = len(results)
    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "total_cases": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": agg["avg_score"],
            "hit_rate": agg["avg_hit_rate"],
            "mrr": agg["avg_mrr"],
            "agreement_rate": agg["agreement_rate"],
            "avg_latency": agg["avg_latency"],
            "total_tokens": agg["total_tokens"],
            "cost_usd": agg["total_cost_usd"],
            "pass_rate": agg["pass_rate"],
            "error_count": agg["error_count"],
            "faithfulness": sum(_safe_float(r.get("ragas", {}).get("faithfulness", 0.0)) for r in results) / total if total else 0.0,
            "answer_relevancy": sum(_safe_float(r.get("ragas", {}).get("answer_relevancy", 0.0)) for r in results) / total if total else 0.0,
            "context_precision": sum(_safe_float(r.get("ragas", {}).get("context_precision", 0.0)) for r in results) / total if total else 0.0,
        }
    }
    return results, summary

async def main():
    # V1: Baseline (No Rerank)
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", use_rerank=False)
    
    # V2: Optimized (With Rerank)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", use_rerank=True)

    if not v1_summary or not v2_summary:
        print("❌ Lỗi benchmark.")
        return

    thresholds = _load_release_thresholds()
    release_gate = evaluate_release_gate(v1_summary["metrics"], v2_summary["metrics"], thresholds)

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.4f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.4f}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"Release Gate: {release_gate['decision']}")

    v2_summary["regression"] = {
        "v1_score": v1_summary["metrics"]["avg_score"],
        "v2_score": v2_summary["metrics"]["avg_score"],
        "delta": delta,
        "decision": release_gate["decision"],
        "reasoning": release_gate["reasoning"],
    }
    v2_summary["release_gate"] = release_gate

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)

    benchmark_results_payload = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "v1_version": "Agent_V1_Base",
            "v2_version": "Agent_V2_Optimized",
        },
        "v1_results": v1_results,
        "v2_results": v2_results,
    }
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results_payload, f, ensure_ascii=False, indent=2)

    print("\n✅ Benchmark hoàn tất. Reports tại thư mục reports/")

if __name__ == "__main__":
    asyncio.run(main())
