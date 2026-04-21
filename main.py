import asyncio
import json
import os
import time
from typing import Any, Dict, Tuple

from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent

# Giả lập các components Expert
class ExpertEvaluator:
    async def score(self, case, resp): 
        # Giả lập tính toán Hit Rate và MRR
        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
        }

class MultiModelJudge:
    async def evaluate_multi_judge(self, q, a, gt): 
        return {
            "final_score": 4.5, 
            "agreement_rate": 0.8,
            "reasoning": "Cả 2 model đồng ý đây là câu trả lời tốt."
        }


DEFAULT_RELEASE_THRESHOLDS = {
    "min_avg_score": 3.0,
    "min_hit_rate": 0.7,
    "max_hallucination_rate": 0.15,
    "max_latency": 5.0,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_release_thresholds() -> Dict[str, float]:
    thresholds = dict(DEFAULT_RELEASE_THRESHOLDS)

    # Ưu tiên config JSON để dễ override trong CI/CD.
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

    # Fallback theo biến môi trường đơn lẻ.
    env_map = {
        "min_avg_score": "MIN_AVG_SCORE",
        "min_hit_rate": "MIN_HIT_RATE",
        "max_hallucination_rate": "MAX_HALLUCINATION_RATE",
        "max_latency": "MAX_LATENCY",
    }
    for key, env_key in env_map.items():
        if os.getenv(env_key):
            thresholds[key] = _safe_float(os.getenv(env_key), thresholds[key])

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
    v1_cost = _safe_float(v1_metrics.get("cost_usd"))
    v2_cost = _safe_float(v2_metrics.get("cost_usd"))
    v2_faithfulness = _safe_float(v2_metrics.get("faithfulness"), default=-1.0)

    checks = [
        {
            "name": "score_improved",
            "passed": v2_score > v1_score,
            "detail": f"avg_score v2 ({v2_score:.4f}) > v1 ({v1_score:.4f})",
        },
        {
            "name": "hit_rate_not_degraded",
            "passed": v2_hit >= v1_hit,
            "detail": f"hit_rate v2 ({v2_hit:.4f}) >= v1 ({v1_hit:.4f})",
        },
        {
            "name": "latency_within_5_percent",
            "passed": v2_latency <= (v1_latency * 1.05 if v1_latency > 0 else v2_latency),
            "detail": f"avg_latency v2 ({v2_latency:.4f}) <= 1.05 * v1 ({v1_latency:.4f})",
        },
        {
            "name": "cost_reduced",
            "passed": v2_cost < v1_cost if v1_cost > 0 else True,
            "detail": f"cost_usd v2 ({v2_cost:.6f}) < v1 ({v1_cost:.6f})",
        },
        {
            "name": "threshold_min_avg_score",
            "passed": v2_score >= thresholds["min_avg_score"],
            "detail": f"avg_score v2 ({v2_score:.4f}) >= min_avg_score ({thresholds['min_avg_score']:.4f})",
        },
        {
            "name": "threshold_min_hit_rate",
            "passed": v2_hit >= thresholds["min_hit_rate"],
            "detail": f"hit_rate v2 ({v2_hit:.4f}) >= min_hit_rate ({thresholds['min_hit_rate']:.4f})",
        },
        {
            "name": "threshold_max_latency",
            "passed": v2_latency <= thresholds["max_latency"],
            "detail": f"avg_latency v2 ({v2_latency:.4f}) <= max_latency ({thresholds['max_latency']:.4f})",
        },
    ]

    # Nếu có faithfulness thì kiểm thêm hallucination gate, nếu chưa có thì skip.
    if v2_faithfulness >= 0:
        hallucination_rate = max(0.0, 1.0 - v2_faithfulness)
        checks.append(
            {
                "name": "threshold_max_hallucination_rate",
                "passed": hallucination_rate <= thresholds["max_hallucination_rate"],
                "detail": (
                    f"hallucination_rate ({hallucination_rate:.4f}) <= "
                    f"max_hallucination_rate ({thresholds['max_hallucination_rate']:.4f})"
                ),
            }
        )

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

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    max_cases_env = os.getenv("BENCHMARK_MAX_CASES", "").strip()
    if max_cases_env:
        max_cases = max(1, int(max_cases_env))
        dataset = dataset[:max_cases]
        print(f"ℹ️ Giới hạn benchmark theo BENCHMARK_MAX_CASES={max_cases}")

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)
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
            "faithfulness": (
                sum(_safe_float(r.get("ragas", {}).get("faithfulness", 0.0)) for r in results) / total
                if total
                else 0.0
            ),
            "answer_relevancy": (
                sum(_safe_float(r.get("ragas", {}).get("answer_relevancy", 0.0)) for r in results) / total
                if total
                else 0.0
            ),
            "context_precision": (
                sum(_safe_float(r.get("ragas", {}).get("context_precision", 0.0)) for r in results) / total
                if total
                else 0.0
            ),
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary or v1_results is None or v2_results is None:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
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
            "v1_version": v1_summary["metadata"].get("version", "Agent_V1_Base"),
            "v2_version": v2_summary["metadata"].get("version", "Agent_V2_Optimized"),
            "v1_total": len(v1_results),
            "v2_total": len(v2_results),
        },
        "v1_results": v1_results,
        "v2_results": v2_results,
    }
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results_payload, f, ensure_ascii=False, indent=2)

    # Verify output format ngay sau khi generate reports.
    try:
        from check_lab import validate_lab

        print("\n🧪 Kiểm tra định dạng nộp bài bằng check_lab.py...")
        validate_lab()
    except Exception as exc:
        print(f"⚠️ Không thể chạy validate_lab tự động: {exc}")

    if release_gate["passed"]:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")
        print(f"Lý do: {release_gate['reasoning']}")

if __name__ == "__main__":
    asyncio.run(main())
