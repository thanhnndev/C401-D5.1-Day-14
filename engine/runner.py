import asyncio
import math
import re
import time
from typing import Any, Dict, List, Optional, cast

from tqdm import tqdm
# Import other components...

class BenchmarkRunner:
    def __init__(
        self,
        agent,
        evaluator,
        judge,
        pricing_per_1k_tokens: Optional[Dict[str, float]] = None,
        enable_ragas: bool = True,
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.enable_ragas = enable_ragas
        # Giá mẫu để ước tính cost. Có thể truyền vào từ bên ngoài để khớp model pricing thực tế.
        self.pricing_per_1k_tokens = pricing_per_1k_tokens or {
            "agent": 0.0005,
            "judge": 0.0015,
        }

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    @classmethod
    def _tokenize(cls, text: Any) -> List[str]:
        normalized = cls._normalize_text(text).lower()
        return re.findall(r"\w+", normalized)

    @staticmethod
    def _jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @classmethod
    def _safe_contexts(cls, response: Dict[str, Any], test_case: Dict[str, Any]) -> List[str]:
        contexts = response.get("contexts")
        if isinstance(contexts, list):
            return [cls._normalize_text(c) for c in contexts if cls._normalize_text(c)]
        if isinstance(contexts, str) and cls._normalize_text(contexts):
            return [cls._normalize_text(contexts)]

        fallback = test_case.get("context")
        if isinstance(fallback, str) and cls._normalize_text(fallback):
            return [cls._normalize_text(fallback)]
        return []

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, tuple):
            return [str(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        return []

    def _extract_expected_ids(self, test_case: Dict[str, Any]) -> List[str]:
        return self._to_str_list(
            test_case.get("expected_retrieval_ids")
            or test_case.get("ground_truth_ids")
            or test_case.get("expected_ids")
        )

    def _extract_retrieved_ids(self, response: Dict[str, Any]) -> List[str]:
        direct = response.get("retrieved_ids")
        if direct is not None:
            return self._to_str_list(direct)

        metadata = response.get("metadata")
        if isinstance(metadata, dict):
            source_ids = metadata.get("source_ids")
            if source_ids is not None:
                return self._to_str_list(source_ids)

            sources = metadata.get("sources")
            if sources is not None:
                return self._to_str_list(sources)

        return []

    def _evaluate_retrieval_metrics(self, test_case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = self._extract_expected_ids(test_case)
        retrieved_ids = self._extract_retrieved_ids(response)

        if (
            self.evaluator
            and hasattr(self.evaluator, "calculate_hit_rate")
            and hasattr(self.evaluator, "calculate_mrr")
        ):
            try:
                hit_rate = float(self.evaluator.calculate_hit_rate(expected_ids, retrieved_ids))
                mrr = float(self.evaluator.calculate_mrr(expected_ids, retrieved_ids))
                return {
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                }
            except Exception:
                pass

        # Fallback local nếu evaluator không expose retrieval methods.
        if not expected_ids or not retrieved_ids:
            return {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            }

        hit_rate = 1.0 if any(doc_id in retrieved_ids[:3] for doc_id in expected_ids) else 0.0
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                mrr = 1.0 / (i + 1)
                break

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
        }

    @staticmethod
    def _extract_total_tokens(response: Dict[str, Any], judge_result: Dict[str, Any]) -> Dict[str, int]:
        response_tokens = 0
        metadata = response.get("metadata")
        if isinstance(metadata, dict):
            response_tokens = int(metadata.get("tokens_used", 0) or 0)

        judge_tokens = 0
        if isinstance(judge_result, dict):
            usage = judge_result.get("usage")
            if isinstance(usage, dict):
                judge_tokens = int(usage.get("total_tokens", 0) or 0)

        total_tokens = response_tokens + judge_tokens
        return {
            "agent_tokens": response_tokens,
            "judge_tokens": judge_tokens,
            "total_tokens": total_tokens,
        }

    def _estimate_cost_usd(self, token_usage: Dict[str, int], judge_result: Dict[str, Any]) -> float:
        agent_tokens = token_usage.get("agent_tokens", 0)
        judge_tokens = token_usage.get("judge_tokens", 0)

        # Nếu judge chưa trả usage thì ước lượng thô theo số judge model đã gọi.
        if judge_tokens == 0:
            individual_scores = {}
            if isinstance(judge_result, dict):
                maybe_scores = judge_result.get("individual_scores")
                if isinstance(maybe_scores, dict):
                    individual_scores = maybe_scores
            judge_calls = len(individual_scores) if individual_scores else 2
            judge_tokens = 250 * judge_calls

        agent_cost = (agent_tokens / 1000.0) * float(self.pricing_per_1k_tokens.get("agent", 0.0))
        judge_cost = (judge_tokens / 1000.0) * float(self.pricing_per_1k_tokens.get("judge", 0.0))
        return max(0.0, agent_cost + judge_cost)

    async def _evaluate_ragas(self, test_case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        question = self._normalize_text(test_case.get("question"))
        answer = self._normalize_text(response.get("answer"))
        contexts = self._safe_contexts(response, test_case)
        ground_truth = self._normalize_text(test_case.get("expected_answer"))

        ragas_result: Dict[str, Any] = {}

        # Ưu tiên dùng evaluator hiện có để tận dụng metric retrieval nếu evaluator đang cung cấp.
        if self.evaluator and hasattr(self.evaluator, "score"):
            try:
                evaluator_scores = await self.evaluator.score(test_case, response)
                if isinstance(evaluator_scores, dict):
                    ragas_result.update(evaluator_scores)
            except Exception:
                pass

        # Thử RAGAS nếu package và runtime phù hợp.
        ragas_scores: Dict[str, float] = {}
        if self.enable_ragas:
            try:
                from datasets import Dataset
                from ragas import evaluate

                try:
                    from ragas.metrics.collections import answer_relevancy, context_precision, faithfulness
                except Exception:
                    from ragas.metrics import answer_relevancy, context_precision, faithfulness

                dataset = Dataset.from_dict(
                    {
                        "question": [question],
                        "answer": [answer],
                        "contexts": [contexts],
                        "ground_truth": [ground_truth],
                    }
                )
                metrics_to_use = [faithfulness, answer_relevancy, context_precision]
                eval_result = evaluate(
                    dataset,
                    metrics=cast(Any, metrics_to_use),
                )

                # Truy xuất kết quả theo cách an toàn để tương thích nhiều phiên bản RAGAS
                # và tránh lỗi static typing khi type stub không khai báo to_pandas.
                score_df = None
                to_pandas_fn = getattr(eval_result, "to_pandas", None)
                if callable(to_pandas_fn):
                    score_df = to_pandas_fn()

                if score_df is not None:
                    score_df_any: Any = score_df
                    faithfulness_score = float(score_df_any["faithfulness"].iloc[0])
                    answer_relevancy_score = float(score_df_any["answer_relevancy"].iloc[0])
                    context_precision_score = float(score_df_any["context_precision"].iloc[0])
                else:
                    # Fallback cho trường hợp eval_result có API dạng dict-like.
                    faithfulness_score = float(
                        getattr(eval_result, "get", lambda *_: [0.0])("faithfulness", [0.0])[0]
                    )
                    answer_relevancy_score = float(
                        getattr(eval_result, "get", lambda *_: [0.0])("answer_relevancy", [0.0])[0]
                    )
                    context_precision_score = float(
                        getattr(eval_result, "get", lambda *_: [0.0])("context_precision", [0.0])[0]
                    )

                ragas_scores = {
                    "faithfulness": faithfulness_score,
                    "answer_relevancy": answer_relevancy_score,
                    "context_precision": context_precision_score,
                }
                ragas_result["metric_source"] = "ragas"
            except Exception:
                # Fallback custom metrics (0-1) nếu RAGAS không khả dụng hoặc lỗi runtime.
                answer_tokens = self._tokenize(answer)
                question_tokens = self._tokenize(question)
                gt_tokens = self._tokenize(ground_truth)
                ctx_tokens = self._tokenize(" ".join(contexts))

                faithful_overlap = self._jaccard_similarity(answer_tokens, ctx_tokens)
                relevancy_overlap = self._jaccard_similarity(answer_tokens, question_tokens)

                if ctx_tokens and gt_tokens:
                    # Đo độ chính xác context với expected answer như proxy cho precision.
                    precision_overlap = len(set(ctx_tokens) & set(gt_tokens)) / len(set(ctx_tokens))
                else:
                    precision_overlap = 0.0

                ragas_scores = {
                    "faithfulness": max(0.0, min(1.0, faithful_overlap)),
                    "answer_relevancy": max(0.0, min(1.0, relevancy_overlap)),
                    "context_precision": max(0.0, min(1.0, precision_overlap)),
                }
                ragas_result["metric_source"] = "custom-fallback"
        else:
            # Fallback custom metrics khi chủ động tắt RAGAS.
            answer_tokens = self._tokenize(answer)
            question_tokens = self._tokenize(question)
            gt_tokens = self._tokenize(ground_truth)
            ctx_tokens = self._tokenize(" ".join(contexts))

            faithful_overlap = self._jaccard_similarity(answer_tokens, ctx_tokens)
            relevancy_overlap = self._jaccard_similarity(answer_tokens, question_tokens)

            if ctx_tokens and gt_tokens:
                # Đo độ chính xác context với expected answer như proxy cho precision.
                precision_overlap = len(set(ctx_tokens) & set(gt_tokens)) / len(set(ctx_tokens))
            else:
                precision_overlap = 0.0

            ragas_scores = {
                "faithfulness": max(0.0, min(1.0, faithful_overlap)),
                "answer_relevancy": max(0.0, min(1.0, relevancy_overlap)),
                "context_precision": max(0.0, min(1.0, precision_overlap)),
            }
            ragas_result["metric_source"] = "custom-fallback-disabled-ragas"

        ragas_result.update(ragas_scores)

        # Đảm bảo 3 metric chính luôn có mặt theo acceptance criteria.
        for key in ["faithfulness", "answer_relevancy", "context_precision"]:
            value = ragas_result.get(key, 0.0)
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                ragas_result[key] = 0.0
            else:
                ragas_result[key] = float(value)

        return ragas_result

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        question = test_case.get("question", "")

        try:
            # 1. Gọi Agent
            response = await self.agent.query(question)

            # 2. Chạy RAGAS metrics (hoặc custom fallback nếu RAGAS lỗi/không khả dụng)
            ragas_scores = await self._evaluate_ragas(test_case, response)

            # 2.1 Chạy retrieval metrics cho hit_rate/mrr
            retrieval_scores = self._evaluate_retrieval_metrics(test_case, response)
            ragas_scores.setdefault("retrieval", retrieval_scores)

            # 3. Chạy Multi-Judge
            judge_result = await self.judge.evaluate_multi_judge(
                question,
                response.get("answer", ""),
                test_case.get("expected_answer", ""),
            )

            # 4. Track token/cost
            token_usage = self._extract_total_tokens(response, judge_result)
            estimated_cost_usd = self._estimate_cost_usd(token_usage, judge_result)
            latency = time.perf_counter() - start_time

            final_score = float(judge_result.get("final_score", 0.0))
            return {
                "test_case": question,
                "agent_response": response.get("answer", ""),
                "latency": latency,
                "ragas": ragas_scores,
                "judge": judge_result,
                "token_usage": token_usage,
                "estimated_cost_usd": estimated_cost_usd,
                "status": "fail" if final_score < 3 else "pass",
            }
        except Exception as exc:
            latency = time.perf_counter() - start_time
            return {
                "test_case": question,
                "agent_response": "",
                "latency": latency,
                "ragas": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "retrieval": {"hit_rate": 0.0, "mrr": 0.0, "expected_ids": [], "retrieved_ids": []},
                    "metric_source": "error-fallback",
                },
                "judge": {"final_score": 0.0, "agreement_rate": 0.0, "reasoning": f"run_single_test error: {exc}"},
                "token_usage": {"agent_tokens": 0, "judge_tokens": 0, "total_tokens": 0},
                "estimated_cost_usd": 0.0,
                "status": "error",
                "error": str(exc),
            }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        Có progress bar và xử lý lỗi từng case để không dừng toàn bộ pipeline.
        """
        results = []
        progress = tqdm(total=len(dataset), desc="Benchmark", unit="case")
        try:
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                tasks = [self.run_single_test(case) for case in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for idx, item in enumerate(batch_results):
                    if isinstance(item, Exception):
                        question = batch[idx].get("question", f"case_{i + idx + 1}")
                        results.append(
                            {
                                "test_case": question,
                                "agent_response": "",
                                "latency": 0.0,
                                "ragas": {
                                    "faithfulness": 0.0,
                                    "answer_relevancy": 0.0,
                                    "context_precision": 0.0,
                                    "retrieval": {
                                        "hit_rate": 0.0,
                                        "mrr": 0.0,
                                        "expected_ids": [],
                                        "retrieved_ids": [],
                                    },
                                    "metric_source": "batch-error-fallback",
                                },
                                "judge": {
                                    "final_score": 0.0,
                                    "agreement_rate": 0.0,
                                    "reasoning": f"batch error: {item}",
                                },
                                "token_usage": {"agent_tokens": 0, "judge_tokens": 0, "total_tokens": 0},
                                "estimated_cost_usd": 0.0,
                                "status": "error",
                                "error": str(item),
                            }
                        )
                    else:
                        results.append(item)
                progress.update(len(batch))
        finally:
            progress.close()
        return results

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "total": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "avg_latency": 0.0,
                "agreement_rate": 0.0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "cost_per_eval": 0.0,
                "cost_per_1k_tokens": 0.0,
                "error_count": 0,
            }

        total = len(results)
        pass_count = sum(1 for r in results if r.get("status") == "pass")
        error_count = sum(1 for r in results if r.get("status") == "error")

        score_sum = 0.0
        hit_sum = 0.0
        mrr_sum = 0.0
        latency_sum = 0.0
        agreement_sum = 0.0
        total_tokens = 0
        total_cost = 0.0

        for r in results:
            judge = r.get("judge") or {}
            ragas = r.get("ragas") or {}
            retrieval = ragas.get("retrieval") or {}
            token_usage = r.get("token_usage") or {}

            score_sum += float(judge.get("final_score", 0.0) or 0.0)
            agreement_sum += float(judge.get("agreement_rate", 0.0) or 0.0)
            hit_sum += float(retrieval.get("hit_rate", 0.0) or 0.0)
            mrr_sum += float(retrieval.get("mrr", 0.0) or 0.0)
            latency_sum += float(r.get("latency", 0.0) or 0.0)
            total_tokens += int(token_usage.get("total_tokens", 0) or 0)
            total_cost += float(r.get("estimated_cost_usd", 0.0) or 0.0)

        cost_per_eval = total_cost / total if total else 0.0
        cost_per_1k_tokens = (total_cost / (total_tokens / 1000.0)) if total_tokens > 0 else 0.0

        return {
            "total": total,
            "pass_rate": pass_count / total,
            "avg_score": score_sum / total,
            "avg_hit_rate": hit_sum / total,
            "avg_mrr": mrr_sum / total,
            "avg_latency": latency_sum / total,
            "agreement_rate": agreement_sum / total,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "cost_per_eval": cost_per_eval,
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "error_count": error_count,
        }
