import asyncio
import math
import re
import time
from typing import Any, Dict, List
# Import other components...

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

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
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            dataset = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth],
                }
            )
            eval_result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
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
                faithfulness_score = float(getattr(eval_result, "get", lambda *_: [0.0])("faithfulness", [0.0])[0])
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
        
        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time
        
        # 2. Chạy RAGAS metrics (hoặc custom fallback nếu RAGAS lỗi/không khả dụng)
        ragas_scores = await self._evaluate_ragas(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"], 
            response["answer"], 
            test_case.get("expected_answer", "")
        )
        
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
