from typing import Any, Dict, List

class RetrievalEvaluator:
    def __init__(self):
        pass

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

    def _extract_expected_ids(self, case: Dict[str, Any]) -> List[str]:
        # Hỗ trợ cả format cũ và format dataset mới.
        return self._to_str_list(
            case.get("expected_retrieval_ids")
            or case.get("ground_truth_ids")
            or case.get("expected_ids")
        )

    def _extract_retrieved_ids(self, case: Dict[str, Any]) -> List[str]:
        direct = case.get("retrieved_ids")
        if direct is not None:
            return self._to_str_list(direct)

        retrieval = case.get("retrieval")
        if isinstance(retrieval, dict):
            return self._to_str_list(retrieval.get("retrieved_ids"))

        response = case.get("response")
        if isinstance(response, dict):
            return self._to_str_list(response.get("retrieved_ids"))

        return []

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        if top_k <= 0:
            return 0.0
        if not expected_ids or not retrieved_ids:
            return 0.0

        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Reciprocal Rank cho một query: 1/rank của expected_id đầu tiên trong retrieved_ids.
        Nếu không có expected_id nào xuất hiện thì trả về 0.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy retrieval eval cho toàn bộ dataset.
        Hỗ trợ:
        - expected IDs: expected_retrieval_ids hoặc ground_truth_ids
        - retrieved IDs: retrieved_ids hoặc retrieval.retrieved_ids hoặc response.retrieved_ids
        """
        if not dataset:
            return {
                "total_cases": 0,
                "evaluated_cases": 0,
                "skipped_cases": 0,
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "per_query": [],
            }

        per_query: List[Dict[str, Any]] = []
        hit_sum = 0.0
        mrr_sum = 0.0
        evaluated = 0
        skipped = 0

        for idx, case in enumerate(dataset):
            expected_ids = self._extract_expected_ids(case)
            retrieved_ids = self._extract_retrieved_ids(case)

            question = case.get("question") or case.get("test_case") or f"case_{idx + 1}"

            if not expected_ids:
                skipped += 1
                per_query.append(
                    {
                        "index": idx,
                        "question": question,
                        "expected_ids": expected_ids,
                        "retrieved_ids": retrieved_ids,
                        "hit_rate": None,
                        "mrr": None,
                        "evaluated": False,
                        "reason": "missing_expected_ids",
                    }
                )
                continue

            hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            hit_sum += hit_rate
            mrr_sum += mrr
            evaluated += 1

            per_query.append(
                {
                    "index": idx,
                    "question": question,
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "evaluated": True,
                }
            )

        avg_hit = hit_sum / evaluated if evaluated else 0.0
        avg_mrr = mrr_sum / evaluated if evaluated else 0.0

        return {
            "total_cases": len(dataset),
            "evaluated_cases": evaluated,
            "skipped_cases": skipped,
            "avg_hit_rate": avg_hit,
            "avg_mrr": avg_mrr,
            "per_query": per_query,
        }
