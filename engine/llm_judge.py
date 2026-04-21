import asyncio
import json
import os
import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

class LLMJudge:
    def __init__(
        self,
        model: Optional[str] = None,
        second_model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
    ):
        load_dotenv()

        primary = model or os.getenv("QWEN_MODEL_NAME_JUDGE") or "qwen3.6-plus"
        secondary = second_model or os.getenv("QWEN_MODEL_NAME_AGENT") or primary

        self.model_a = primary
        self.model_b = secondary
        self.timeout = timeout

        key = api_key if api_key is not None else os.getenv("QWEN_API_KEY")
        base_url = endpoint if endpoint is not None else os.getenv("DASHBOARD_ENDPOINT")

        self.client: Optional[AsyncOpenAI] = None
        if key:
            client_kwargs: Dict[str, Any] = {"api_key": key, "timeout": timeout}
            if base_url:
                client_kwargs["base_url"] = base_url.rstrip("/")
            self.client = AsyncOpenAI(**client_kwargs)

        # Rubrics chấm điểm có trọng số rõ ràng để 2 model đánh giá nhất quán.
        self.rubrics = {
            "accuracy": {"weight": 0.30, "desc": "Độ chính xác so với ground truth"},
            "completeness": {"weight": 0.20, "desc": "Mức độ đầy đủ, bao quát ý chính"},
            "relevance": {"weight": 0.20, "desc": "Mức độ liên quan trực tiếp tới câu hỏi"},
            "safety": {"weight": 0.15, "desc": "Không bịa đặt hoặc đưa thông tin không có căn cứ"},
            "tone": {"weight": 0.15, "desc": "Văn phong rõ ràng, chuyên nghiệp"},
        }

    @staticmethod
    def _clamp_score(score: float) -> float:
        return max(1.0, min(5.0, float(score)))

    @staticmethod
    def _agreement_rate(score_a: float, score_b: float) -> float:
        # Chuẩn hoá chênh lệch trên thang 1-5 thành [0, 1].
        diff = abs(score_a - score_b)
        return max(0.0, 1.0 - (diff / 4.0))

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _build_judge_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        rubric_lines = [
            f"- {name}: weight={spec['weight']:.2f}, {spec['desc']}"
            for name, spec in self.rubrics.items()
        ]
        rubric_block = "\n".join(rubric_lines)

        return (
            "You are an evaluation judge for a RAG assistant. Score from 1.0 to 5.0.\n"
            "Use this weighted rubric:\n"
            f"{rubric_block}\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            '  "score": <float 1.0-5.0>,\n'
            '  "criterion_scores": {"accuracy": <float>, "completeness": <float>, "relevance": <float>, "safety": <float>, "tone": <float>},\n'
            '  "reasoning": "<short reason>"\n'
            "}\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Ground truth: {ground_truth}\n"
        )

    async def _judge_with_model(self, model_name: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        if self.client is None:
            return self._fallback_judge(model_name, question, answer, ground_truth)

        prompt = self._build_judge_prompt(question, answer, ground_truth)

        try:
            resp = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a strict but fair QA evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=350,
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            parsed = self._extract_json_block(raw or "")

            score = self._clamp_score(parsed.get("score", 3.0))
            return {
                "model": model_name,
                "score": score,
                "criterion_scores": parsed.get("criterion_scores", {}),
                "reasoning": parsed.get("reasoning", "No reasoning provided."),
                "raw": raw,
            }
        except Exception as exc:
            fallback = self._fallback_judge(model_name, question, answer, ground_truth)
            fallback["reasoning"] = f"Fallback scoring due to API error: {exc}"
            return fallback

    def _fallback_judge(self, model_name: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        # Heuristic offline scorer để pipeline không bị block nếu thiếu API/config.
        q_words = set(re.findall(r"\w+", question.lower()))
        a_words = set(re.findall(r"\w+", answer.lower()))
        gt_words = set(re.findall(r"\w+", ground_truth.lower()))

        overlap_gt = (len(a_words & gt_words) / len(gt_words)) if gt_words else 0.0
        overlap_q = (len(a_words & q_words) / len(q_words)) if q_words else 0.0
        hallucination_penalty = 0.3 if "không biết" in answer.lower() or "i don't know" in answer.lower() else 0.0

        score = 1.0 + (2.5 * overlap_gt) + (1.8 * overlap_q) - hallucination_penalty
        score = self._clamp_score(score)

        return {
            "model": model_name,
            "score": score,
            "criterion_scores": {
                "accuracy": round(self._clamp_score(1.0 + 4.0 * overlap_gt), 2),
                "completeness": round(self._clamp_score(1.0 + 4.0 * overlap_gt), 2),
                "relevance": round(self._clamp_score(1.0 + 4.0 * overlap_q), 2),
                "safety": round(self._clamp_score(4.2 - hallucination_penalty), 2),
                "tone": 4.0,
            },
            "reasoning": "Heuristic fallback scoring based on lexical overlap.",
            "raw": None,
        }

    @staticmethod
    def _resolve_final_score(score_a: float, score_b: float) -> Tuple[float, bool, float]:
        diff = abs(score_a - score_b)
        conflict = diff > 0.5
        if conflict:
            return float(median([score_a, score_b])), conflict, diff
        return (score_a + score_b) / 2.0, conflict, diff

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Chấm bởi 2 model độc lập và hợp nhất kết quả theo consensus logic.
        """
        judge_a_task = self._judge_with_model(self.model_a, question, answer, ground_truth)
        judge_b_task = self._judge_with_model(self.model_b, question, answer, ground_truth)
        judge_a, judge_b = await asyncio.gather(judge_a_task, judge_b_task)

        score_a = float(judge_a["score"])
        score_b = float(judge_b["score"])
        final_score, conflict, score_diff = self._resolve_final_score(score_a, score_b)
        agreement = self._agreement_rate(score_a, score_b)

        return {
            "final_score": round(final_score, 4),
            "conflict": conflict,
            "score_diff": round(score_diff, 4),
            "agreement_rate": agreement,
            "individual_scores": {
                self.model_a: round(score_a, 4),
                self.model_b: round(score_b, 4),
            },
            "individual_reasons": {
                self.model_a: judge_a.get("reasoning", ""),
                self.model_b: judge_b.get("reasoning", ""),
            },
            "reasoning": (
                f"{self.model_a}={score_a:.2f}, {self.model_b}={score_b:.2f}, "
                f"agreement={agreement:.2f}, conflict={'yes' if conflict else 'no'}."
            ),
        }

    async def check_position_bias(
        self,
        response_a: str,
        response_b: str,
        question: str = "Which answer is better?",
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        """
        Đổi vị trí A/B để đo mức thiên vị vị trí của judge.
        """
        prompt_ab = (
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Answer A: {response_a}\n"
            f"Answer B: {response_b}\n"
            "Return ONLY JSON: {\"winner\": \"A\"|\"B\"|\"tie\", \"confidence\": <0-1>, \"reasoning\": \"...\"}"
        )
        prompt_ba = (
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Answer A: {response_b}\n"
            f"Answer B: {response_a}\n"
            "Return ONLY JSON: {\"winner\": \"A\"|\"B\"|\"tie\", \"confidence\": <0-1>, \"reasoning\": \"...\"}"
        )

        async def pick(prompt: str) -> Dict[str, Any]:
            if self.client is None:
                winner = "A" if len(response_a) >= len(response_b) else "B"
                return {"winner": winner, "confidence": 0.5, "reasoning": "Offline fallback."}

            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_a,
                    messages=[
                        {"role": "system", "content": "You compare two answers objectively."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=180,
                )
                raw = resp.choices[0].message.content if resp.choices else ""
                parsed = self._extract_json_block(raw or "")
                winner = parsed.get("winner", "tie")
                confidence = float(parsed.get("confidence", 0.5))
                return {
                    "winner": winner if winner in {"A", "B", "tie"} else "tie",
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reasoning": parsed.get("reasoning", "No reasoning provided."),
                }
            except Exception as exc:
                return {"winner": "tie", "confidence": 0.0, "reasoning": f"Error: {exc}"}

        direct, swapped = await asyncio.gather(pick(prompt_ab), pick(prompt_ba))

        # Nếu direct chọn A thì swapped nên chọn B (vì thứ tự đảo).
        mapped_swapped_winner = (
            "A" if swapped["winner"] == "B" else "B" if swapped["winner"] == "A" else "tie"
        )
        consistent = direct["winner"] == mapped_swapped_winner
        bias_score = 0.0 if consistent else 1.0

        return {
            "direct": direct,
            "swapped": swapped,
            "mapped_swapped_winner": mapped_swapped_winner,
            "position_bias_score": bias_score,
            "is_position_biased": not consistent,
        }
