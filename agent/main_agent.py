import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from fastembed.sparse.bm25 import Bm25

load_dotenv()

class MainAgent:
    """
    RAG Agent chuyên biệt cho tư vấn thiết bị y tế.
    Sử dụng Hybrid Search (Dense + BM25) trên Qdrant và Qwen LLM.
    """

    def __init__(self):
        # Config từ .env
        self.qdrant_url = os.environ["QRANT_URL"]
        self.qdrant_api_key = os.environ["QRANT_API_KEY"]
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "products")
        
        self.jina_api_key = os.environ["EMBEDDING_API_KEY"].strip("'\"")
        self.jina_embed_url = "https://api.jina.ai/v1/embeddings"
        self.jina_rerank_url = "https://api.jina.ai/v1/rerank"
        self.jina_model = os.environ["EMBEDDING_MODEL_NAME"].strip("'\"")
        self.rerank_model = os.environ.get("RERANKL_MODEL_NAME", "jina-reranker-v3").strip("'\"")
        
        self.qwen_api_key = os.environ["QWEN_API_KEY"].strip("'\"")
        self.qwen_model = os.environ["QWEN_MODEL_NAME_AGENT"].strip("'\"")
        self.qwen_api_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

        # Clients & Encoders
        self.qdrant_client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.bm25_encoder = Bm25("Qdrant/bm25") # Init local BM25 model
        
    async def _get_dense_embedding(self, text: str, http: httpx.AsyncClient) -> List[float]:
        resp = await http.post(
            self.jina_embed_url,
            headers={"Authorization": f"Bearer {self.jina_api_key}", "Content-Type": "application/json"},
            json={"model": self.jina_model, "input": [text], "dimensions": 256},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def _get_sparse_embedding(self, text: str) -> models.SparseVector:
        # fastembed trả về generator
        emb = list(self.bm25_encoder.embed([text]))[0]
        return models.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist()
        )

    async def _rerank(self, query: str, documents: List[str], http: httpx.AsyncClient) -> List[int]:
        """Trả về list các index đã được sắp xếp lại."""
        if not documents:
            return []
        resp = await http.post(
            self.jina_rerank_url,
            headers={"Authorization": f"Bearer {self.jina_api_key}", "Content-Type": "application/json"},
            json={
                "model": self.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents)
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return [r["index"] for r in results]

    async def retrieve(self, question: str, http: httpx.AsyncClient, top_k: int = 5, use_rerank: bool = False) -> List[Dict]:
        # 1. Embeddings
        dense_vec = await self._get_dense_embedding(question, http)
        sparse_vec = self._get_sparse_embedding(question)

        # 2. Hybrid Search with RRF Fusion
        # Chúng ta có thể thêm filter nếu phát hiện keyword hãng trong question
        query_filter = None
        for brand in ["melag", "bvi", "schwind"]:
            if brand in question.lower():
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="brand", match=models.MatchValue(value=brand))]
                )
                break

        points = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=top_k * 2, filter=query_filter),
                models.Prefetch(query=sparse_vec, using="bm25", limit=top_k * 2, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k * 2 if use_rerank else top_k,
            with_payload=True
        )

        results = [
            {
                "id": p.payload.get("chunk_id"),
                "text": p.payload.get("text"),
                "metadata": {k: v for k, v in p.payload.items() if k not in ["text", "chunk_id"]}
            }
            for p in points.points
        ]

        # 3. Optional Rerank
        if use_rerank and results:
            texts = [r["text"] for r in results]
            indices = await self._rerank(question, texts, http)
            results = [results[i] for i in indices[:top_k]]

        return results

    async def generate(self, question: str, contexts: List[str], http: httpx.AsyncClient) -> Dict:
        context_str = "\n\n".join([f"--- Context {i+1} ---\n{ctx}" for i, ctx in enumerate(contexts)])
        
        system_prompt = """Bạn là chuyên viên tư vấn kỹ thuật cao cấp của một công ty phân phối thiết bị y tế hàng đầu (đối tác của Melag, BVI, Schwind).
Nhiệm vụ của bạn là cung cấp thông tin chính xác, chuyên nghiệp và hữu ích dựa trên tài liệu kỹ thuật được cung cấp.

HƯỚNG DẪN:
1. Chỉ sử dụng thông tin trong phần CONTEXT để trả lời. 
2. Nếu thông tin không có trong CONTEXT, hãy lịch sự từ chối và đề nghị khách hàng liên hệ bộ phận kỹ thuật để được hỗ trợ chuyên sâu.
3. Luôn giữ thái độ chuyên nghiệp, tin cậy.
4. Trình bày rõ ràng, sử dụng bullet points nếu cần thiết.
5. Nếu câu hỏi yêu cầu so sánh, hãy nêu rõ ưu nhược điểm dựa trên thông số kỹ thuật."""

        user_content = f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"

        resp = await http.post(
            self.qwen_api_url,
            headers={
                "Authorization": f"Bearer {self.qwen_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.qwen_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.1 # Thấp để đảm bảo độ chính xác y tế
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        
        return {
            "answer": data["choices"][0]["message"]["content"],
            "model": self.qwen_model,
            "tokens_used": data["usage"]["total_tokens"]
        }

    async def query(self, question: str, use_rerank: bool = False) -> Dict:
        """
        Quy trình RAG: Retrieve -> (Rerank) -> Generate
        """
        async with httpx.AsyncClient() as http:
            # 1. Retrieval
            retrieved_items = await self.retrieve(question, http, use_rerank=use_rerank)
            contexts = [item["text"] for item in retrieved_items]
            source_ids = [item["id"] for item in retrieved_items]

            # 2. Generation
            gen_result = await self.generate(question, contexts, http)

            return {
                "answer": gen_result["answer"],
                "contexts": contexts,
                "retrieved_ids": source_ids, # Trả về để đánh giá Hit Rate/MRR
                "metadata": {
                    "model": gen_result["model"],
                    "tokens_used": gen_result["tokens_used"],
                    "sources": [item["metadata"].get("product", "Unknown") for item in retrieved_items],
                    "source_ids": source_ids
                },
            }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = MainAgent()
        print("Đang test Agent...")
        res = await agent.query("Máy hấp tiệt trùng Melag có những dòng nào?")
        print(f"\nCâu hỏi: Máy hấp tiệt trùng Melag có những dòng nào?")
        print(f"Trả lời: {res['answer']}")
        print(f"\nSources: {res['metadata']['sources']}")

    asyncio.run(main())
