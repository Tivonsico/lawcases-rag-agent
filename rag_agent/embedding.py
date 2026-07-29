"""
Embedding 模块
- API 地址和 key 由用户在 config 中填写
- 维度可配置，默认 1024
- 提供 mock 模式用于离线测试
"""
import json
import time
import requests
import numpy as np
from typing import List, Optional

try:
    from .config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM
except ImportError:  # pragma: no cover - direct script compatibility
    from config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM


class EmbeddingService:
    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        dim: int = None,
        model: str = None,
        mock: bool = False,
    ):
        self.api_url = api_url or EMBEDDING_API_URL
        self.api_key = api_key or EMBEDDING_API_KEY
        self.dim = dim or EMBEDDING_DIM
        self.model = model or EMBEDDING_MODEL
        self.mock = mock or (not self.api_url)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转为向量
        返回 shape = (len(texts), dim) 的列表
        """
        if not texts:
            return []

        if self.mock:
            return self._mock_embed(texts)

        if not self.api_url:
            raise ValueError(
                "请先在 .env 文件中配置 EMBEDDING_API_URL 和 EMBEDDING_API_KEY"
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dim,
        }

        for attempt in range(2):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=(5, 15)
                )
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"][:self.dim] for item in data["data"]]
            except requests.RequestException as exc:
                if attempt == 0:
                    print(f"[WARN] Embedding API 第一次调用失败，正在重试: {exc}")
                    time.sleep(1)
                    continue
                print(f"[ERROR] Embedding API 调用失败: {exc}")
                if not self.mock:
                    raise
        print(f"[WARN] 使用 mock embedding 作为回退")
        return self._mock_embed(texts)

    def embed_one(self, text: str) -> List[float]:
        """单个文本向量化"""
        result = self.embed([text])
        return result[0] if result else [0.0] * self.dim

    def _mock_embed(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding：用文本hash生成伪向量（仅用于离线测试）"""
        rng = np.random.RandomState(42)
        # 用固定种子 + 文本hash 保证同一文本向量一致
        result = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # 归一化
            result.append(vec.tolist())
        return result

    @property
    def dimension(self) -> int:
        return self.dim


if __name__ == "__main__":
    emb = EmbeddingService(mock=True)
    vec = emb.embed(["测试文本", "保险知识"])
    print(f"向量维度: {len(vec[0])}")
    print(f"向量数: {len(vec)}")
