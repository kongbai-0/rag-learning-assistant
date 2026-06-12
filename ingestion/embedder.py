# -*- coding: utf-8 -*-
import dashscope
from dashscope import TextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL

dashscope.api_key = DASHSCOPE_API_KEY


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def get_embedding(text: str):
    resp = TextEmbedding.call(
        model=EMBEDDING_MODEL,
        input=text,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Embedding API error: {resp.code} {resp.message}")

    return resp["output"]["embeddings"][0]["embedding"]
