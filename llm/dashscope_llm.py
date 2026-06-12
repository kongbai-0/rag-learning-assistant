# -*- coding: utf-8 -*-
from dashscope import Generation
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DASHSCOPE_API_KEY, LLM_MODEL
import dashscope

dashscope.api_key = DASHSCOPE_API_KEY

SYSTEM_PROMPT = """你是基于RAG的大学课程学习助手。请严格根据提供的课程资料回答问题。

规则：
1. 如果参考文档包含答案，请引用原文关键句，然后给出通俗易懂的解释。
2. 如果参考文档与问题无关，或无法覆盖问题，请明确说"当前课程资料中未包含这部分内容"。
3. 回答末尾请注明引用的来源，格式：[来源: 片段N - 文件名 - 页码]。
4. 解释时尽量使用大学生容易理解的语言，善用举例和类比。"""

# 无相关内容时的标准回复（不调用 LLM，避免自由发挥）
NO_CONTENT_REPLY = "未在当前课程资料中找到相关内容。\n\n建议：\n- 检查是否选择了正确的课程\n- 尝试用不同的关键词提问\n- 上传更多相关课程资料"


def build_prompt(query, docs, metas):
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        src = meta.get("source", "unknown")
        page = meta.get("page", 0)
        section = meta.get("section", "")

        header = f"[片段{i} - {src}"
        if section:
            header += f" - {section}"
        if page:
            header += f" - 第{page}页"
        header += "]"

        context_parts.append(f"{header}\n{doc}")
    context = "\n\n".join(context_parts)

    return f"参考文档：\n{context}\n\n问题：{query}\n\n请根据参考文档回答上述问题。"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def generate(query, docs, metas=None):
    """非流式生成（兼容旧接口）。"""
    if not docs:
        return NO_CONTENT_REPLY

    if metas is None:
        metas = [{}] * len(docs)

    user_prompt = build_prompt(query, docs, metas)

    resp = Generation.call(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    if resp.status_code != 200:
        raise RuntimeError(f"LLM API error: {resp.code} {resp.message}")

    return resp["output"]["text"]


def generate_stream(query, docs, metas=None):
    """
    流式生成（逐字输出）。

    使用 DashScope Generation API 的 stream 模式，
    每次 yield 一段增量文本。

    用法：
      for chunk in generate_stream(query, docs, metas):
          print(chunk, end="", flush=True)
    """
    if not docs:
        yield NO_CONTENT_REPLY
        return

    if metas is None:
        metas = [{}] * len(docs)

    user_prompt = build_prompt(query, docs, metas)

    try:
        responses = Generation.call(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            incremental_output=True,
        )

        for resp in responses:
            if resp.status_code == 200:
                text = resp.output.get("text", "")
                if text:
                    yield text
            else:
                raise RuntimeError(f"LLM stream error: {resp.code} {resp.message}")

    except Exception as e:
        yield f"\n\n[生成中断: {e}]"
