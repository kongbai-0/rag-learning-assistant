# -*- coding: utf-8 -*-
"""
RecursiveCharacterTextSplitter — 分层递归文本切分器。

设计原则：
  标题 → 段落 → 换行 → 句子 → 子句 → 字符（从粗到细逐级切分）

  不直接按字符长度硬切，优先在语义边界（句号、换行、段落）处切分，
  确保每个 chunk 尽可能表达一个完整知识点，而不是半个知识点。

  参考：LangChain RecursiveCharacterTextSplitter 的设计理念，
        但针对中文学术教材场景做了定制。
"""

import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


# ── 核心切分逻辑 ──

def _hard_split(text: str, chunk_size: int) -> list[str]:
    """兜底方案：按字符数硬切。"""
    if len(text) <= chunk_size:
        return [text.strip()]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _split_by_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    按分隔符列表递归切分，返回尽量接近 chunk_size 的文本块。

    算法：
      1. 用当前分隔符切分文本
      2. 如果只有一个片段（分隔符无效），降级到下一级分隔符
      3. 合并相邻小片段使每个片段接近 chunk_size
      4. 对仍然超过 chunk_size 的片段递归用更细的分隔符继续切
    """
    if not separators:
        return _hard_split(text, chunk_size)

    separator = separators[0]
    remaining = separators[1:]

    if not separator:
        return _split_by_separators(text, remaining, chunk_size)

    splits = text.split(separator)

    # 只有一个片段 → 此分隔符无效，降级
    if len(splits) <= 1:
        return _split_by_separators(text, remaining, chunk_size)

    # 把分隔符加回到片段之间
    parts = []
    for i, s in enumerate(splits):
        if i > 0:
            parts.append(separator)
        parts.append(s)

    # 合并相邻片段，使每个接近 chunk_size
    segments = []
    buf = ""
    for p in parts:
        candidate = buf + p
        if len(candidate) >= chunk_size and buf:
            segments.append(buf)
            buf = p
        else:
            buf = candidate
    if buf.strip():
        segments.append(buf)

    # 递归处理过长片段
    result = []
    for seg in segments:
        if len(seg) > chunk_size and remaining:
            result.extend(_split_by_separators(seg, remaining, chunk_size))
        elif len(seg) > chunk_size * 1.5:
            result.extend(_hard_split(seg, chunk_size))
        else:
            result.append(seg)

    return result


# ── 合并与重叠 ──

def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    """合并过小的 chunk（小于 chunk_size/3），避免碎片化。"""
    if not chunks:
        return []

    min_size = max(chunk_size // 3, 100)
    merged = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if merged and len(merged[-1]) < min_size:
            # 前一个太小，合并到前一个
            merged[-1] = merged[-1] + "\n" + chunk
        elif len(chunk) < min_size and merged:
            # 当前太小，合并到前一个
            merged[-1] = merged[-1] + "\n" + chunk
        else:
            merged.append(chunk)

    return merged


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """为相邻 chunk 添加重叠区域，保证上下文连贯。"""
    if not chunks or overlap <= 0:
        return chunks

    result = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev = result[-1]
            # 从前一个 chunk 末尾取 overlap 字符
            prefix = prev[-overlap:] if len(prev) > overlap else prev
            chunk = prefix + "\n" + chunk
        result.append(chunk)

    return result


# ── 公开接口 ──

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    将长文本切分为语义完整的 chunk 列表。

    切分流程：
      1. 按 CHUNK_SEPARATORS 的分层优先级递归切分
      2. 合并过小的碎片 chunk
      3. 相邻 chunk 之间添加 overlap 重叠

    参数：
      text:       原始文本
      chunk_size: 目标 chunk 大小（字符数），默认 800
      overlap:    相邻 chunk 重叠字符数，默认 150

    返回：
      chunk 字符串列表
    """
    if not text or not text.strip():
        return []

    # 第一步：分层递归切分
    raw = _split_by_separators(text, CHUNK_SEPARATORS, chunk_size)

    # 第二步：合并过小的 chunk
    merged = _merge_small_chunks(raw, chunk_size)

    # 第三步：添加重叠
    final = _add_overlap(merged, overlap)

    return [c.strip() for c in final if c.strip()]


def chunk_text_with_meta(
    pages_text: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    切分带页码信息的文本（用于任务2：Metadata增强）。

    参数：
      pages_text: [{"page": N, "text": "...", "section": "..."}, ...]
      chunk_size: 目标 chunk 大小
      overlap:    重叠字符数

    返回：
      [{"text": "...", "page": N, "section": "..."}, ...]
    """
    result = []
    for page_info in pages_text:
        page_num = page_info.get("page", 0)
        section = page_info.get("section", "")
        text = page_info.get("text", "")

        chunks = chunk_text(text, chunk_size, overlap)
        for chunk in chunks:
            result.append({
                "text": chunk,
                "page": page_num,
                "section": section,
            })

    return result


# ── 自测 ──

if __name__ == "__main__":
    test_text = (
        "# 第一章 数据结构概论\n\n"
        "数据结构是计算机存储、组织数据的方式。"
        "通常情况下，精心选择的数据结构可以带来更高的运行效率或存储效率。\n\n"
        "## 1.1 什么是数据结构\n\n"
        "数据结构是指相互之间存在一种或多种特定关系的数据元素的集合。"
        "它包括三个方面的内容：数据的逻辑结构、数据的存储结构、数据的运算。\n\n"
        "最常见的几种数据结构有：数组、链表、栈、队列、树、图等。\n\n"
        "## 1.2 算法复杂度分析\n\n"
        "算法的时间复杂度是指执行算法所需要的计算工作量。"
        "空间复杂度是指执行算法所需要的内存空间。\n\n"
        "大O表示法是一种描述算法复杂度的数学符号。"
        "常见的时间复杂度有O(1)、O(log n)、O(n)、O(n log n)、O(n²)等。"
    )

    # 用小 chunk_size 测试切分行为
    chunks = chunk_text(test_text, chunk_size=300, overlap=50)
    print(f"chunk_size=300 → {len(chunks)} 个 chunk")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] len={len(c):3d} | {c[:80].replace(chr(10), '↵')}...")

    print()
    chunks = chunk_text(test_text, chunk_size=800, overlap=150)
    print(f"chunk_size=800 → {len(chunks)} 个 chunk")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] len={len(c):3d} | {c[:80].replace(chr(10), '↵')}...")
