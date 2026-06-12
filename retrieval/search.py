# -*- coding: utf-8 -*-
import chromadb
from ingestion.embedder import get_embedding
from config import (
    CHROMA_PATH, COLLECTION_NAME, TOP_K, MIN_SCORE,
    MMR_LAMBDA, MMR_DEDUP_THRESHOLD,
    RERANK_ENABLED, RERANK_TOP_N, RERANK_CANDIDATE_K,
    DYNAMIC_TOPK, TOPK_SIMPLE, TOPK_NORMAL, TOPK_COMPLEX,
    FILTERED_MIN_SCORE, FILTERED_MIN_SCORE_RATIO,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _estimate_query_complexity(query: str) -> int:
    """
    根据问题复杂度估算合适的 TopK 值。

    判断规则：
      - 简单问题 (<15字, 单一概念): Top3
      - 复杂问题 (>60字 或 含多步推理关键词): Top8
      - 普通问题: Top5
    """
    q = query.strip()
    length = len(q)

    # 复杂问题关键词
    complex_keywords = [
        "为什么", "如何", "怎么", "比较", "区别", "异同",
        "分析", "关系", "联系", "影响", "原因", "过程",
        "步骤", "流程", "原理", "机制",
        "和.*区别", "与.*区别", "对比",
    ]

    # 简单问题特征
    simple_patterns = [
        "是什么", "什么是", "定义", "简称", "缩写",
        "公式", "定理",
    ]

    # 多问题检测（含多个问号）
    question_count = q.count("？") + q.count("?")

    # 判断复杂度
    import re

    # 复杂：多问题
    if question_count >= 2:
        return TOPK_COMPLEX

    # 复杂：含复杂关键词
    for kw in complex_keywords:
        if kw in q:
            return TOPK_COMPLEX

    # 复杂：很长的问题
    if length > 60:
        return TOPK_COMPLEX

    # 简单：含简单关键词且较短
    for pat in simple_patterns:
        if pat in q and length < 25:
            return TOPK_SIMPLE

    # 简单：非常短的问题
    if length < 10:
        return TOPK_SIMPLE

    # 普通
    return TOPK_NORMAL


def _to_similarity(distance: float) -> float:
    """
    将 ChromaDB 返回的 distance 转换为相似度 (0~1)。

    根据 collection 的距离度量自动选择转换公式：
      - cosine: distance ∈ [0, 2], sim = 1 - distance          (范围 [-1, 1])
      - l2:     distance ∈ [0, ∞),  sim = 1 / (1 + distance)   (范围 (0, 1])
    """
    coll_meta = collection.metadata or {}
    space = coll_meta.get("hnsw:space", "l2")

    if space == "cosine":
        return 1.0 - distance
    else:
        # L2 / IP (内积) 等非 cosine 度量，用归一化公式
        return 1.0 / (1.0 + distance)


def _jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """
    计算两个文本的字符 n-gram Jaccard 相似度。

    用于 MMR 中的 chunk 间相似度计算，无需额外 API 调用。
    """
    def _ngrams(s: str) -> set:
        s = s.replace("\n", " ").replace(" ", "")
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    ng_a = _ngrams(text_a)
    ng_b = _ngrams(text_b)

    if not ng_a or not ng_b:
        return 0.0

    intersection = len(ng_a & ng_b)
    union = len(ng_a | ng_b)
    return intersection / union if union > 0 else 0.0


def _mmr_select(
    docs: list[str],
    metas: list[dict],
    scores: list[float],
    k: int,
    lam: float = MMR_LAMBDA,
) -> tuple[list[str], list[dict], list[float]]:
    """
    MMR (Maximal Marginal Relevance) 去重选择。

    算法：
      MMR = argmax [ λ·Sim(Di, Q) - (1-λ)·max_{Dj∈S} Sim(Di, Dj) ]

      其中：
        λ=1.0: 纯相关性排序（无去重）
        λ=0.0: 纯多样性排序
        λ=0.7: 推荐默认值

    时间复杂度：O(n²) where n = len(docs)
    """
    if len(docs) <= k:
        return docs, metas, scores

    # 预计算 chunk 间相似度矩阵（对称）
    n = len(docs)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard_similarity(docs[i], docs[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    selected_idx = []
    remaining_idx = list(range(n))

    # 第一步：选最相关的
    best_idx = max(remaining_idx, key=lambda i: scores[i])
    selected_idx.append(best_idx)
    remaining_idx.remove(best_idx)

    # 后续步骤：MMR 选择
    while len(selected_idx) < k and remaining_idx:
        mmr_scores = []
        for i in remaining_idx:
            relevance = scores[i]
            # 与已选中 chunk 的最大相似度
            max_red = max(sim_matrix[i][j] for j in selected_idx) if selected_idx else 0.0
            mmr = lam * relevance - (1 - lam) * max_red
            mmr_scores.append(mmr)

        best_local = max(range(len(mmr_scores)), key=lambda i: mmr_scores[i])
        selected_idx.append(remaining_idx[best_local])
        remaining_idx.pop(best_local)

    return (
        [docs[i] for i in selected_idx],
        [metas[i] for i in selected_idx],
        [scores[i] for i in selected_idx],
    )


def _simple_dedup(docs, metas, scores, threshold=MMR_DEDUP_THRESHOLD):
    """
    简单相似度去重：移除与已保留 chunk 高度相似的 chunk。
    作为 MMR 的轻量替代方案。
    """
    if not docs:
        return [], [], []

    kept_docs = [docs[0]]
    kept_metas = [metas[0]]
    kept_scores = [scores[0]]

    for doc, meta, score in zip(docs[1:], metas[1:], scores[1:]):
        is_dup = False
        for kd in kept_docs:
            if _jaccard_similarity(doc, kd) > threshold:
                is_dup = True
                break
        if not is_dup:
            kept_docs.append(doc)
            kept_metas.append(meta)
            kept_scores.append(score)

    return kept_docs, kept_metas, kept_scores


def _build_where_clause(course=None, source=None, section=None):
    """
    构建 ChromaDB where 条件，支持多条件 $and 组合。

    所有参数均为可选；返回 None 表示无需过滤。
    """
    conditions = []
    if course:
        conditions.append({"course": course})
    if source:
        conditions.append({"source": source})
    if section:
        conditions.append({"section": section})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def search(
    query,
    course=None,
    source=None,
    section=None,
    top_k=TOP_K,
    min_score=MIN_SCORE,
    dynamic_min_score=True,
    enable_mmr=True,
    mmr_lambda=MMR_LAMBDA,
    enable_rerank=RERANK_ENABLED,
):
    """
    语义检索 + Rerank 重排序 + 相关性过滤 + MMR 去重。

    完整流程：
      1. Embedding 初检（fetch_k = max(top_k*3, RERANK_CANDIDATE_K)）
      2. 转换距离为相似度、过滤低分结果
      3. [Rerank] 对候选结果进行 Cross-Encoder 重排序
      4. [MMR] 去重选择 top_k 个多样化结果

    参数：
      query:        查询文本
      course:       课程过滤（None 表示所有课程）
      source:       文件名过滤（如 "第2章.pdf"），精确匹配
      section:      章节标题过滤（如 "第二章 进程管理"），精确匹配
      top_k:        最终返回的结果数量
      min_score:    最低相似度阈值（0~1）
      dynamic_min_score: 精确过滤时自动降低 min_score 阈值
      enable_mmr:   是否启用 MMR 去重
      mmr_lambda:   MMR 参数（0~1）
      enable_rerank:是否启用 Rerank

    返回：
      (docs, metas, scores): 结果列表，无结果时为空列表
    """
    # ── Step 0: 动态 TopK + Embedding 初检 ──
    if DYNAMIC_TOPK:
        top_k = _estimate_query_complexity(query)

    fetch_k = max(top_k * 3, RERANK_CANDIDATE_K, 30)
    query_embedding = get_embedding(query)

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    where_clause = _build_where_clause(course=course, source=source, section=section)
    if where_clause:
        kwargs["where"] = where_clause

    results = collection.query(**kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # ── Step 1: 转换为相似度并过滤 ──
    # 精确过滤时使用更低的阈值（元数据已保证相关性）
    effective_min_score = min_score
    if dynamic_min_score and (source or section):
        effective_min_score = max(min_score * FILTERED_MIN_SCORE_RATIO, FILTERED_MIN_SCORE)

    filtered_docs = []
    filtered_metas = []
    filtered_scores = []

    for doc, meta, dist in zip(docs, metas, distances):
        sim = _to_similarity(dist)
        if sim >= effective_min_score:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_scores.append(sim)

    if not filtered_docs:
        return [], [], []

    # ── Step 2: Rerank 重排序 ──
    if enable_rerank and len(filtered_docs) > 1:
        try:
            from retrieval.rerank import rerank_search_results
            rerank_k = max(top_k, RERANK_TOP_N)
            filtered_docs, filtered_metas, filtered_scores = rerank_search_results(
                query, filtered_docs, filtered_metas, filtered_scores,
                top_n=min(rerank_k, len(filtered_docs)),
            )
        except Exception as e:
            print(f"[Search] Rerank 失败，使用 Embedding 结果: {e}")

    # ── Step 3: MMR 去重 ──
    if enable_mmr and len(filtered_docs) > top_k:
        dedup_docs, dedup_metas, dedup_scores = _mmr_select(
            filtered_docs, filtered_metas, filtered_scores,
            k=top_k, lam=mmr_lambda,
        )
    elif len(filtered_docs) > top_k:
        dedup_docs, dedup_metas, dedup_scores = _simple_dedup(
            filtered_docs[:top_k], filtered_metas[:top_k], filtered_scores[:top_k],
        )
    else:
        dedup_docs, dedup_metas, dedup_scores = filtered_docs, filtered_metas, filtered_scores

    return dedup_docs, dedup_metas, dedup_scores


if __name__ == "__main__":
    print("start search...")

    docs, metas, scores = search("抗战精神是什么", top_k=5, min_score=0.3)

    if not docs:
        print("未找到相关内容（所有结果相似度低于阈值）")
    else:
        for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
            src = meta.get("source", "?")
            course = meta.get("course", "?")
            page = meta.get("page", 0)
            section = meta.get("section", "")
            source_info = f"{course}/{src}"
            if section:
                source_info += f" · {section}"
            if page:
                source_info += f" · 第{page}页"
            print(f"\n--- result {i} ({source_info}, sim={score:.3f}) ---\n")
            print(doc[:200])
    print(f"\n库中 chunk 总数: {collection.count()}")
