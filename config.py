# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-turbo"

CHROMA_PATH = "database/chroma_store"
COLLECTION_NAME = "rag_db"

# ── Chunk 策略（RecursiveCharacterTextSplitter） ──
CHUNK_SIZE = 800       # 目标 chunk 大小（字符数），优先在语义边界切分
CHUNK_OVERLAP = 150    # 相邻 chunk 重叠字符数，保证跨 chunk 上下文连贯
# 切分优先级（从高到低）：标题 → 段落 → 换行 → 句子 → 子句 → 字符
CHUNK_SEPARATORS = [
    "\n#",      # Markdown 标题
    "\n##",
    "\n###",
    "\n####",
    "\n\n",     # 段落
    "\n",       # 换行
    "。",       # 中文句号
    "！",       # 中文感叹号
    "？",       # 中文问号
    ". ",       # 英文句号
    "! ",       # 英文感叹号
    "? ",       # 英文问号
    "；",       # 中文分号
    "; ",       # 英文分号
    "，",       # 中文逗号
    ", ",       # 英文逗号
]

TOP_K = 5
# ── 动态 TopK ──
DYNAMIC_TOPK = True        # 是否根据问题复杂度自动调整 TopK
TOPK_SIMPLE = 3            # 简单问题（概念查询、单项查找）
TOPK_NORMAL = 5            # 普通问题（默认）
TOPK_COMPLEX = 8           # 复杂问题（多步推理、比较分析）
# 复杂度判断依据：问题长度、关键词（如何/为什么/比较/区别/分析/关系）
# ── 检索质量 ──
MIN_SCORE = 0.3          # 最低相似度阈值：低于此值的 chunk 直接过滤
                         # 余弦相似度 1-distance，0.3 以下视为无关内容
NO_RESULT_MSG = "未在当前课程资料中找到相关内容。\n\n建议：\n- 检查是否选择了正确的课程\n- 尝试用不同的关键词提问\n- 上传更多相关课程资料"

# ── MMR 去重 ──
MMR_LAMBDA = 0.7         # MMR 多样性参数：1.0=纯相关度排序，0.0=纯多样性排序
                          # 0.7 为推荐值，在相关性和多样性之间取得平衡
MMR_DEDUP_THRESHOLD = 0.85  # 简单相似度去重阈值：两个 chunk 相似度超过此值视为重复

# ── Rerank 重排序 ──
RERANK_MODEL = "gte-rerank"    # DashScope Rerank 模型
RERANK_ENABLED = True          # 是否启用 Rerank
RERANK_TOP_N = 5               # Rerank 后返回的结果数
RERANK_CANDIDATE_K = 20        # Embedding 初检数量（送入 Rerank 的候选数）

# ── 精确过滤时的相似度阈值 ──
FILTERED_MIN_SCORE = 0.1        # 当通过 source/section 精确过滤时使用更低阈值
FILTERED_MIN_SCORE_RATIO = 0.4  # 过滤模式下 min_score 乘以该系数，不低于 0.1
