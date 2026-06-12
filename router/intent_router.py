# router/intent_router.py
# -*- coding: utf-8 -*-
"""
意图路由器 — 将用户自然语言消息分类为功能意图。

三层策略：
  1. 斜杠命令正则匹配 → 不消耗 LLM 调用
  2. LLM 意图分类 (qwen-turbo) → 主路径
  3. 降级为 qa → 兜底
"""
import json
import re

from llm.dashscope_llm import generate

_DEFAULT_EXAM_COUNT = 5
_MAX_EXAM_COUNT = 20


INTENT_PROMPT = """你是学习助手的意图路由器。分析用户消息，返回 JSON。

意图类型：          
- "qa": 一般问答，如"什么是进程"、"第二章讲了什么"
- "chapter_summary": 要求总结某个章节，如"总结第二章"、"第二章重点"
- "exam": 要求出题，如"出5道选择"、"给我出几道关于B树的判断题"
- "explain": 要求解释概念，如"解释红黑树"、"什么是死锁"
- "mark_mastery": 标记掌握度，如"标记死锁为薄弱点"、"进程同步我学会了"
- "query_weak": 查询薄弱知识点，如"我的薄弱点有哪些"、"薄弱点"
- "course_mgmt": 课程管理，如"有哪些文件"、"上传PDF"

参数说明：
- chapter: 章节名（chapter_summary 时提取）
- concept: 概念名（explain / mark_mastery 时提取）
- question_type: "choice"/"truefalse"/"shortanswer"/"mixed"，仅 exam
- count: 出题数量（1-20），仅 exam
- mastery_level: "mastered"/"weak"/"unmarked"，仅 mark_mastery

上下文：
{context}

用户消息：{message}

只返回 JSON，不要其他内容。无法判断时 intent 为 "qa"。
JSON:"""


# ── 斜杠命令正则（Layer 1 兜底）──

_SLASH_PATTERNS = [
    (r"^/(?:章节|chapter)\s+(.+)", "chapter_summary"),
    (r"^/(?:出题|exam)\s+(.*)", "exam"),
    (r"^/(?:解释|explain)\s+(.+)", "explain"),
    (r"^/(?:帮助|help|\?)$", "help"),
    (r"^/(?:历史)\s*(.*)", "history"),
]

# ── 自然语言正则（Layer 1.5，高频指令不依赖 LLM）──
# 格式: (pattern, intent, param_extractor)
# param_extractor 接收 re.match 对象，返回部分 intent dict

_NL_PATTERNS = [
    # 标记薄弱点: "标记死锁为薄弱点" / "把XX标记为薄弱点"
    (r"^(?:把\s*)?标记\s*(.+?)\s*为\s*(?:薄弱点|弱点|weak)\s*$",
     "mark_mastery", lambda m: {"concept": m.group(1).strip(), "mastery_level": "weak"}),
    # 标记已掌握: "标记XX为已掌握" / "XX已掌握" / "XX我学会了"
    (r"^(?:把\s*)?标记\s*(.+?)\s*为\s*(?:已掌握|已学会|mastered)\s*$",
     "mark_mastery", lambda m: {"concept": m.group(1).strip(), "mastery_level": "mastered"}),
    (r"^(.+?)\s*(?:已掌握|我学会了|掌握了)\s*$",
     "mark_mastery", lambda m: {"concept": m.group(1).strip(), "mastery_level": "mastered"}),
    # 查询薄弱点: "我的薄弱点有哪些" / "薄弱点" / "查看薄弱点"
    (r"^(?:我的|查看|显示)?\s*薄弱点\s*(?:有哪些|列表|是什么)?\s*$",
     "query_weak", lambda m: {}),
]


def _parse_nl_command(message: str) -> dict | None:
    """自然语言正则匹配（Layer 1.5），命中返回 intent dict，否则返回 None。"""
    msg = message.strip()
    for pattern, intent, extractor in _NL_PATTERNS:
        m = re.match(pattern, msg)
        if m:
            result = {"intent": intent, "chapter": None, "concept": None,
                      "question_type": None, "count": None, "mastery_level": None}
            result.update(extractor(m))
            return result
    return None


def _parse_slash_command(message: str) -> dict | None:
    """斜杠命令正则匹配，命中返回 intent dict，否则返回 None。"""
    msg = message.strip()
    for pattern, intent in _SLASH_PATTERNS:
        m = re.match(pattern, msg)
        if m:
            result = {"intent": intent, "chapter": None, "concept": None,
                      "question_type": None, "count": None, "mastery_level": None}
            arg = m.group(1).strip() if m.lastindex and m.group(1) else ""

            if intent == "chapter_summary":
                result["chapter"] = arg if arg else None
            elif intent == "explain":
                result["concept"] = arg if arg else None
            elif intent == "exam":
                result["question_type"] = "mixed"
                result["count"] = _DEFAULT_EXAM_COUNT
                if arg:
                    parts = arg.split()
                    if parts and parts[-1].isdigit():
                        result["count"] = max(1, min(int(parts[-1]), _MAX_EXAM_COUNT))
                        parts = parts[:-1]
                    # 题型检测
                    type_map = {"选择": "choice", "选择题": "choice",
                                "判断": "truefalse", "判断题": "truefalse",
                                "简答": "shortanswer", "简答题": "shortanswer"}
                    if parts and parts[-1] in type_map:
                        result["question_type"] = type_map[parts[-1]]
                        parts = parts[:-1]
                    if parts:
                        result["chapter"] = " ".join(parts)
            elif intent == "history":
                result["chapter"] = arg if arg else None

            return result
    return None


def _classify_by_llm(message: str, context: str) -> dict:
    """用 qwen-turbo 做意图分类，返回 intent dict。异常时降级为 qa。"""
    prompt = INTENT_PROMPT.format(context=context, message=message)

    try:
        response = generate(prompt, docs=[""], metas=[{}])
    except Exception:
        return _qa_fallback()

    # 提取 JSON
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        result = json.loads(text)
        result.setdefault("chapter", None)
        result.setdefault("concept", None)
        result.setdefault("question_type", None)
        result.setdefault("count", None)
        result.setdefault("mastery_level", None)
        result.setdefault("intent", "qa")
        return result
    except (json.JSONDecodeError, ValueError):
        return _qa_fallback()


def _qa_fallback() -> dict:
    return {"intent": "qa", "chapter": None, "concept": None,
            "question_type": None, "count": None, "mastery_level": None}


# ── 公开 API ──


def route(message: str, context_prompt: str = "") -> dict:
    """
    路由用户消息到意图。

    参数:
      message:         用户输入消息
      context_prompt:   课程上下文（来自 memory.tracker）

    返回:
      {
        "intent": str,        # qa|chapter_summary|exam|explain|mark_mastery|course_mgmt|help|history
        "chapter": str|None,
        "concept": str|None,
        "question_type": str|None,  # choice|truefalse|shortanswer|mixed
        "count": int|None,
        "mastery_level": str|None,  # mastered|weak|unmarked
      }
    """
    # Layer 1: 斜杠命令
    slash = _parse_slash_command(message)
    if slash:
        return slash

    # Layer 1.5: 自然语言正则（高频指令，不依赖 LLM）
    nl = _parse_nl_command(message)
    if nl:
        return nl

    # Layer 2: LLM 分类
    context = context_prompt or ""
    return _classify_by_llm(message, context)
