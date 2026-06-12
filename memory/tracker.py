# memory/tracker.py
# -*- coding: utf-8 -*-
"""
课程记忆追踪 — 章节学习记录 + 知识点掌握度标记。

存储位置: storage/course_memory.json
"""
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

MEMORY_PATH = os.path.join("storage", "course_memory.json")
MAX_CHAPTERS_IN_CONTEXT = 10


def _load() -> dict:
    """加载记忆文件，文件不存在或损坏时返回空 dict。"""
    if not os.path.exists(MEMORY_PATH):
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        # 损坏 → 备份后返回空
        logger.warning("记忆文件损坏，重建: %s", e)
        backup = MEMORY_PATH + ".bak"
        try:
            os.rename(MEMORY_PATH, backup)
        except OSError:
            pass
        return {}


def _save(data: dict) -> None:
    """保存记忆文件。写入失败时记录日志但不抛异常。"""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.warning("无法写入记忆文件 %s: %s", MEMORY_PATH, e)


def _ensure_course(data: dict, course: str) -> dict:
    """确保课程条目存在，返回该课程的 dict。"""
    if course not in data:
        data[course] = {
            "chapters_learned": [],
            "mastery": {},
            "last_active": "",
        }
    return data[course]


# ── 公开 API ──


def record_chapter(course: str, chapter: str) -> None:
    """记录某课程的一个章节已被学习（去重）。"""
    if not course or not chapter:
        return
    data = _load()
    entry = _ensure_course(data, course)
    if chapter not in entry["chapters_learned"]:
        entry["chapters_learned"].append(chapter)
    entry["last_active"] = datetime.now().isoformat()
    _save(data)


def mark_mastery(course: str, concept: str, level: str) -> None:
    """
    标记一个知识点的掌握程度。

    level 取值: "mastered" | "weak" | "unmarked"
    """
    if not course or not concept:
        return
    if level not in ("mastered", "weak", "unmarked"):
        return
    data = _load()
    entry = _ensure_course(data, course)
    entry["mastery"][concept] = level
    entry["last_active"] = datetime.now().isoformat()
    _save(data)


def get_weak_concepts(course: str) -> list[str]:
    """获取某课程所有标记为薄弱的知识点列表。"""
    data = _load()
    entry = data.get(course, {})
    return [k for k, v in entry.get("mastery", {}).items() if v == "weak"]


def get_chapters_learned(course: str) -> list[str]:
    """获取某课程已学习的章节列表。"""
    data = _load()
    entry = data.get(course, {})
    return entry.get("chapters_learned", [])


def get_context_prompt(course: str | None) -> str:
    """
    生成注入 LLM 的上下文摘要（~几十 token）。

    返回空字符串表示无上下文。
    """
    if not course:
        return ""

    data = _load()
    entry = data.get(course, {})
    chapters = entry.get("chapters_learned", [])
    weak = [k for k, v in entry.get("mastery", {}).items() if v == "weak"]

    if not chapters and not weak:
        return ""

    parts = [f"当前课程：{course}"]
    if chapters:
        parts.append(f"已学章节：{', '.join(chapters[-MAX_CHAPTERS_IN_CONTEXT:])}")
    if weak:
        parts.append(f"薄弱知识点（建议加强）：{', '.join(weak)}")

    return "\n".join(parts)


def get_summary(course: str) -> dict:
    """获取某课程的完整记忆摘要，供 UI 展示。"""
    data = _load()
    entry = data.get(course, {})
    mastery = entry.get("mastery", {})
    return {
        "chapters_learned": entry.get("chapters_learned", []),
        "mastery": mastery,
        "weak_count": sum(1 for v in mastery.values() if v == "weak"),
        "last_active": entry.get("last_active", ""),
    }
