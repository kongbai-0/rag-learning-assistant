# -*- coding: utf-8 -*-
"""
学习记录系统 — 持久化提问历史和回答。

存储格式：JSON 文件，按日期组织。

数据结构：
  {
    "records": [
      {
        "id": "uuid",
        "timestamp": "2026-06-02T15:30:00",
        "course": "数据结构",
        "question": "什么是红黑树？",
        "answer": "红黑树是一种自平衡二叉查找树...",
        "sources": ["数据结构.pdf (第23页)"],
        "msg_type": "qa" | "summary" | "chapter" | "review" | "exam" | "explain"
      }
    ]
  }

未来扩展：
  - 学习轨迹分析（按课程统计提问频率、知识薄弱点识别）
  - 学习报告生成
"""

import json
import os
import uuid
from datetime import datetime

# 存储路径
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_FILE = os.path.join(STORAGE_DIR, "learning_history.json")


def _ensure_storage():
    """确保存储目录和文件存在。"""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"records": []}, f, ensure_ascii=False, indent=2)


def save_record(
    question: str,
    answer: str,
    course: str = "",
    sources: list[str] | None = None,
    msg_type: str = "qa",
) -> str:
    """
    保存一条学习记录。

    参数：
      question: 用户提问
      answer:   系统回答
      course:   课程名称
      sources:  来源列表（如 ["数据结构.pdf (第23页)"]）
      msg_type: 消息类型 — qa/summary/chapter/review/exam/explain

    返回：
      记录的 UUID
    """
    _ensure_storage()

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    record_id = str(uuid.uuid4())[:8]
    record = {
        "id": record_id,
        "timestamp": datetime.now().isoformat(),
        "course": course,
        "question": question,
        "answer": answer[:2000],  # 截断过长回答
        "sources": sources or [],
        "msg_type": msg_type,
    }

    data["records"].append(record)

    # 只保留最近 500 条记录
    if len(data["records"]) > 500:
        data["records"] = data["records"][-500:]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return record_id


def get_history(course: str = "", limit: int = 50) -> list[dict]:
    """
    获取学习历史记录。

    参数：
      course: 按课程过滤（空字符串表示所有课程）
      limit:  返回条数上限

    返回：
      记录列表（按时间倒序）
    """
    _ensure_storage()

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])

    if course:
        records = [r for r in records if r.get("course") == course]

    return sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:limit]


def get_course_stats_from_history() -> dict:
    """
    从学习记录中统计各课程的学习情况。

    返回：
      { "数据结构": {"total": 15, "qa": 10, "summary": 2, ...}, ... }
    """
    _ensure_storage()

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {}
    for r in data.get("records", []):
        course = r.get("course", "未知")
        msg_type = r.get("msg_type", "qa")

        if course not in stats:
            stats[course] = {"total": 0}
        stats[course]["total"] = stats[course].get("total", 0) + 1
        stats[course][msg_type] = stats[course].get(msg_type, 0) + 1

    return stats


def clear_history(course: str = ""):
    """清空学习记录。"""
    if course:
        _ensure_storage()
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["records"] = [r for r in data["records"] if r.get("course") != course]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"records": []}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 自测
    rid = save_record("什么是红黑树？", "红黑树是一种自平衡二叉查找树...",
                      course="数据结构", sources=["数据结构.pdf (第23页)"])
    print(f"保存记录: {rid}")

    history = get_history(course="数据结构", limit=5)
    print(f"数据结构 最近记录: {len(history)} 条")
    for h in history:
        print(f"  [{h['timestamp'][:19]}] {h['question']}")

    stats = get_course_stats_from_history()
    print(f"学习统计: {stats}")
