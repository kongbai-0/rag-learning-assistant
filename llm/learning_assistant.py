# -*- coding: utf-8 -*-
"""
学习助手模块 — 课程学习相关的 AI 功能。

包含：
  任务5: 课程总结    generate_course_summary()
  任务6: 章节总结    generate_chapter_summary()
  任务7: 复习提纲    generate_review_outline()
  任务8: 自动出题    generate_exam_questions()
  任务9: 知识点解释  explain_concept()

设计原则：
  - 每个功能都先检索相关知识，再构建专用 Prompt，最后调用 LLM
  - Prompt 针对大学生学习场景做了优化
  - 不照抄教材，用通俗语言解释
"""

import os

from retrieval.search import search
from llm.dashscope_llm import generate
from ingestion.indexer import list_sources


def _resolve_chapter_target(course: str, target: str) -> tuple:
    """
    将用户输入的目标词解析为 ChromaDB 元数据过滤条件。

    三层匹配策略：
      1. 文件名匹配 — 去扩展名后做模糊匹配
      2. section 字段匹配 — 含中阿数字归一化（"二"→"2"）
      3. 兜底 — 返回空过滤条件，走纯语义搜索

    返回:
      (source_filter, section_filter, resolve_info_dict)
    """
    from ingestion.indexer import collection as _coll

    target_clean = target.strip()
    sources = list_sources(course)

    def _normalize(s: str) -> str:
        """中阿数字归一化：二→2, 十二→12"""
        cn_map = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
            "十一": "11", "十二": "12", "十三": "13", "十四": "14",
            "十五": "15", "十六": "16", "十七": "17", "十八": "18",
            "十九": "19", "二十": "20",
        }
        result = s
        for cn in sorted(cn_map.keys(), key=len, reverse=True):
            result = result.replace(cn, cn_map[cn])
        return result

    norm_target = _normalize(target_clean)

    # ── Tier 1: 文件名匹配（含中阿数字归一化）──
    for src in sources:
        src_noext = os.path.splitext(src)[0]
        norm_src = _normalize(src)
        norm_src_noext = _normalize(src_noext)
        if (target_clean == src or target_clean == src_noext
            or target_clean.lower() in src.lower()
            or src_noext.lower() in target_clean.lower()
            or norm_target == norm_src or norm_target == norm_src_noext
            or norm_target in norm_src or norm_src_noext in norm_target):
            return (src, None, {
                "matched_type": "source",
                "matched_value": src,
                "available_sources": sources,
                "available_sections": [],
            })

    # ── Tier 2: section 字段匹配 ──
    results = _coll.get(
        where={"course": course},
        include=["metadatas"],
    )
    sections_set = set()
    for meta in results.get("metadatas", []):
        s = meta.get("section", "")
        if s:
            sections_set.add(s)

    # 复用上面定义的 _normalize 和 norm_target
    best_match = None
    for sec in sections_set:
        norm_sec = _normalize(sec)
        if norm_target in norm_sec or norm_sec in norm_target or target_clean in sec:
            best_match = sec
            break

    if best_match:
        return (None, best_match, {
            "matched_type": "section",
            "matched_value": best_match,
            "available_sources": sources,
            "available_sections": sorted(sections_set),
        })

    # ── Tier 3: 兜底 ──
    return (None, None, {
        "matched_type": "semantic",
        "matched_value": None,
        "available_sources": sources,
        "available_sections": sorted(sections_set),
    })


def _build_notfound_hint(course: str, target: str, resolve_info: dict) -> str:
    """搜不到时生成友好错误提示，列出当前课程的文件和章节。"""
    lines = [
        f"未在课程「{course}」中找到与「{target}」相关的内容。",
        "",
    ]

    available_sources = resolve_info.get("available_sources", [])
    available_sections = resolve_info.get("available_sections", [])

    if available_sources:
        lines.append("**该课程包含的文件：**")
        for s in available_sources:
            lines.append(f"  - {s}")

    if available_sections:
        lines.append("")
        lines.append("**检测到的章节标题：**")
        for s in available_sections:
            lines.append(f"  - {s}")

    if not available_sources and not available_sections:
        lines.append("建议：请先上传包含该章节的 PDF 资料。")
    else:
        lines.append("")
        lines.append("**提示：** 请用上面列出的文件名或章节名重新查询。")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 任务6: 章节总结
# ═══════════════════════════════════════════════════════════

CHAPTER_SUMMARY_PROMPT = """你是一位大学课程辅导老师。请根据提供的章节资料，生成一份详细的章节总结。

要求：
1. **章节简介**：用1-2句话概括这章讲什么
2. **重点知识**：列出3-6个本章最重要的知识点，每个附带简要说明
3. **难点知识**：指出本章最难理解的部分（1-3个），并解释为什么难
4. **考试常考内容**：指出哪些知识点最常出现在考试中
5. **关键概念**：列出本章必须掌握的5-10个关键术语/概念

格式要求：使用 Markdown，重要概念用 **粗体** 标出。

注意：
- 如果资料中找不到指定章节，请明确说明
- 不要编造资料中没有的内容
- 回答末尾请标注信息来源：
  - **来自课件**的内容
  - **补充扩展**的内容（如有）"""


def generate_chapter_summary(course: str, section: str) -> str:
    """
    生成章节总结（任务6）。

    三层检索策略：
      1. 文件名匹配 → source 精确过滤
      2. section 字段匹配 → section 精确过滤
      3. 兜底 → 纯语义搜索

    参数：
      course:   课程名称
      section:  章节名称/标题（如"红黑树"、"第三章"、"第2章"）

    返回：
      Markdown 格式的章节总结
    """
    if not section.strip():
        return ("请指定要总结的章节名称。\n\n示例：`/章节 红黑树`", [], [], [], True)

    # 三层匹配
    source_filter, section_filter, resolve_info = _resolve_chapter_target(course, section)

    # Tier 1/2: 精确元数据过滤
    if source_filter or section_filter:
        docs, metas, scores = search(
            section,
            course=course,
            source=source_filter,
            section=section_filter,
            top_k=10,
            enable_mmr=True,
        )
    else:
        # Tier 3: 兜底语义搜索
        query = f"{section} 主要内容 知识点"
        docs, metas, scores = search(query, course=course, top_k=10, enable_mmr=True)

    if not docs:
        return (_build_notfound_hint(course, section, resolve_info), [], [], [], False)

    prompt = (f"课程：{course}\n"
              f"章节：{section}\n\n"
              f"章节相关资料：\n\n")

    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        page = meta.get("page", 0)
        section_name = meta.get("section", "")
        header = f"[片段{i}]"
        if section_name:
            header += f" 章节: {section_name}"
        if page:
            header += f" 页码: {page}"
        prompt += f"{header}\n{doc[:1500]}\n\n"

    prompt += f"\n请根据以上资料生成「{section}」的章节总结。"
    reply = generate(prompt, docs, metas)
    return (reply, docs, metas, scores, True)


# ═══════════════════════════════════════════════════════════
# 任务8: 自动出题
# ═══════════════════════════════════════════════════════════

EXAM_QUESTIONS_PROMPT = """你是一位大学课程出题老师。请根据提供的课程资料，生成考试题目。

要求：
1. 题目必须基于提供的资料内容
2. 题目类型按需生成（选择题、判断题、简答题）
3. 每道题都要包含：题目、正确答案、解析、来源章节
4. 题目难度适中，适合大学生期末考试水平
5. 选择题提供4个选项（A/B/C/D）

输出格式：
```
## 选择题

**1. [题目]**
A. ...  B. ...  C. ...  D. ...
> 答案：X
> 解析：...
> 来源：[章节名]

## 判断题

**1. [题目]**
> 答案：正确/错误
> 解析：...
> 来源：[章节名]

## 简答题

**1. [题目]**
> 参考答案：...
> 解析：...
> 来源：[章节名]
```

注意：
- 题目要覆盖不同难度层次
- 解析要详细，帮助学生理解为什么对/错
- 不要出资料中没有的题目
- 每道题请标注信息来源：
  - **来自课件**的题目
  - **补充扩展**的题目（如有）"""


def generate_exam_questions(
    course: str,
    section: str = "",
    question_type: str = "mixed",
    count: int = 5,
) -> str:
    """
    自动出题（任务8）。

    参数：
      course:        课程名称
      section:       章节（可选，如"第三章"、"红黑树"）
      question_type: 题型 — "choice"(选择), "truefalse"(判断),
                     "shortanswer"(简答), "mixed"(混合，默认)
      count:         题目数量

    返回：
      Markdown 格式的题目列表
    """
    # 构建查询（有章节时走三层匹配）
    if section:
        source_filter, section_filter, resolve_info = _resolve_chapter_target(course, section)

        if source_filter or section_filter:
            docs, metas, scores = search(
                f"{section} 知识点 考点 重点",
                course=course,
                source=source_filter,
                section=section_filter,
                top_k=12,
                enable_mmr=True,
            )
        else:
            query = f"{section} 知识点 考点 重点"
            docs, metas, scores = search(query, course=course, top_k=12, enable_mmr=True)
    else:
        query = "重点 考点 关键概念 核心知识点"
        docs, metas, scores = search(query, course=course, top_k=12, enable_mmr=True)
        resolve_info = {"available_sources": [], "available_sections": []}

    if not docs:
        if section:
            return (_build_notfound_hint(course, section, resolve_info), [], [], [], False)
        return (f"课程「{course}」暂无相关资料。\n\n"
                "请先上传该课程的 PDF 教材或课件。", [], [], [], False)

    type_desc = {"choice": "选择题", "truefalse": "判断题",
                 "shortanswer": "简答题", "mixed": "混合题型"}
    type_str = type_desc.get(question_type, "混合题型")

    scope = f"课程「{course}」" + (f" 的「{section}」章节" if section else "全部内容")

    prompt = (f"出题范围：{scope}\n"
              f"题型要求：{type_str}\n"
              f"题目数量：共 {count} 道\n\n"
              f"参考资料：\n\n")

    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        section_name = meta.get("section", "")
        page = meta.get("page", 0)
        header = f"[片段{i}]"
        if section_name:
            header += f" 章节: {section_name}"
        if page:
            header += f" 页码: {page}"
        prompt += f"{header}\n{doc[:1500]}\n\n"

    prompt += (f"\n请根据以上资料生成 {count} 道{type_str}。"
               f"确保题目覆盖资料中的不同知识点。")
    reply = generate(prompt, docs, metas)
    return (reply, docs, metas, scores, True)


# ═══════════════════════════════════════════════════════════
# 任务9: 知识点解释模式
# ═══════════════════════════════════════════════════════════

EXPLAIN_CONCEPT_PROMPT = """你是一位善于讲课的大学助教。你的任务是用大学生**最容易理解的方式**解释知识点。

核心原则：
1. **不要照抄教材**。教材上的定义通常很抽象，你需要翻译成"人话"。
2. **举例优先**。每个抽象概念至少给一个具体、生动的例子。
3. **善用类比**。把陌生的概念类比成日常生活中熟悉的事物。
4. **图景化描述**。用文字描绘画面，帮助学生在脑中建立直观理解。
5. **由浅入深**。先给最直观的理解，再逐步深入细节。

回答结构：
1. 一句话概览（最直观的理解）
2. 通俗解释（类比 + 举例）
3. 技术细节（如果需要）
4. 常见误区（如果存在）
5. 考试小贴士（如果相关内容在考试中常出现）

注意：
- 如果课程资料包含该知识点的解释，引用并展开
- 如果资料不完整，基于你的知识补充，但要注明哪些来自资料、哪些来自补充
- 语言要亲切自然，像学长/学姐在给你讲题
- 回答末尾请明确标注信息来源：
  - **来自课件**：引用资料中的定义和解释
  - **补充知识**：基于AI通用知识的补充
- 如果课程资料不包含该知识点，请明确说明并基于通用知识解释。"""


def explain_concept(course: str, concept: str, style: str = "通俗") -> str:
    """
    知识点解释（任务9）。

    参数：
      course:  课程名称
      concept: 要解释的知识点（如"TCP三次握手"）
      style:   解释风格 — "通俗"(默认), "学术", "应试"

    返回：
      Markdown 格式的知识点解释
    """
    if not concept.strip():
        return ("请输入要解释的知识点。\n\n示例：`请解释：红黑树的旋转操作`", [], [], [])

    # 检索相关知识
    docs, metas, scores = search(concept, course=course, top_k=8, enable_mmr=True)

    style_guide = {
        "通俗": "用通俗易懂的语言，多举例、多类比，像学长学姐在讲课",
        "学术": "保持学术严谨，引用资料中的定义，但也要解释清楚",
        "应试": "聚焦考试要点，说明怎么考、怎么答、常见陷阱",
    }
    style_prompt = style_guide.get(style, style_guide["通俗"])

    prompt = (f"课程：{course}\n"
              f"知识点：{concept}\n"
              f"解释风格：{style_prompt}\n\n")

    if docs:
        prompt += "课程相关资料：\n\n"
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            section = meta.get("section", "")
            header = f"[片段{i}]" + (f" (章节: {section})" if section else "")
            prompt += f"{header}\n{doc[:1200]}\n\n"

    prompt += (f"\n请用{style_prompt}的方式解释「{concept}」。"
               f"回答末尾请注明哪些内容来自课程资料，哪些来自补充知识。")

    reply = generate(prompt, docs if docs else ["（无资料，请基于通用知识解释）"],
                    metas if metas else [{}])
    return (reply, docs, metas, scores)


# ═══════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("学习助手模块加载成功。")
    print("可用函数：")
    print("  generate_course_summary(course)")
    print("  generate_chapter_summary(course, section)")
    print("  generate_review_outline(course)")
    print("  generate_exam_questions(course, section, type, count)")
    print("  explain_concept(course, concept, style)")
