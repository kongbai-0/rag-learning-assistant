# -*- coding: utf-8 -*-
import os
import shutil

import gradio as gr

from ingestion.pdf_loader import load_pdf, load_pdf_with_meta
from ingestion.chunker import chunk_text, chunk_text_with_meta
from ingestion.indexer import (
    index_chunks, list_courses, get_course_stats, delete_course,
    list_sources, delete_source, get_source_count, collection,
)
from retrieval.search import search
from llm.dashscope_llm import generate, generate_stream
from llm.learning_assistant import (
    generate_chapter_summary,
    generate_exam_questions,
    explain_concept,
)
from router.intent_router import route as route_intent
from memory.tracker import (
    record_chapter, mark_mastery, get_context_prompt,
    get_summary, get_chapters_learned, get_weak_concepts,
)
from utils.text_clean import clean_text
from utils.ppt_converter import convert_pptx_to_pdf
from storage.learning_log import save_record, get_history


# ── helpers ─────────────────────────────────────────────

def _build_course_choices():
    """返回课程下拉列表选项。"""
    courses = list_courses()
    return ["全部"] + courses


def _build_guide() -> str:
    """返回新用户引导消息。"""
    return """
### 课程即刻开始

上传教材。

提出问题。

获得答案。

让知识触手可及，从现在开始拥有自己的知识库。


> **须知**：每一次回答都基于你的课程资料，而非互联网。支持 PDF、PPT、PPTX 格式。
"""


def _build_welcome(course: str | None) -> str:
    """生成切换课程后的欢迎消息。"""
    if not course or course == "全部":
        return _build_guide()

    from ingestion.indexer import list_sections
    sources = list_sources(course)
    sections = list_sections(course)
    memory = get_summary(course)

    from storage.learning_log import get_course_stats_from_history
    stats = get_course_stats_from_history()
    course_stats = stats.get(course, {"total": 0})

    lines = [f"已切换到课程「{course}」\n\n## {course}\n"]
    lines.append(f"{len(sources)} 个文件 | 已提问 {course_stats.get('total', 0)} 次 | "
                 f"已学 {len(memory.get('chapters_learned', []))} 章节\n")

    if sections:
        lines.append("\n**检测到的章节：**")
        for s in sections:
            learned = " [已完成]" if s in memory.get("chapters_learned", []) else ""
            lines.append(f"- {s}{learned}")

    if memory.get("weak_count", 0) > 0:
        lines.append(f"\n**注意：** {memory['weak_count']} 个薄弱知识点待加强。")

    lines.append('\n**提示：** 你可以直接说："总结第二章" / "出5道选择" / "解释关键概念"')
    return "\n".join(lines)


def _format_sources_detail(docs, metas, scores) -> str:
    """生成折叠的检索来源 HTML details/summary。"""
    if not docs:
        return ""
    lines = ["\n<details>\n<summary>检索来源 ({n}个片段)</summary>\n".format(n=len(docs))]
    for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
        src = meta.get("source", "?")
        page = meta.get("page", 0)
        section = meta.get("section", "")
        preview = doc[:120].replace("\n", " ")
        source_info = f"[{src}]"
        if section:
            source_info += f" · 章节: {section}"
        if page:
            source_info += f" · 第{page}页"
        lines.append(
            f"- **片段{i + 1}** "
            f"(相似度: {score:.3f}) {source_info}: {preview}..."
        )
    lines.append("\n</details>")
    return "\n".join(lines)


def _detect_msg_type(message: str) -> str:
    msg = message.strip()
    if msg.startswith("/总结") or msg.startswith("/章节"):
        return "chapter"
    elif msg.startswith("/复习"):
        return "review"
    elif msg.startswith("/出题"):
        return "exam"
    elif msg.startswith("/解释"):
        return "explain"
    return "qa"


def _save_qa_record(question: str, answer: str, course: str | None,
                    sources: list[str] | None = None, msg_type: str = "qa"):
    try:
        save_record(question=question, answer=answer, course=course or "",
                    sources=sources, msg_type=msg_type)
    except Exception:
        pass


def _list_sections_safe(course):
    from ingestion.indexer import list_sections
    try:
        return list_sections(course)
    except Exception:
        return []


def _course_count(course):
    stats = get_course_stats()
    return stats.get(course, 0)


def _build_file_choices(course):
    """返回课程的文件列表（供文件管理下拉使用）。"""
    if not course or course == "全部":
        return []
    try:
        return list_sources(course)
    except Exception:
        return []


# ── chat callbacks ──────────────────────────────────────

def send_message(message, chat_history, chat_course):
    if not message.strip():
        yield chat_history, ""
        return

    course_name = None if chat_course == "全部" else chat_course

    # Step 1: Context
    context = get_context_prompt(course_name)

    # Step 2: Intent routing
    intent = route_intent(message, context_prompt=context)

    # ── help ──
    if intent["intent"] == "help":
        reply = """## 使用帮助

**用自然语言学习。**
* "总结第二章" → 快速掌握章节内容
* "出5道选择题" → 即刻开始练习
* "解释红黑树" → 理解复杂概念
* "标记XX为薄弱点" → 记录学习重点
* "我的薄弱点有哪些" → 回顾掌握情况
* "有哪些文件" → 查看课程资料


支持的文件操作：上传 PDF/PPT/PPTX，删除课程/文件。"""
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        yield chat_history, ""
        return

    # ── history ──
    if intent["intent"] == "history":
        target = intent.get("chapter") or course_name or ""
        records = get_history(course=target, limit=20)
        if not records:
            reply = "暂无学习记录。"
        else:
            lines = [f"## 学习记录 ({target or '全部课程'})\n"]
            for r in records:
                ts = r["timestamp"][:19].replace("T", " ")
                c = r.get("course", "")
                q = r["question"][:80]
                lines.append(f"- **{ts}** [{c}] {q}")
            reply = "\n".join(lines)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        yield chat_history, ""
        return

    # ── mark_mastery ──
    if intent["intent"] == "mark_mastery":
        concept = intent.get("concept", "")
        level = intent.get("mastery_level", "unmarked")
        if not concept:
            reply = "请说明要标记哪个知识点，例如：\"标记死锁为薄弱点\""
        else:
            mark_mastery(course_name, concept, level)
            level_label = {"mastered": "[已完成] 已掌握", "weak": "[注意] 薄弱点", "unmarked": "-- 未标记"}
            reply = f"已将「{concept}」标记为：{level_label.get(level, level)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        _save_qa_record(message, reply, course_name, msg_type="mastery")
        yield chat_history, ""
        return

    # ── query_weak ──
    if intent["intent"] == "query_weak":
        if not course_name:
            reply = "请先在顶部下拉菜单选择课程。"
        else:
            weak_list = get_weak_concepts(course_name)
            if weak_list:
                lines = [f"## 课程「{course_name}」的薄弱知识点\n"]
                for i, c in enumerate(weak_list, 1):
                    lines.append(f"{i}. {c}")
                lines.append(f"\n共 {len(weak_list)} 个薄弱知识点需要加强。")
                reply = "\n".join(lines)
            else:
                reply = f"课程「{course_name}」暂无薄弱知识点。\n\n输入\"标记XX为薄弱点\"来标记。"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        _save_qa_record(message, reply, course_name, msg_type="query_weak")
        yield chat_history, ""
        return

    # ── course_mgmt ──
    if intent["intent"] == "course_mgmt":
        sources = list_sources(course_name) if course_name else []
        from ingestion.indexer import list_sections
        sections = list_sections(course_name) if course_name else []
        lines = [f"## 课程「{course_name or '全部'}」\n"]
        if sources:
            lines.append("**文件列表：**")
            for s in sources:
                cnt = get_source_count(course_name, s)
                lines.append(f"- {s} ({cnt} chunks)")
        else:
            lines.append("暂无文件。")
        if sections:
            lines.append("\n**章节：**")
            for s in sections:
                lines.append(f"- {s}")
        lines.append("\n**提示：** 上传 PDF 请点击上传按钮")
        reply = "\n".join(lines)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        yield chat_history, ""
        return

    # ── chapter_summary ──
    if intent["intent"] == "chapter_summary":
        chapter = intent.get("chapter") or message
        if not course_name:
            reply = "请先在顶部下拉菜单选择课程，或输入课程名点击「创建」"
        else:
            reply, c_docs, c_metas, c_scores, found_content = generate_chapter_summary(course_name, chapter)
            if found_content:
                record_chapter(course_name, chapter)
            if c_docs:
                reply += _format_sources_detail(c_docs, c_metas, c_scores)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        _save_qa_record(message, reply, course_name, msg_type="chapter")
        yield chat_history, ""
        return

    # ── exam ──
    if intent["intent"] == "exam":
        if not course_name:
            reply = "请先在顶部下拉菜单选择课程，或输入课程名点击「创建」"
        else:
            chapter = intent.get("chapter") or ""
            qtype = intent.get("question_type") or "mixed"
            count = intent.get("count") or 5
            reply, e_docs, e_metas, e_scores, found_content = generate_exam_questions(
                course_name, section=chapter, question_type=qtype, count=count
            )
            if chapter and found_content:
                record_chapter(course_name, chapter)
            if e_docs:
                reply += _format_sources_detail(e_docs, e_metas, e_scores)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        _save_qa_record(message, reply, course_name, msg_type="exam")
        yield chat_history, ""
        return

    # ── explain ──
    if intent["intent"] == "explain":
        concept = intent.get("concept") or message
        if not course_name:
            reply = "请先在顶部下拉菜单选择课程，或输入课程名点击「创建」"
        else:
            reply, x_docs, x_metas, x_scores = explain_concept(course_name, concept)
            if x_docs:
                reply += _format_sources_detail(x_docs, x_metas, x_scores)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        _save_qa_record(message, reply, course_name, msg_type="explain")
        yield chat_history, ""
        return

    # ── default: qa ──
    docs, metas, scores = search(message, course=course_name)

    if not docs:
        reply = "未在当前课程资料中找到相关内容。\n\n"
        if course_name:
            secs = _list_sections_safe(course_name)
            if secs:
                reply += "**该课程已有章节：**\n"
                for s in secs:
                    reply += f"- {s}\n"
                reply += "\n建议：\n- 换个说法试试\n- 切换到「全部」检索\n- 上传更多课程资料"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        yield chat_history, ""
        return

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    full_answer = ""
    try:
        for chunk in generate_stream(message, docs, metas):
            full_answer += chunk
            chat_history[-1]["content"] = full_answer
            yield chat_history, ""
    except Exception:
        full_answer = generate(message, docs, metas)
        chat_history[-1]["content"] = full_answer
        yield chat_history, ""

    sources_html = _format_sources_detail(docs, metas, scores)
    chat_history[-1]["content"] = full_answer + sources_html

    sources_list = []
    for meta in metas:
        src = meta.get("source", "?")
        page = meta.get("page", 0)
        info = src
        if page:
            info += f" (第{page}页)"
        sources_list.append(info)
    _save_qa_record(message, full_answer, course_name, sources=sources_list)

    yield chat_history, ""


def clear_chat():
    return [], ""


# ── document management callbacks ────────────────────────

def upload_files_handler(files, course):
    PPT_EXTENSIONS = {".ppt", ".pptx"}
    ALL_SUPPORTED = {".pdf", ".pptx", ".ppt"}

    if files is None:
        return "请先选择文件", gr.update()
    if not course or course == "全部":
        return "请先选择或创建一个课程", gr.update()

    if not isinstance(files, list):
        files = [files]

    total_chunks = 0
    errors = []
    success_count = 0

    for f in files:
        path = f.name
        ext = os.path.splitext(path)[1].lower()
        source_name = os.path.basename(path)

        if ext not in ALL_SUPPORTED:
            errors.append(f"{source_name}: 不支持的文件类型 ({ext})")
            continue

        pdf_path = path
        temp_dir = None

        try:
            if ext in PPT_EXTENSIONS:
                pdf_path, temp_dir = convert_pptx_to_pdf(path)

            pages = load_pdf_with_meta(pdf_path)
            for p in pages:
                p["text"] = clean_text(p["text"])
            chunk_dicts = chunk_text_with_meta(pages)
            chunk_texts = [c["text"] for c in chunk_dicts]
            chunk_metas = [{"page": c["page"], "section": c["section"]} for c in chunk_dicts]

            index_chunks(chunk_texts, course=course, source=source_name,
                         chunk_metas=chunk_metas)
            total_chunks += len(chunk_texts)
            success_count += 1
        except Exception as e:
            errors.append(f"{source_name}: {e}")
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    if success_count > 0:
        if success_count == 1:
            msg = f"<span style='color: #16a34a; font-weight: 600;'>学习完成！已解析 {total_chunks} 个知识点，现在可以提问了</span>"
        else:
            msg = f"<span style='color: #16a34a; font-weight: 600;'>学习完成！已解析 {success_count} 个文件、{total_chunks} 个知识点，现在可以提问了</span>"
    else:
        msg = ""

    if errors:
        if msg:
            msg += f"\n\n<span style='color: #d97706; font-weight: 500;'>{len(errors)} 个文件处理失败: " + "; ".join(errors) + "</span>"
        else:
            msg = "<span style='color: #dc2626; font-weight: 600;'>文件处理失败，请检查文件格式: " + "; ".join(errors) + "</span>"

    if not msg:
        msg = "<span style='color: #dc2626; font-weight: 600;'>文件处理失败，请检查文件格式</span>"

    return msg, gr.update(choices=_build_course_choices())


def delete_course_handler(course):
    if not course or course == "全部":
        return "请选择要删除的课程", gr.update(choices=_build_course_choices(), value="全部")
    delete_course(course)
    return f"已删除课程「{course}」", gr.update(choices=_build_course_choices(), value="全部")


# ── UI ──────────────────────────────────────────────────

with gr.Blocks(title="知识，自有答案") as demo:
    gr.Markdown("""# 知识，自有答案
> 让课程资料真正成为知识。————学习，重新定义""")

    # ── Top toolbar ──
    with gr.Row():
        course_dd = gr.Dropdown(
            label="当前课程",
            choices=["全部"],
            value="全部",
            scale=3,
            interactive=True,
        )
        new_course_tb = gr.Textbox(
            label="新建课程",
            placeholder="输入课程名...",
            scale=2,
        )
        create_btn = gr.Button("创建", scale=1)
        upload_btn = gr.UploadButton(
            "上传 PDF/PPT",
            file_types=[".pdf", ".pptx", ".ppt"],
            file_count="multiple",
            scale=1,
        )

    # ── File management ──
    with gr.Accordion("文件管理", open=False):
        with gr.Row():
            file_dd = gr.Dropdown(
                label="课程文件",
                choices=[],
                value=None,
                scale=4,
                interactive=True,
            )
            file_delete_btn = gr.Button("删除选中文件", variant="stop", scale=1)
        delete_btn = gr.Button("删除整个课程", variant="stop", scale=1)

    top_msg = gr.Markdown("")

    # ── Chat area ──
    chatbot = gr.Chatbot(label="对话", height=500)

    with gr.Row():
        msg_input = gr.Textbox(
            label="用自然语言学习",
            placeholder="即刻开始，比如：总结第二章 / 出5道选择 / 解释死锁...",
            scale=5,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    # ── Quick buttons ──
    with gr.Row():
        quick_exam_btn = gr.Button("出题练习", size="sm")
        quick_weak_btn = gr.Button("薄弱点", size="sm")
        clear_btn = gr.Button("清空对话", size="sm")

    # ── State ──
    current_course_state = gr.State("全部")

    # ── Events ──

    def _on_load():
        choices = _build_course_choices()
        return gr.update(choices=choices, value="全部"), _build_guide(), gr.update(choices=[], value=None)

    demo.load(fn=_on_load, outputs=[course_dd, top_msg, file_dd])

    def _create_course(name):
        name = name.strip()
        empty_files = gr.update(choices=[], value=None)
        if not name:
            return gr.update(), gr.update(choices=_build_course_choices()), "请输入课程名称", empty_files, "全部"
        if name == "全部":
            return gr.update(), gr.update(choices=_build_course_choices()), "课程名不能为'全部'", empty_files, "全部"
        if name in list_courses():
            return gr.update(), gr.update(choices=_build_course_choices()), f"课程「{name}」已存在", empty_files, "全部"
        choices = _build_course_choices()
        if name not in choices:
            choices.append(name)
        return "", gr.update(choices=choices, value=name), f"<span style='color: #16a34a; font-weight: 600;'>课程「{name}」创建成功！请上传课件开始学习</span>", empty_files, name

    create_btn.click(
        fn=_create_course,
        inputs=[new_course_tb],
        outputs=[new_course_tb, course_dd, top_msg, file_dd, current_course_state],
    )

    def _on_course_change(course):
        welcome = _build_welcome(course)
        file_choices = _build_file_choices(course)
        file_value = file_choices[0] if file_choices else None
        return welcome, course, gr.update(choices=file_choices, value=file_value), []

    course_dd.change(
        fn=_on_course_change,
        inputs=[course_dd],
        outputs=[top_msg, current_course_state, file_dd, chatbot],
    )

    def _on_upload(files, course):
        msg, dd_update = upload_files_handler(files, course)
        welcome = _build_welcome(course)
        file_choices = _build_file_choices(course)
        file_value = file_choices[0] if file_choices else None
        return f"{msg}\n\n{welcome}", dd_update, gr.update(choices=file_choices, value=file_value)

    upload_btn.upload(
        fn=_on_upload,
        inputs=[upload_btn, current_course_state],
        outputs=[top_msg, course_dd, file_dd],
    )

    def _on_delete(course):
        empty_files = gr.update(choices=[], value=None)
        if not course or course == "全部":
            return "请先选择要删除的课程", gr.update(), "全部", empty_files
        msg, dd_update = delete_course_handler(course)
        welcome = _build_welcome(None)
        return f"{msg}\n\n{welcome}", dd_update, "全部", empty_files

    delete_btn.click(
        fn=_on_delete,
        inputs=[current_course_state],
        outputs=[top_msg, course_dd, current_course_state, file_dd],
    )

    # File delete
    def delete_file_handler(course, file):
        if not course or course == "全部":
            return "请先选择一个课程", gr.update()
        if not file:
            return "请先选择要删除的文件", gr.update()
        delete_source(course, file)
        new_choices = _build_file_choices(course)
        new_value = new_choices[0] if new_choices else None
        welcome = _build_welcome(course)
        msg = f"已删除文件「{file}」，课程「{course}」剩余 {len(new_choices)} 个文件。\n\n{welcome}"
        return msg, gr.update(choices=new_choices, value=new_value)

    file_delete_btn.click(
        fn=delete_file_handler,
        inputs=[current_course_state, file_dd],
        outputs=[top_msg, file_dd],
    )

    # Chat
    send_btn.click(
        fn=send_message,
        inputs=[msg_input, chatbot, current_course_state],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, chatbot, current_course_state],
        outputs=[chatbot, msg_input],
    )

    # Quick buttons
    quick_exam_btn.click(
        fn=lambda: "出5道选择题",
        outputs=[msg_input],
    ).then(
        fn=send_message,
        inputs=[msg_input, chatbot, current_course_state],
        outputs=[chatbot, msg_input],
    )

    quick_weak_btn.click(
        fn=lambda: "我的薄弱点有哪些",
        outputs=[msg_input],
    ).then(
        fn=send_message,
        inputs=[msg_input, chatbot, current_course_state],
        outputs=[chatbot, msg_input],
    )
    clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch(ssr_mode=False, server_name="0.0.0.0", show_error=True)
