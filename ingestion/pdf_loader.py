# -*- coding: utf-8 -*-
"""
PDF 解析器 — 提取文本、页码、章节标题。

使用 pypdf 逐页提取文本，同时尝试识别章节标题。
"""

import re
from pypdf import PdfReader


def _detect_section(text: str) -> str:
    """
    从文本中检测章节标题。

    匹配模式：
      - 第X章 / 第X节 / Chapter X / Section X
      - Markdown 风格标题（# 开头）
      - 数字编号标题（1. 1.1 1.1.1）
    """
    patterns = [
        r"第[一二三四五六七八九十百千0-9]+\s*[章节].*",
        r"Chapter\s+\d+.*",
        r"Section\s+\d+.*",
        r"^#{1,4}\s+.+",
        r"^\d+(?:\.\d+)*\s+.+",
    ]

    lines = text.strip().split("\n")
    for line in lines[:5]:  # 只检查前几行
        line = line.strip()
        if not line or len(line) < 3:
            continue
        for pat in patterns:
            if re.match(pat, line):
                return line
    return ""


def load_pdf(file_path: str) -> str:
    """
    加载 PDF 并返回纯文本（兼容旧接口）。

    参数：
      file_path: PDF 文件路径

    返回：
      所有页面的文本拼接（用换行符分隔）
    """
    pages = load_pdf_with_meta(file_path)
    return "\n\n".join(p["text"] for p in pages if p["text"].strip())


def load_pdf_with_meta(file_path: str) -> list[dict]:
    """
    加载 PDF 并返回每页的文本和元信息。

    参数：
      file_path: PDF 文件路径

    返回：
      [{"page": 页码(1-based), "text": "页面文本", "section": "检测到的章节标题"}, ...]

    设计说明：
      - 页码是 1-based（符合人类阅读习惯）
      - section 字段尝试自动检测章节标题
      - 对于扫描版 PDF，text 可能为空（pypdf 的限制）
    """
    reader = PdfReader(file_path)

    # 尝试从 PDF 元数据中获取标题
    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = reader.metadata.title.strip()

    pages = []
    current_section = meta_title  # 当前章节标题（跨页继承）

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        # 检测本章节标题
        detected = _detect_section(text)
        if detected:
            current_section = detected

        pages.append({
            "page": i + 1,  # 1-based 页码
            "text": text.strip(),
            "section": current_section or meta_title or "",
        })

    return pages


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs/test.pdf"
    pages = load_pdf_with_meta(path)
    print(f"总页数: {len(pages)}")
    for p in pages[:3]:
        print(f"\n--- 第{p['page']}页 | 章节: {p['section'][:40] if p['section'] else '无'} ---")
        print(p["text"][:200])
