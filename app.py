# -*- coding: utf-8 -*-
import sys

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from ingestion.indexer import index_chunks, collection
from retrieval.search import search
from llm.dashscope_llm import generate
from utils.text_clean import clean_text
from config import NO_RESULT_MSG


def run_ingestion(pdf_path="data/pdfs/test.pdf"):
    print(f"加载 PDF: {pdf_path}")
    text = load_pdf(pdf_path)

    print("清洗文本...")
    text = clean_text(text)

    print("切分 chunk...")
    chunks = chunk_text(text)

    print("向量化并入库...")
    index_chunks(chunks, source=pdf_path)


def query_loop():
    print("\n" + "=" * 50)
    print("RAG 问答系统已就绪")
    print(f"当前库中 chunk 数: {collection.count()}")
    print("输入问题后回车，输入 /exit 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not query:
            continue
        if query == "/exit":
            print("再见")
            break

        print("检索中...")
        docs, metas, scores = search(query)

        if not docs:
            print(f"\n{NO_RESULT_MSG}\n")
            continue

        print(f"找到 {len(docs)} 个相关片段（已过滤低相关结果）：")
        for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
            preview = doc[:100].replace("\n", " ")
            src = meta.get("source", "?")
            page = meta.get("page", 0)
            section = meta.get("section", "")

            source_info = f"[{src}]"
            if section:
                source_info += f" · {section}"
            if page:
                source_info += f" · 第{page}页"

            print(f"  [{i + 1}] 相似度={score:.3f} | {source_info} | {preview}...")

        print("\n生成回答中...")
        answer = generate(query, docs, metas)
        print(f"\n回答：\n{answer}\n")


def main():
    if collection.count() == 0:
        print("数据库为空，开始自动入库...")
        try:
            run_ingestion()
        except Exception as e:
            print(f"入库失败: {e}")
            print("请检查 PDF 文件和 API Key 配置后重试。")
            sys.exit(1)

    query_loop()


if __name__ == "__main__":
    main()
