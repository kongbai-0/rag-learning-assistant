# -*- coding: utf-8 -*-
"""
PPT/PPTX → PDF 转换模块。

使用 LibreOffice headless 模式将 PowerPoint 文件精准转换为 PDF，
然后交由现有的 PDF 处理管道（load_pdf_with_meta → clean_text → chunk → index）。
"""

import os
import shutil
import subprocess
import tempfile

# 转换超时（秒）
TIMEOUT = 120

# 支持的 PowerPoint 扩展名
PPT_EXTENSIONS = {".ppt", ".pptx"}


def _find_libreoffice() -> str:
    """
    查找 LibreOffice soffice 二进制文件路径。

    查找优先级：
      1. macOS App Bundle 内的 soffice
      2. PATH 环境变量中的 soffice
      3. PATH 环境变量中的 libreoffice

    Returns:
        soffice 二进制文件的绝对路径。

    Raises:
        FileNotFoundError: 未找到 LibreOffice 安装。
    """
    # macOS: LibreOffice App Bundle
    mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if os.path.isfile(mac_path) and os.access(mac_path, os.X_OK):
        return mac_path

    # Linux / Homebrew: soffice on PATH
    soffice = shutil.which("soffice")
    if soffice:
        return soffice

    # Fallback: libreoffice on PATH (older installs)
    libreoffice = shutil.which("libreoffice")
    if libreoffice:
        return libreoffice

    raise FileNotFoundError(
        "未找到 LibreOffice。\n"
        "macOS 用户请访问 https://www.libreoffice.org/download/ 下载安装。\n"
        "Linux 用户请执行: sudo apt install libreoffice-impress"
    )


def convert_pptx_to_pdf(pptx_path: str) -> tuple[str, str]:
    """
    使用 LibreOffice headless 将 PPT/PPTX 文件转换为 PDF。

    每次转换在独立的临时目录中进行，避免并发上传时的文件名冲突。

    Args:
        pptx_path: .ppt 或 .pptx 文件的绝对路径。

    Returns:
        (pdf_path, temp_dir) 二元组：
          - pdf_path: 生成的 PDF 文件的绝对路径
          - temp_dir:  临时目录路径（由调用方负责清理）

    Raises:
        FileNotFoundError: 输入文件不存在或扩展名不支持。
        RuntimeError: LibreOffice 转换失败（非零退出码）。
        subprocess.TimeoutExpired: 转换超时。
    """
    if not os.path.isfile(pptx_path):
        raise FileNotFoundError(f"文件不存在: {pptx_path}")

    ext = os.path.splitext(pptx_path)[1].lower()
    if ext not in PPT_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}，仅支持 {PPT_EXTENSIONS}")

    soffice = _find_libreoffice()

    # 创建隔离的临时目录，防止并发上传同名文件冲突
    temp_dir = tempfile.mkdtemp(prefix="rag_ppt_")

    try:
        cmd = [
            soffice,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", temp_dir,
            pptx_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "未知错误"
            raise RuntimeError(f"PPT/PPTX 转换失败: {stderr}")

        # LibreOffice 输出文件名 = 输入文件 stem + .pdf
        stem = os.path.splitext(os.path.basename(pptx_path))[0]
        pdf_path = os.path.join(temp_dir, f"{stem}.pdf")

        if not os.path.isfile(pdf_path):
            # LibreOffice 有时在文件名末尾追加额外扩展名（如 .pptx.pdf）
            alt_path = os.path.join(temp_dir, f"{stem}{ext}.pdf")
            if os.path.isfile(alt_path):
                pdf_path = alt_path
            else:
                # 列出临时目录中的文件以便调试
                files_in_dir = os.listdir(temp_dir)
                raise RuntimeError(
                    f"未找到 LibreOffice 生成的 PDF。"
                    f"临时目录内容: {files_in_dir}"
                )

        if os.path.getsize(pdf_path) == 0:
            raise RuntimeError("生成的 PDF 文件为空")

        return pdf_path, temp_dir

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"PPT/PPTX 转换超时（{TIMEOUT} 秒），文件可能过大。"
            f"建议拆分幻灯片后重试。"
        )
    except Exception:
        # 转换失败时清理临时目录
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
