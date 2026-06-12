# -*- coding: utf-8 -*-
import re


def clean_text(text: str) -> str:
    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Remove Unicode replacement characters and private-use areas
    text = text.replace("�", "")

    # Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize multiple spaces/tabs to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace per line, keep paragraphs
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Remove lines that are purely garbled (very short + unusual char mix)
    text = re.sub(r"^\s*[^一-鿿\w\s]{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Collapse resulting empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text
