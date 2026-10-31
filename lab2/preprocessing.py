"""Начальная предобработка текстов постов"""

import csv
import re
from html import unescape
from pathlib import Path


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
VK_LINK_PATTERN = re.compile(r"\[(?:id|club|public|event)\d+\|([^\]]+)\]")
WORD_PATTERN = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Очистить текст и вернуть слова в исходном порядке, сохраняя повторы"""
    text = unescape(text)
    text = VK_LINK_PATTERN.sub(r"\1", text)
    text = URL_PATTERN.sub(" ", text)
    return WORD_PATTERN.findall(text.lower())


def load_texts(filename: str | Path) -> list[str]:
    """Извлечь поле text из CSV"""
    with open(filename, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        return [row["text"] for row in reader]


def preprocess_texts(texts: list[str]) -> list[list[str]]:
    """Вернуть очищенные документы без пустых строк и повторов"""
    documents: list[str] = []
    seen: set[str] = set()
    for text in texts:
        document = " ".join(tokenize(text))
        if document and document not in seen:
            documents.append(document)
            seen.add(document)
    return documents
