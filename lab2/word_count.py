"""Лемматизация и подсчёт частоты слов."""

import csv
from collections import Counter
from collections.abc import Set
from pathlib import Path

import pymorphy3

from .preprocessing import tokenize

def lemmatize_texts(
    texts: list[str],
    keep_stop_words: bool = False,
    *,
    stop_words: Set[str] = frozenset(),
    stop_pos: Set[str] = frozenset(),
    stop_tags: Set[str] = frozenset(),
) -> list[str]:
    """Лемматизировать тексты с переданными исключениями"""
    morph = pymorphy3.MorphAnalyzer()
    cache: dict[str, str] = {}
    documents = []
    for text in texts:
        lemmas = []
        for word in tokenize(text):
            if word not in cache:
                parsed = morph.parse(word)[0]
                cache[word] = (
                    parsed.normal_form
                    if keep_stop_words or (
                        parsed.tag.POS not in stop_pos
                        and stop_tags.isdisjoint(parsed.tag.grammemes)
                        and parsed.normal_form not in stop_words
                    ) else ""
                )
            if cache[word]:
                lemmas.append(cache[word])
        documents.append(" ".join(lemmas))
    return documents


def count_lemmas(documents: list[str]) -> Counter[str]:
    """Подсчитать частоты уже лемматизированных документов"""
    return Counter(word for document in documents for word in document.split())


def save_word_counts(counts: Counter[str], filename: str | Path) -> None:
    """Сохранить частоты по убыванию, при равенстве — по алфавиту"""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["word", "count"])
        writer.writerows(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
