"""TF-IDF, сравнение кластеризации и визуализация"""

import csv
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits


def build_features(
    documents: list[str],
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[csr_matrix, TfidfVectorizer]:
    """Построить разреженную матрицу: строки — посты, столбцы — термины"""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        min_df=min_df, max_df=max_df, max_features=max_features,
        ngram_range=ngram_range, norm="l2",
    )
    matrix = vectorizer.fit_transform(documents).tocsr()
    return matrix, vectorizer


def cluster_documents(matrix, algorithms: Mapping[str, BaseEstimator], max_clusters: int = 8):
    """Подобрать k; модели без n_clusters запустить с их настройками"""
    results = {}
    comparisons = []
    dense = matrix.toarray()
    with threadpool_limits(limits=1):
        for name, estimator in algorithms.items():
            candidates = []
            model_comparisons = []
            cluster_counts = (range(2, min(max_clusters, matrix.shape[0] - 1) + 1) if "n_clusters" in estimator.get_params() else [None])
            for k in cluster_counts:
                model = clone(estimator)
                if k is not None:
                    model.set_params(n_clusters=k)
                labels = model.fit_predict(dense)
                clustered = labels != -1
                actual_k = len(np.unique(labels[clustered]))
                score = None
                if 2 <= actual_k < clustered.sum():
                    score = float(silhouette_score(matrix[clustered], labels[clustered], metric="cosine"))
                candidates.append((score, labels, k))
                model_comparisons.append({
                    "algorithm": name, 
                    "requested_clusters": k,
                    "actual_clusters": actual_k, 
                    "noise_count": int((~clustered).sum()),
                    "silhouette_cosine": score,
                    "selected": False
                })
            best = max(candidates, key=lambda candidate: candidate[0] if candidate[0] is not None else -np.inf)
            results[name] = best
            for row in model_comparisons:
                row["selected"] = row["requested_clusters"] == best[2]
            comparisons.extend(model_comparisons)
    return results, comparisons


def build_cluster_tables(texts, matrix, vectorizer, results):
    """Вернуть распределение документов и размеры кластеров с ключевыми словами"""
    assignments, sizes = [], []
    terms = vectorizer.get_feature_names_out()
    for name, (score, labels, _) in results.items():
        assignments.extend({
            "algorithm": name, 
            "document_index": index,
            "cluster": int(labels[index]), 
            "text": text
        } for index, text in enumerate(texts))
        
        for cluster in np.unique(labels):
            weights = np.asarray(matrix[labels == cluster].mean(axis=0)).ravel()
            top = [terms[index] for index in np.argsort(-weights) if weights[index] > 0][:7]
            sizes.append({
                "algorithm": name, 
                "cluster": int(cluster),
                "document_count": int(np.sum(labels == cluster)),
                "top_terms": ", ".join(top)
            })
    return assignments, sizes


def plot_clusters(matrix, results):
    """Вернуть фигуру PCA для показа в notebook или сохранения"""
    # Проекция служит только для рисунка, кластеризация проводится по всем признакам
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(matrix.toarray())
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6), squeeze=False, layout="constrained")
    for ax, (name, (score, labels, _)) in zip(axes.ravel(), results.items()):
        for cluster in np.unique(labels):
            mask = labels == cluster
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=60, alpha=0.85, label=f"Кластер {cluster} (n={mask.sum()})")
        score_text = f"{score:.3f}" if score is not None else "не определён"
        actual_k = len(set(labels) - {-1})
        ax.set_title(f"{name}\nk={actual_k}, силуэт={score_text}")
        ax.set_xlabel("Главная компонента 1")
        ax.set_ylabel("Главная компонента 2")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Кластеризация постов · PCA: {pca.explained_variance_ratio_.sum():.1%} дисперсии")
    return fig


def write_csv(path: Path, fields: list[str], rows) -> None:
    """Записать итоговую таблицу"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_results(output: str | Path, assignments, sizes, comparisons, figure) -> None:
    """Сохранить итоговые таблицы кластеризации и график"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "clusters.csv", ["algorithm", "document_index", "cluster", "text"], assignments)
    write_csv(output / "cluster_sizes.csv", ["algorithm", "cluster", "document_count", "top_terms"], sizes)
    write_csv(output / "comparison.csv", ["algorithm", "requested_clusters", "actual_clusters", "noise_count", "silhouette_cosine", "selected"], comparisons)
    figure.savefig(output / "clusters.png", dpi=160)
