from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from .columns import Col


def get_encode_devices() -> list[str] | str:
    if torch.cuda.is_available():
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        return devices if len(devices) > 1 else devices[0]
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

EMBEDDING_MODEL_ID = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384


def add_title_embeddings(ds):
    model = SentenceTransformer(EMBEDDING_MODEL_ID)

    keep_mask = ds[Col.KEEP]
    active_indices = [i for i, keep in enumerate(keep_mask) if keep]
    active_titles = [ds[i][Col.BOOK_TITLE] for i in active_indices]

    print(f"Embedding {len(active_titles)} titles...")
    computed_vecs = model.encode(active_titles, batch_size=256, show_progress_bar=True, device=get_encode_devices())

    all_embeddings = np.zeros((len(ds), EMBEDDING_DIM))
    for i, global_idx in enumerate(active_indices):
        all_embeddings[global_idx] = computed_vecs[i]

    return ds.add_column(Col.TITLE_EMBEDDING, all_embeddings.tolist())


TITLE_CLUSTER_DISTANCE_THRESHOLD = 0.25
TITLE_CLUSTER_METRIC = "cosine"
TITLE_CLUSTER_LINKAGE = "average"


def build_title_clusters(ds, threshold=TITLE_CLUSTER_DISTANCE_THRESHOLD):
    keep_mask = np.array(ds[Col.KEEP])
    if not keep_mask.any():
        print("No active books to cluster.")
        return ds

    active_indices = np.where(keep_mask)[0]
    active_embeddings = np.array(ds[Col.TITLE_EMBEDDING])[active_indices]

    print(f"Clustering {len(active_embeddings)} active titles...")
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric=TITLE_CLUSTER_METRIC,
        linkage=TITLE_CLUSTER_LINKAGE,
    ).fit_predict(active_embeddings)

    cluster_counts = Counter(labels)

    cluster_ids = np.full(len(ds), -1, dtype=int)
    cluster_sizes = np.zeros(len(ds), dtype=int)
    cluster_ids[active_indices] = labels
    cluster_sizes[active_indices] = [cluster_counts[l] for l in labels]

    # keep first occurrence of each cluster, mark the rest as duplicates
    seen_clusters: set[int] = set()
    duplicate_indices: set[int] = set()
    for global_idx, label in zip(active_indices, labels):
        if label in seen_clusters:
            duplicate_indices.add(int(global_idx))
        else:
            seen_clusters.add(label)

    def annotate(book, idx):
        book[Col.CLUSTER_ID] = int(cluster_ids[idx])
        book[Col.CLUSTER_SIZE] = int(cluster_sizes[idx])
        if idx in duplicate_indices:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "dedup_title_cluster"
        return book

    return ds.map(annotate, with_indices=True, num_proc=1, desc="annotating clusters")


def write_clusters_stats(ds, output_dir: Path):
    cluster_titles: dict[int, list[str]] = defaultdict(list)
    for row in ds:
        c_id = row[Col.CLUSTER_ID]
        if c_id != -1 and row[Col.CLUSTER_SIZE] > 1:
            cluster_titles[c_id].append(row[Col.BOOK_TITLE])

    sorted_clusters = sorted(cluster_titles.items(), key=lambda x: len(x[1]), reverse=True)

    total_books = sum(len(titles) for _, titles in sorted_clusters)
    total_dropped = total_books - len(sorted_clusters)

    path = output_dir / "cluster_stats.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"clusters={len(sorted_clusters)}  books_in_clusters={total_books}  dropped={total_dropped}\n")
        f.write(f"Agglomerative Clustering:\n  distance_threshold={TITLE_CLUSTER_DISTANCE_THRESHOLD}\n  metric={TITLE_CLUSTER_METRIC}\n  linkage={TITLE_CLUSTER_LINKAGE}\n\n")
        for rank, (c_id, titles) in enumerate(sorted_clusters, 1):
            f.write(f"{rank}. cluster={c_id}  size={len(titles)}\n")
            for t in titles:
                f.write(f"    {t}\n")
            f.write("\n")
    print(f"Cluster stats -> {path}")
