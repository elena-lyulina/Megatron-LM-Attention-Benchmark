from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import numpy as np

# 1. Load a fast, reliable model
# 'all-MiniLM-L6-v2' is the best balance of speed/accuracy for titles
model = SentenceTransformer('all-MiniLM-L6-v2')


def group_titles(titles, threshold=0.35):
    # 2. Vectorize and Normalize
    # Normalizing allows us to use 'cosine' distance effectively
    embeddings = model.encode(titles, batch_size=64, show_progress_bar=True)
    embeddings = normalize(embeddings)

    # 3. Cluster
    # distance_threshold=0.15 is quite strict (high similarity required)
    # linkage='average' prevents the "chaining effect"
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )

    labels = clusterer.fit_predict(embeddings)

    # 4. Organize results
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append((idx, titles[idx]))

    return clusters


# Example Gutenberg-style mess:
my_titles = [
    "The United States Constitution",
    "Constitution of the United States",
    "The Constitution of the U.S.A.",
    "Moby Dick; Or, The Whale",
    "Moby Dick",
    "Pride and Prejudice",
    "History of the World Vol 1",
    "History of the World Vol 2"
]

groups = group_titles(my_titles)

for cluster_id, items in groups.items():
    print(f"Group {cluster_id}: {[t for _, t in items]}")
