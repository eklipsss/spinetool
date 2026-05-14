from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import umap


def normalize_spine_key(path_value: str) -> str:
    cleaned = str(path_value).strip().replace("\\", "/")
    if cleaned.endswith(".off"):
        cleaned = cleaned[:-4]
    return cleaned


def extract_dataset_label(spine_key: str) -> str:
    parts = normalize_spine_key(spine_key).split("/")
    return parts[0] if parts else ""


def parse_numeric(value: str, take_abs: bool = False) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        number = float(text)
        return abs(number) if take_abs else number
    except ValueError:
        pass

    normalized = text.replace("i", "j")
    try:
        number = complex(normalized)
        return abs(number) if take_abs else float(number.real)
    except ValueError:
        return float("nan")


def _read_nonempty_csv_rows(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError("CSV file is empty: {}".format(csv_path))
    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def load_descriptor_csv(
    csv_path: Path,
    feature_prefix: str,
    take_abs: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    max_len = 0

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            spine_key = normalize_spine_key(row[0])
            values = [parse_numeric(value, take_abs=take_abs) for value in row[1:] if str(value).strip()]
            max_len = max(max_len, len(values))
            rows.append({"spine_key": spine_key, "values": values})

    records: List[Dict[str, float]] = []
    for row in rows:
        record: Dict[str, float] = {"spine_key": row["spine_key"]}
        for index in range(max_len):
            value = row["values"][index] if index < len(row["values"]) else float("nan")
            record["{}_{}".format(feature_prefix, index)] = value
        records.append(record)

    frame = pd.DataFrame.from_records(records).set_index("spine_key")
    frame = frame.dropna(axis=1, how="all")
    return frame


def load_scalar_metrics_csv(csv_path: Path, metric_names: Sequence[str]) -> pd.DataFrame:
    header, data_rows = _read_nonempty_csv_rows(csv_path)
    metric_indices = {name: header.index(name) for name in metric_names}

    records: List[Dict[str, float]] = []
    for row in data_rows:
        if len(row) < len(header):
            continue
        record: Dict[str, float] = {"spine_key": normalize_spine_key(row[0])}
        for metric_name, metric_index in metric_indices.items():
            record[metric_name] = parse_numeric(row[metric_index])
        records.append(record)

    frame = pd.DataFrame.from_records(records).set_index("spine_key")
    return frame


def merge_sphharm_and_volume(
    sphharm_csv_path: Path,
    metrics_csv_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sphharm = load_descriptor_csv(sphharm_csv_path, feature_prefix="sphharm", take_abs=False)
    volume = load_scalar_metrics_csv(metrics_csv_path, metric_names=["Volume"])
    features = sphharm.join(volume, how="inner")
    metadata = pd.DataFrame(
        {
            "spine_key": features.index,
            "dataset_label": [extract_dataset_label(spine_key) for spine_key in features.index],
        }
    ).set_index("spine_key")
    return features, metadata


def load_lf_modulus_features(lf_csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = load_descriptor_csv(lf_csv_path, feature_prefix="lf_modulus", take_abs=True)
    metadata = pd.DataFrame(
        {
            "spine_key": features.index,
            "dataset_label": [extract_dataset_label(spine_key) for spine_key in features.index],
        }
    ).set_index("spine_key")
    return features, metadata


def agresti_caffo_p_value(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return float("nan")

    p1 = (x1 + 1.0) / (n1 + 2.0)
    p2 = (x2 + 1.0) / (n2 + 2.0)
    variance = (p1 * (1.0 - p1) / (n1 + 2.0)) + (p2 * (1.0 - p2) / (n2 + 2.0))
    if variance <= 0 or not np.isfinite(variance):
        return float("nan")

    z_value = (p1 - p2) / math.sqrt(variance)
    return float(2.0 * norm.sf(abs(z_value)))


def run_kmeans_clustering(
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int = 0,
    dim_reduction: Optional[str] = None,
    n_components: int = 3,
) -> Dict[str, object]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features.values)

    embedding = standardized
    embedding_columns = list(features.columns)
    if dim_reduction == "umap":
        sample_count = standardized.shape[0]
        safe_components = max(1, min(int(n_components), sample_count - 1))
        if sample_count <= 2:
            embedding = standardized
            embedding_columns = list(features.columns)
        else:
            safe_neighbors = max(2, min(15, sample_count - 1))
            reducer = umap.UMAP(
                n_components=safe_components,
                n_neighbors=safe_neighbors,
                random_state=random_state,
                init="random",
            )
            embedding = reducer.fit_transform(standardized)
            embedding_columns = ["UMAP_{}".format(index + 1) for index in range(safe_components)]
    elif dim_reduction == "pca":
        sample_count = standardized.shape[0]
        safe_components = max(1, min(int(n_components), sample_count, standardized.shape[1]))
        reducer = PCA(n_components=safe_components, random_state=random_state)
        embedding = reducer.fit_transform(standardized)
        embedding_columns = ["PCA_{}".format(index + 1) for index in range(safe_components)]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    cluster_labels = kmeans.fit_predict(embedding)

    cluster_frame = pd.DataFrame(
        {
            "spine_key": features.index,
            "cluster_id": cluster_labels.astype(int),
        }
    ).set_index("spine_key")

    embedding_frame = pd.DataFrame(embedding, index=features.index, columns=embedding_columns)

    return {
        "cluster_frame": cluster_frame,
        "embedding_frame": embedding_frame,
        "standardized": standardized,
        "embedding_columns": embedding_columns,
    }


def evaluate_cluster_significance(
    cluster_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, object]:
    merged = cluster_frame.join(metadata, how="left")
    label_values = sorted(value for value in merged["dataset_label"].dropna().unique())
    cluster_ids = sorted(int(value) for value in merged["cluster_id"].dropna().unique())

    pair_rows: List[Dict[str, object]] = []
    agresti_rows: List[Dict[str, object]] = []
    significant_clusters = set()

    for index_a, cluster_a in enumerate(cluster_ids):
        for cluster_b in cluster_ids[index_a + 1 :]:
            pair_subset = merged[merged["cluster_id"].isin([cluster_a, cluster_b])].copy()
            contingency = pd.crosstab(pair_subset["cluster_id"], pair_subset["dataset_label"])

            chi_square_p = float("nan")
            if contingency.shape[0] == 2 and contingency.shape[1] >= 2:
                chi_square_p = float(chi2_contingency(contingency.values)[1])

            pair_label_rows: List[Dict[str, object]] = []
            for label_value in label_values:
                x1 = int(((pair_subset["cluster_id"] == cluster_a) & (pair_subset["dataset_label"] == label_value)).sum())
                n1 = int((pair_subset["cluster_id"] == cluster_a).sum())
                x2 = int(((pair_subset["cluster_id"] == cluster_b) & (pair_subset["dataset_label"] == label_value)).sum())
                n2 = int((pair_subset["cluster_id"] == cluster_b).sum())
                p_value = agresti_caffo_p_value(x1, n1, x2, n2)
                label_row = {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "label": label_value,
                    "cluster_a_successes": x1,
                    "cluster_a_total": n1,
                    "cluster_b_successes": x2,
                    "cluster_b_total": n2,
                    "agresti_caffo_p_value": p_value,
                }
                agresti_rows.append(label_row)
                pair_label_rows.append(label_row)

            finite_p_values = [
                row["agresti_caffo_p_value"]
                for row in pair_label_rows
                if np.isfinite(row["agresti_caffo_p_value"])
            ]
            min_p_value = min(finite_p_values) if finite_p_values else float("nan")
            significant = bool(np.isfinite(min_p_value) and min_p_value < alpha)
            if significant:
                significant_clusters.update([cluster_a, cluster_b])

            pair_rows.append(
                {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "cluster_a_size": int((merged["cluster_id"] == cluster_a).sum()),
                    "cluster_b_size": int((merged["cluster_id"] == cluster_b).sum()),
                    "agresti_caffo_min_p_value": min_p_value,
                    "chi_square_p_value": chi_square_p,
                    "statistically_distinct": significant,
                }
            )

    pairwise_results = pd.DataFrame(pair_rows)
    agresti_results = pd.DataFrame(agresti_rows)
    summary = pd.DataFrame(
        [
            {
                "num_clusters": len(cluster_ids),
                "statistically_distinct_pairs": int(pairwise_results["statistically_distinct"].sum()) if not pairwise_results.empty else 0,
                "statistically_distinct_clusters": len(significant_clusters),
                "alpha": alpha,
            }
        ]
    )
    return {
        "pairwise_results": pairwise_results,
        "agresti_results": agresti_results,
        "summary": summary,
    }


def run_experiment(
    experiment_name: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    n_clusters: int,
    output_dir: Path,
    random_state: int = 0,
    dim_reduction: Optional[str] = None,
    n_components: int = 3,
    alpha: float = 0.05,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    clustering_result = run_kmeans_clustering(
        features=features,
        n_clusters=n_clusters,
        random_state=random_state,
        dim_reduction=dim_reduction,
        n_components=n_components,
    )
    cluster_frame = clustering_result["cluster_frame"].join(metadata, how="left")
    stats_result = evaluate_cluster_significance(cluster_frame[["cluster_id"]], metadata, alpha=alpha)

    embedding = clustering_result["embedding_frame"].values
    labels = cluster_frame["cluster_id"].to_numpy()
    unique_labels = np.unique(labels)
    silhouette_value = float("nan")
    davies_bouldin_value = float("nan")
    calinski_harabasz_value = float("nan")
    if embedding.shape[0] >= 2 and unique_labels.size >= 2 and unique_labels.size < embedding.shape[0]:
        silhouette_value = float(silhouette_score(embedding, labels))
        davies_bouldin_value = float(davies_bouldin_score(embedding, labels))
        calinski_harabasz_value = float(calinski_harabasz_score(embedding, labels))

    cluster_sizes = (
        cluster_frame["cluster_id"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster_id")
        .reset_index(name="size")
    )
    unsupervised_summary = pd.DataFrame(
        [
            {
                "num_clusters": int(unique_labels.size),
                "silhouette_score": silhouette_value,
                "davies_bouldin_score": davies_bouldin_value,
                "calinski_harabasz_score": calinski_harabasz_value,
                "num_samples": int(embedding.shape[0]),
            }
        ]
    )

    cluster_frame.to_csv(output_dir / "{}_clusters.csv".format(experiment_name))
    clustering_result["embedding_frame"].to_csv(output_dir / "{}_embedding.csv".format(experiment_name))
    stats_result["summary"].to_csv(output_dir / "{}_summary.csv".format(experiment_name), index=False)
    stats_result["pairwise_results"].to_csv(output_dir / "{}_pairwise_pvalues.csv".format(experiment_name), index=False)
    stats_result["agresti_results"].to_csv(output_dir / "{}_agresti_caffo_pvalues.csv".format(experiment_name), index=False)
    unsupervised_summary.to_csv(output_dir / "{}_unsupervised_summary.csv".format(experiment_name), index=False)
    cluster_sizes.to_csv(output_dir / "{}_cluster_sizes.csv".format(experiment_name), index=False)

    return {
        "cluster_frame": cluster_frame,
        "embedding_frame": clustering_result["embedding_frame"],
        "summary": stats_result["summary"],
        "pairwise_results": stats_result["pairwise_results"],
        "agresti_results": stats_result["agresti_results"],
        "unsupervised_summary": unsupervised_summary,
        "cluster_sizes": cluster_sizes,
        "experiment_name": experiment_name,
        "output_dir": output_dir,
    }


def plot_3d_embedding(result: Dict[str, object], title: str) -> object:
    embedding_frame = result["embedding_frame"].copy()
    cluster_frame = result["cluster_frame"]
    merged = embedding_frame.join(cluster_frame[["cluster_id", "dataset_label"]], how="left")

    if embedding_frame.shape[1] < 3:
        raise ValueError("3D plot requires at least 3 embedding dimensions.")

    return px.scatter_3d(
        merged,
        x=embedding_frame.columns[0],
        y=embedding_frame.columns[1],
        z=embedding_frame.columns[2],
        color="cluster_id",
        symbol="dataset_label",
        title=title,
        hover_name=merged.index,
    )


def summarize_result(result: Dict[str, object]) -> pd.DataFrame:
    return result["summary"]


def sweep_cluster_counts(
    features: pd.DataFrame,
    cluster_counts: Iterable[int],
    random_state: int = 0,
    dim_reduction: Optional[str] = None,
    n_components: int = 3,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for n_clusters in cluster_counts:
        result = run_kmeans_clustering(
            features=features,
            n_clusters=int(n_clusters),
            random_state=random_state,
            dim_reduction=dim_reduction,
            n_components=n_components,
        )

        embedding = result["embedding_frame"].values
        labels = result["cluster_frame"]["cluster_id"].to_numpy()
        unique_labels = np.unique(labels)

        silhouette_value = float("nan")
        davies_bouldin_value = float("nan")
        calinski_harabasz_value = float("nan")
        if embedding.shape[0] >= 2 and unique_labels.size >= 2 and unique_labels.size < embedding.shape[0]:
            silhouette_value = float(silhouette_score(embedding, labels))
            davies_bouldin_value = float(davies_bouldin_score(embedding, labels))
            calinski_harabasz_value = float(calinski_harabasz_score(embedding, labels))

        rows.append(
            {
                "n_clusters": int(n_clusters),
                "silhouette_score": silhouette_value,
                "davies_bouldin_score": davies_bouldin_value,
                "calinski_harabasz_score": calinski_harabasz_value,
            }
        )

    return pd.DataFrame(rows)


def plot_cluster_sweep(sweep_frame: pd.DataFrame, title_prefix: str) -> object:
    melted = sweep_frame.melt(
        id_vars=["n_clusters"],
        value_vars=[
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
        ],
        var_name="metric",
        value_name="value",
    )
    return px.line(
        melted,
        x="n_clusters",
        y="value",
        color="metric",
        markers=True,
        title="{}: cluster-count sweep".format(title_prefix),
    )


def choose_best_cluster_counts(sweep_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    metric_rules = [
        ("silhouette_score", "max"),
        ("davies_bouldin_score", "min"),
        ("calinski_harabasz_score", "max"),
    ]

    for metric_name, rule in metric_rules:
        metric_frame = sweep_frame[["n_clusters", metric_name]].dropna()
        if metric_frame.empty:
            best_k = float("nan")
            best_value = float("nan")
        elif rule == "max":
            best_index = metric_frame[metric_name].idxmax()
            best_k = int(metric_frame.loc[best_index, "n_clusters"])
            best_value = float(metric_frame.loc[best_index, metric_name])
        else:
            best_index = metric_frame[metric_name].idxmin()
            best_k = int(metric_frame.loc[best_index, "n_clusters"])
            best_value = float(metric_frame.loc[best_index, metric_name])

        rows.append(
            {
                "metric": metric_name,
                "optimization": rule,
                "best_n_clusters": best_k,
                "best_value": best_value,
            }
        )

    return pd.DataFrame(rows)
