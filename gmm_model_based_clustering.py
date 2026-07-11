from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from scipy.linalg import eigh
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import subprocess
import tempfile
import json

try:
    import umap
except ImportError:  # pragma: no cover - optional runtime dependency
    umap = None


HUMAN_DATASET_PATTERN = re.compile(
    r"Human_age_?(?P<age>\d+)_(?P<compartment>apical|basal)_OFF",
    flags=re.IGNORECASE,
)
MOUSE_APICAL_DATASET_PATTERN = re.compile(
    r"Mouse_Apical/Mouse_Apical_(?P<series>\d+)",
    flags=re.IGNORECASE,
)
HUMAN_DATASETS = [
    "Human_age40_apical_OFF",
    "Human_age40_basal_OFF",
    "Human_age85_apical_OFF",
    "Human_age85_basal_OFF",
]
MOUSE_DATASETS = [
    "Mouse_Apical/Mouse_Apical_1",
    "Mouse_Apical/Mouse_Apical_2",
    "Mouse_Apical/Mouse_Apical_3",
    "Mouse_basal",
]
HUMAN_AND_MOUSE_DATASETS = [*HUMAN_DATASETS, *MOUSE_DATASETS]
# Backwards-compatible name used by the original human-only notebook cells.
DEFAULT_DATASETS = HUMAN_DATASETS
DEFAULT_THRESHOLDS = (0.99, 0.9, 0.8, 0.7, 0.6, 0.5)
DEFAULT_COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
CLUSTER_COLORS = [
    "#4b4b8f",
    "#e6830f",
    "#E15759",
    "#008695",
    "#80bb5a",
    "#f2b800",
    "#803c8d",
    "#FF9DA7",
    "#f97b72",
    "#cf1c90",
    "#1F77B4",
]


def log_progress(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


def normalize_spine_key(path_value: str) -> str:
    cleaned = str(path_value).strip().replace("\\", "/")
    if cleaned.endswith(".off"):
        cleaned = cleaned[:-4]
    return cleaned


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def resolve_spine_mesh_path(base_dir: Path, spine_key: str) -> Path:
    normalized = normalize_spine_key(spine_key)
    parts = normalized.split("/")
    if len(parts) < 2:
        raise ValueError(f"Spine key must include a dataset path: {spine_key}")
    dataset_parts = parts[:-1]
    spine_name = parts[-1]
    dataset_dir = base_dir.joinpath(*dataset_parts)
    candidates = [
        dataset_dir / "z-corr" / f"{spine_name}.off",
        dataset_dir / f"{spine_name}.off",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mesh file for spine: {spine_key}")


def load_off_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with mesh_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header not in {"OFF", "COFF"}:
            raise ValueError(f"Unsupported OFF header in {mesh_path}: {header}")

        counts_line = handle.readline().strip()
        while counts_line.startswith("#") or not counts_line:
            counts_line = handle.readline().strip()
        num_vertices, num_faces, *_ = map(int, counts_line.split())

        vertices = []
        for _ in range(num_vertices):
            parts = handle.readline().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

        faces = []
        for _ in range(num_faces):
            parts = handle.readline().split()
            face_size = int(parts[0])
            if face_size != 3:
                continue
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def parse_dataset_metadata(dataset_name: str) -> dict[str, object]:
    normalized_name = str(dataset_name).strip().replace("\\", "/").strip("/")
    human_match = HUMAN_DATASET_PATTERN.fullmatch(normalized_name)
    if human_match:
        age = int(human_match.group("age"))
        compartment = human_match.group("compartment").lower()
        return {
            "dataset": normalized_name,
            "type": "human",
            "age": age,
            "compartment": compartment,
            "age_label": str(age),
            "group": f"{compartment}{age}",
        }

    if MOUSE_APICAL_DATASET_PATTERN.fullmatch(normalized_name):
        return {
            "dataset": normalized_name,
            "type": "mouse",
            "age": pd.NA,
            "compartment": "apical",
            "age_label": pd.NA,
            "group": pd.NA,
        }

    if normalized_name.lower() == "mouse_basal":
        return {
            "dataset": normalized_name,
            "type": "mouse",
            "age": pd.NA,
            "compartment": "basal",
            "age_label": pd.NA,
            "group": pd.NA,
        }

    raise ValueError(f"Cannot parse dataset metadata from name: {dataset_name}")


def ensure_metadata_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()

    if "dataset" not in frame.columns:
        if "dataset_label" in frame.columns:
            frame["dataset"] = frame["dataset_label"].astype(str)
        else:
            frame["dataset"] = [
                "/".join(normalize_spine_key(index).split("/")[:-1])
                for index in frame.index
            ]

    parsed = pd.DataFrame(
        [parse_dataset_metadata(str(value)) for value in frame["dataset"]],
        index=frame.index,
    )
    for column in ("type", "compartment", "age", "age_label", "group"):
        if column not in frame.columns:
            frame[column] = parsed[column]
        elif column in ("type", "compartment"):
            frame[column] = frame[column].fillna(parsed[column])

    frame["age"] = pd.to_numeric(frame["age"], errors="coerce").astype("Int64")
    frame["age_label"] = frame["age"].astype("string")
    frame["group"] = (
        frame["compartment"].astype("string") + frame["age_label"]
    ).where(frame["age"].notna(), pd.NA)

    return frame


def load_spherical_harmonics(csv_path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    max_len = 0

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            spine_key = normalize_spine_key(row[0])
            values = [float(value) for value in row[1:] if str(value).strip()]
            max_len = max(max_len, len(values))
            rows.append({"spine_key": spine_key, "values": values})

    records: list[dict[str, float | str]] = []
    for row in rows:
        record: dict[str, float | str] = {"spine_key": row["spine_key"]}
        values = row["values"]
        for index in range(max_len):
            record[f"sphharm_{index:03d}"] = values[index] if index < len(values) else np.nan
        records.append(record)

    frame = pd.DataFrame.from_records(records).set_index("spine_key")
    return frame.dropna(axis=1, how="all")


def load_volume_from_metrics(metrics_csv_path: Path) -> pd.Series:
    frame = pd.read_csv(metrics_csv_path)
    if "Spine File" not in frame.columns or "Volume" not in frame.columns:
        raise ValueError(f"{metrics_csv_path} must contain 'Spine File' and 'Volume' columns.")
    volume = frame[["Spine File", "Volume"]].copy()
    volume["Spine File"] = volume["Spine File"].astype(str).str.strip()
    volume["Volume"] = pd.to_numeric(volume["Volume"], errors="coerce")
    volume = volume.dropna(subset=["Volume"])
    volume["spine_key"] = volume["Spine File"].map(normalize_spine_key)
    volume = volume[volume["spine_key"] != "Spine File"]
    volume = volume.drop_duplicates(subset="spine_key").set_index("spine_key")["Volume"]
    volume.name = "Volume"
    return volume


def canonicalize_dataset_index(
    frame: pd.DataFrame | pd.Series,
    dataset_name: str,
) -> pd.DataFrame | pd.Series:
    normalized_dataset = str(dataset_name).strip().replace("\\", "/").strip("/")
    result = frame.copy()
    result.index = pd.Index(
        [
            f"{normalized_dataset}/{normalize_spine_key(value).split('/')[-1]}"
            for value in result.index
        ],
        name="spine_key",
    )
    if result.index.has_duplicates:
        duplicates = sorted(set(result.index[result.index.duplicated()].tolist()))
        raise ValueError(
            f"Duplicate spine names in dataset {normalized_dataset}: {duplicates[:5]}"
        )
    return result


def load_dataset_features(
    dataset_dir: Path,
    dataset_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dataset_name = dataset_dir.name if dataset_name is None else dataset_name
    metadata_info = parse_dataset_metadata(dataset_name)
    zcorr_dir = dataset_dir / "z-corr"
    metrics_path = zcorr_dir / "metrics.csv"
    sphharm_path = zcorr_dir / "spherical_garmonics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing file: {metrics_path}")
    if not sphharm_path.exists():
        raise FileNotFoundError(f"Missing file: {sphharm_path}")

    sphharm = canonicalize_dataset_index(
        load_spherical_harmonics(sphharm_path),
        dataset_name,
    )
    metrics_volume = canonicalize_dataset_index(
        load_volume_from_metrics(metrics_path),
        dataset_name,
    )
    volume = metrics_volume.reindex(sphharm.index)

    features = sphharm.copy()
    features["Volume"] = volume
    features = features.dropna(axis=0, subset=["Volume"]).sort_index()

    metadata = pd.DataFrame(index=features.index)
    for key, value in metadata_info.items():
        metadata[key] = value

    diagnostic = {
        "dataset": dataset_name,
        "sphharm_rows": int(len(sphharm)),
        "feature_rows": int(len(features)),
        "missing_volume_after_merge": int(len(sphharm) - len(features)),
        "metrics_path": str(metrics_path.relative_to(dataset_dir)),
        "sphharm_path": str(sphharm_path.relative_to(dataset_dir)),
    }
    return features, metadata, diagnostic


def load_all_datasets(
    base_dir: Path,
    dataset_names: Sequence[str] = DEFAULT_DATASETS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    log_progress(f"[1/8] Loading datasets: {len(dataset_names)} total", verbose)
    for index, dataset_name in enumerate(dataset_names, start=1):
        log_progress(f"  - Dataset {index}/{len(dataset_names)}: {dataset_name}", verbose)
        features, metadata, diagnostic = load_dataset_features(
            dataset_dir=base_dir / dataset_name,
            dataset_name=dataset_name,
        )
        feature_frames.append(features)
        metadata_frames.append(metadata)
        diagnostics.append(diagnostic)
        log_progress(
            f"    loaded {diagnostic['feature_rows']} spines "
            f"(sphharm rows={diagnostic['sphharm_rows']}, "
            f"missing after merge={diagnostic['missing_volume_after_merge']})",
            verbose,
        )

    combined_features = pd.concat(feature_frames, axis=0).sort_index()
    if combined_features.index.has_duplicates:
        duplicates = sorted(set(combined_features.index[combined_features.index.duplicated()].tolist()))
        raise ValueError(f"Duplicate spine keys across datasets: {duplicates[:5]}")
    missing_feature_count = int(combined_features.isna().sum().sum())
    if missing_feature_count:
        missing_columns = combined_features.columns[combined_features.isna().any()].tolist()
        raise ValueError(
            "Feature tables have incompatible columns or missing values after combining datasets. "
            f"Columns with missing values: {missing_columns[:10]}"
        )
    combined_metadata = pd.concat(metadata_frames, axis=0).loc[combined_features.index]
    diagnostics_frame = pd.DataFrame(diagnostics)
    log_progress(
        f"  Combined dataset shape: {combined_features.shape[0]} spines x {combined_features.shape[1]} features",
        verbose,
    )
    return combined_features, combined_metadata, diagnostics_frame


def fit_gmm_sweep(
    standardized_features: np.ndarray,
    cluster_range: Iterable[int] = range(2, 11),
    covariance_types: Sequence[str] = DEFAULT_COVARIANCE_TYPES,
    random_state: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, GaussianMixture]:
    rows: list[dict[str, object]] = []
    best_model: GaussianMixture | None = None
    best_score = -np.inf
    cluster_values = [int(value) for value in cluster_range]
    total_models = len(cluster_values) * len(covariance_types)
    model_counter = 0

    log_progress(
        f"[3/8] GMM sweep: {total_models} models "
        f"({len(cluster_values)} cluster counts x {len(covariance_types)} covariance types)",
        verbose,
    )

    for n_clusters in cluster_values:
        for covariance_type in covariance_types:
            model_counter += 1
            log_progress(
                f"  - Fitting model {model_counter}/{total_models}: "
                f"k={n_clusters}, covariance={covariance_type}",
                verbose,
            )
            model = GaussianMixture(
                n_components=int(n_clusters),
                covariance_type=covariance_type,
                n_init=5,
                max_iter=500,
                reg_covar=1e-6,
                random_state=random_state,
            )
            model.fit(standardized_features)
            sklearn_bic = float(model.bic(standardized_features))
            mclust_style_bic = -sklearn_bic
            row = {
                "n_clusters": int(n_clusters),
                "covariance_type": covariance_type,
                "bic": mclust_style_bic,
                "sklearn_bic": sklearn_bic,
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
            }
            rows.append(row)
            if mclust_style_bic > best_score:
                best_score = mclust_style_bic
                best_model = model
                log_progress(
                    f"    new best model: k={n_clusters}, covariance={covariance_type}, bic={mclust_style_bic:.3f}",
                    verbose,
                )

    if best_model is None:
        raise RuntimeError("GMM sweep produced no fitted models.")

    return pd.DataFrame(rows).sort_values(["n_clusters", "covariance_type"]).reset_index(drop=True), best_model


def responsibilities_to_frame(probabilities: np.ndarray, index: pd.Index) -> pd.DataFrame:
    columns = [f"cluster_{cluster_id}" for cluster_id in range(probabilities.shape[1])]
    frame = pd.DataFrame(probabilities, index=index, columns=columns)
    frame["assigned_cluster"] = np.argmax(probabilities, axis=1).astype(int)
    frame["p_star"] = probabilities.max(axis=1)
    return frame


def build_bic_plot(bic_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for covariance_type in bic_frame["covariance_type"].dropna().unique():
        subset = bic_frame[bic_frame["covariance_type"] == covariance_type].sort_values("n_clusters")
        ax.plot(
            subset["n_clusters"].to_numpy(),
            subset["bic"].to_numpy(),
            marker="o",
            label=str(covariance_type),
        )
    best_row = bic_frame.loc[bic_frame["bic"].idxmax()]
    ax.scatter(
        [best_row["n_clusters"]],
        [best_row["bic"]],
        s=180,
        marker="*",
        color="black",
        label="best model",
        zorder=10,
    )
    ax.set_title("BIC for GMM model sweep")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("BIC")
    ax.legend()
    fig.tight_layout()
    return fig


def build_umap_embedding(
    standardized_features: np.ndarray,
    random_state: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    if umap is None:
        raise ImportError(
            "Package 'umap-learn' is required to build the 3D UMAP projection."
        )
    log_progress("[4/8] Building UMAP(3D) embedding", verbose)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(30, max(2, standardized_features.shape[0] - 1)),
        min_dist=0.1,
        random_state=random_state,
        init="random",
    )
    return reducer.fit_transform(standardized_features)


def build_umap_plot(
    embedding: np.ndarray,
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
) -> plt.Figure:
    merged = ensure_metadata_columns(metadata).join(assignments)
    point_colors = [cluster_color(cluster_id) for cluster_id in merged["assigned_cluster"].to_numpy()]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=point_colors,
        s=18,
        alpha=0.8,
    )
    ax.set_title("UMAP(3D) colored by assigned cluster")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    cluster_ids = sorted(pd.unique(merged["assigned_cluster"]))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cluster_color(cluster_id),
            markeredgecolor=cluster_color(cluster_id),
            label=f"Cluster {cluster_id}",
        )
        for cluster_id in cluster_ids
    ]
    ax.legend(handles=handles, title="Cluster", loc="upper left")
    fig.tight_layout()
    return fig


def build_umap_type_plot(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
) -> plt.Figure:
    typed_metadata = metadata.copy()
    if "type" not in typed_metadata.columns:
        typed_metadata = ensure_metadata_columns(typed_metadata)
    type_values = typed_metadata["type"].astype("string").fillna("unknown")
    palette = {
        "human": "#1f77b4",
        "mouse": "#ff7f0e",
        "unknown": "#7f7f7f",
    }
    fallback_palette = plt.get_cmap("tab10")
    unique_types = sorted(pd.unique(type_values))
    type_colors = {
        type_name: palette.get(type_name, to_hex(fallback_palette(index % 10)))
        for index, type_name in enumerate(unique_types)
    }
    point_colors = [type_colors[type_name] for type_name in type_values]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=point_colors,
        s=18,
        alpha=0.8,
    )
    ax.set_title("UMAP(3D) colored by spine type")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=color,
            label=str(type_name),
        )
        for type_name, color in type_colors.items()
    ]
    ax.legend(handles=handles, title="Type", loc="upper left")
    fig.tight_layout()
    return fig


def build_interactive_umap_type_plot(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame | None = None,
) -> object:
    import plotly.graph_objects as go
    import plotly.io as pio

    try:
        if pio.renderers.default in ("", None):
            pio.renderers.default = "notebook_connected"
    except Exception:
        pass

    typed_metadata = metadata.copy()
    if "type" not in typed_metadata.columns:
        typed_metadata = ensure_metadata_columns(typed_metadata)

    frame = pd.DataFrame(
        embedding,
        index=typed_metadata.index,
        columns=["UMAP_1", "UMAP_2", "UMAP_3"],
    )
    frame = frame.join(typed_metadata[[column for column in ("dataset", "type", "compartment") if column in typed_metadata.columns]])
    frame["type"] = frame["type"].astype("string").fillna("unknown")

    if assignments is not None:
        available_columns = [
            column
            for column in ("assigned_cluster", "p_star")
            if column in assignments.columns
        ]
        if available_columns:
            frame = frame.join(assignments[available_columns])

    palette = {
        "human": "#1f77b4",
        "mouse": "#ff7f0e",
        "unknown": "#7f7f7f",
    }
    fallback_palette = plt.get_cmap("tab10")
    type_counts = frame["type"].value_counts()
    unique_types = sorted(
        pd.unique(frame["type"]),
        key=lambda type_name: (type_counts.get(type_name, 0), str(type_name)),
    )
    type_colors = {
        type_name: palette.get(type_name, to_hex(fallback_palette(index % 10)))
        for index, type_name in enumerate(unique_types)
    }

    fig = go.Figure()
    for type_name in unique_types:
        subset = frame[frame["type"] == type_name]
        customdata = np.column_stack(
            [
                subset.index.astype(str),
                subset["type"].astype(str),
                subset.get("dataset", pd.Series("", index=subset.index)).astype(str),
                subset.get("compartment", pd.Series("", index=subset.index)).astype(str),
                subset.get("assigned_cluster", pd.Series("", index=subset.index)).astype(str),
                subset.get("p_star", pd.Series("", index=subset.index)).astype(str),
            ]
        )
        fig.add_trace(
            go.Scatter3d(
                x=subset["UMAP_1"],
                y=subset["UMAP_2"],
                z=subset["UMAP_3"],
                mode="markers",
                name=f"{type_name} (n={len(subset)})",
                customdata=customdata,
                marker=dict(
                    size=2.5,
                    color=type_colors[type_name],
                    opacity=0.58,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "type=%{customdata[1]}<br>"
                    "dataset=%{customdata[2]}<br>"
                    "compartment=%{customdata[3]}<br>"
                    "assigned_cluster=%{customdata[4]}<br>"
                    "p_star=%{customdata[5]}<br>"
                    "UMAP-1=%{x:.3f}<br>"
                    "UMAP-2=%{y:.3f}<br>"
                    "UMAP-3=%{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            "Interactive UMAP(3D) colored by spine type"
            + " — "
            + ", ".join(f"{type_name}: n={int(type_counts[type_name])}" for type_name in unique_types)
        ),
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
            aspectmode="cube",
        ),
        legend=dict(title="Type"),
        height=780,
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return fig


def distribution_table(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str | None = None,
) -> pd.DataFrame:
    merged = ensure_metadata_columns(metadata).join(assignments[["assigned_cluster"]])
    if group_column is None:
        counts = merged["assigned_cluster"].value_counts().sort_index()
        total = counts.sum()
        return pd.DataFrame(
            {
                "cluster": counts.index.astype(int),
                "count": counts.to_numpy(),
                "percentage": counts.to_numpy() / total * 100.0,
                "group": "all",
            }
        )

    grouped = merged.dropna(subset=[group_column])
    if grouped.empty:
        return pd.DataFrame(columns=["group", "cluster", "percentage", "count"])

    contingency = (
        pd.crosstab(grouped[group_column], grouped["assigned_cluster"])
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    percentages = contingency.div(contingency.sum(axis=1), axis=0) * 100.0
    frame = percentages.reset_index().melt(
        id_vars=[group_column],
        var_name="cluster",
        value_name="percentage",
    )
    counts = contingency.reset_index().melt(
        id_vars=[group_column],
        var_name="cluster",
        value_name="count",
    )
    frame["count"] = counts["count"]
    frame["cluster"] = frame["cluster"].astype(int)
    if group_column == "group":
        return frame
    frame["group"] = frame[group_column]
    return frame.drop(columns=[group_column])


def build_histogram(distribution: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    clusters = sorted(pd.unique(distribution["cluster"]))
    cluster_positions = np.arange(len(clusters), dtype=float)
    group_values = list(pd.unique(distribution["group"]))
    hatches = ["", "//", "\\\\", "xx", "..", "++"]

    if len(group_values) == 1:
        subset = distribution.sort_values("cluster")
        ax.bar(
            cluster_positions,
            subset["percentage"].to_numpy(dtype=float),
            color=[cluster_color(cluster_id) for cluster_id in subset["cluster"].to_numpy()],
            width=0.7,
        )
    else:
        width = 0.8 / max(len(group_values), 1)
        for group_index, group_value in enumerate(group_values):
            subset = (
                distribution[distribution["group"] == group_value]
                .set_index("cluster")
                .reindex(clusters)
                .reset_index()
            )
            offsets = cluster_positions - 0.4 + width / 2.0 + group_index * width
            ax.bar(
                offsets,
                subset["percentage"].fillna(0.0).to_numpy(dtype=float),
                width=width,
                color=[cluster_color(cluster_id) for cluster_id in subset["cluster"].to_numpy()],
                hatch=hatches[group_index % len(hatches)],
                edgecolor="black",
                label=str(group_value),
            )
        ax.legend(title="Group")

    ax.set_xticks(cluster_positions)
    ax.set_xticklabels([str(cluster) for cluster in clusters])
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percent of spines")
    fig.tight_layout()
    return fig


def build_grouped_type_histogram(distribution: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    group_values = list(pd.unique(distribution["group"]))
    cluster_ids = sorted(pd.unique(distribution["cluster"]))
    x_positions = np.arange(len(group_values), dtype=float)
    width = 0.8 / max(len(cluster_ids), 1)

    for cluster_index, cluster_id in enumerate(cluster_ids):
        subset = (
            distribution[distribution["cluster"] == cluster_id]
            .set_index("group")
            .reindex(group_values)
        )
        heights = subset["percentage"].fillna(0.0).to_numpy(dtype=float)
        offsets = x_positions - 0.4 + width / 2.0 + cluster_index * width
        ax.bar(
            offsets,
            heights,
            width=width,
            color=cluster_color(int(cluster_id)),
            edgecolor="black",
            label=f"Cluster {int(cluster_id)}",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(group) for group in group_values])
    ax.set_xlabel("Group")
    ax.set_ylabel("Percent of spines")
    ax.set_title(title)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def p_value_to_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return "-"
    if p_value < 0.0001:
        return "***"
    if p_value < 0.001:
        return "**"
    if p_value < 0.05:
        return "*"
    return "-"


def cramers_v(contingency: pd.DataFrame, chi2_value: float) -> float:
    observations = float(contingency.to_numpy().sum())
    degrees = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    if observations <= 0 or degrees <= 0:
        return float("nan")
    return float(math.sqrt(chi2_value / (observations * degrees)))


def chi_square_summary(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    group_columns: Sequence[str] = ("compartment", "age_label", "group", "type"),
    verbose: bool = True,
) -> pd.DataFrame:
    merged = ensure_metadata_columns(metadata).join(assignments[["assigned_cluster"]])
    rows: list[dict[str, object]] = []
    log_progress("[6/8] Running chi-square tests for grouped distributions", verbose)
    for group_column in group_columns:
        grouped = merged.dropna(subset=[group_column])
        if grouped[group_column].nunique() < 2:
            log_progress(
                f"  - Skipping '{group_column}': fewer than two groups",
                verbose,
            )
            continue
        log_progress(f"  - Comparing distributions by '{group_column}'", verbose)
        contingency = pd.crosstab(grouped[group_column], grouped["assigned_cluster"])
        if contingency.shape[1] < 2:
            log_progress(
                f"  - Skipping '{group_column}': fewer than two represented clusters",
                verbose,
            )
            continue
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        rows.append(
            {
                "comparison": group_column,
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": cramers_v(contingency, float(chi2)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["comparison", "chi2", "dof", "p_value", "cramers_v"],
    )


def per_cluster_independence_tests(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    group_columns: Sequence[str] = ("compartment", "age_label", "group", "type"),
    verbose: bool = True,
) -> pd.DataFrame:
    merged = ensure_metadata_columns(metadata).join(assignments[["assigned_cluster"]])
    rows: list[dict[str, object]] = []
    cluster_ids = sorted(merged["assigned_cluster"].unique())
    log_progress(
        f"[6/8] Running per-cluster Pearson chi-square tests for {len(cluster_ids)} clusters",
        verbose,
    )

    for index, cluster_id in enumerate(cluster_ids, start=1):
        log_progress(f"  - Cluster {cluster_id} ({index}/{len(cluster_ids)})", verbose)
        in_cluster = (merged["assigned_cluster"] == cluster_id).astype(int)
        for group_column in group_columns:
            grouped = merged.dropna(subset=[group_column])
            if grouped[group_column].nunique() < 2:
                continue
            contingency = pd.crosstab(
                grouped[group_column],
                in_cluster.loc[grouped.index],
            )
            if contingency.shape[1] < 2:
                continue
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "comparison": group_column,
                    "chi2": float(chi2),
                    "dof": int(dof),
                    "p_value": float(p_value),
                    "cramers_v": cramers_v(contingency, float(chi2)),
                    "significance": p_value_to_stars(float(p_value)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "cluster",
            "comparison",
            "chi2",
            "dof",
            "p_value",
            "cramers_v",
            "significance",
        ],
    )


def covariance_matrix_for_component(model: GaussianMixture, component_id: int) -> np.ndarray:
    n_features = model.means_.shape[1]
    covariance_type = model.covariance_type
    if covariance_type == "full":
        return model.covariances_[component_id]
    if covariance_type == "tied":
        return model.covariances_
    if covariance_type == "diag":
        return np.diag(model.covariances_[component_id])
    if covariance_type == "spherical":
        return np.eye(n_features) * model.covariances_[component_id]
    raise ValueError(f"Unsupported covariance type: {covariance_type}")
    

def log_gaussian_density(
    x: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Лог-плотность многомерного нормального распределения N(mean, covariance)
    для массива точек x формы (n_samples, n_features).
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    n_features = mean.shape[0]
    delta = x - mean

    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    inv_cov = np.linalg.pinv(covariance)
    mahal = np.sum((delta @ inv_cov) * delta, axis=1)

    return -0.5 * (
        n_features * np.log(2.0 * np.pi)
        + logdet
        + mahal
    )




def bhattacharyya_distance(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray, cov_b: np.ndarray) -> float:
    cov_mean = 0.5 * (cov_a + cov_b)
    delta = mean_b - mean_a
    inv_cov_mean = np.linalg.pinv(cov_mean)
    term1 = 0.125 * float(delta.T @ inv_cov_mean @ delta)
    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    sign_m, logdet_m = np.linalg.slogdet(cov_mean)
    if sign_a <= 0 or sign_b <= 0 or sign_m <= 0:
        return float(term1)
    term2 = 0.5 * (logdet_m - 0.5 * (logdet_a + logdet_b))
    return float(term1 + term2)


def compute_bhattacharyya_matrix(model: GaussianMixture) -> pd.DataFrame:
    n_clusters = model.n_components
    matrix = np.zeros((n_clusters, n_clusters), dtype=float)
    for i in range(n_clusters):
        mean_i = model.means_[i]
        cov_i = covariance_matrix_for_component(model, i)
        for j in range(i + 1, n_clusters):
            mean_j = model.means_[j]
            cov_j = covariance_matrix_for_component(model, j)
            distance = bhattacharyya_distance(mean_i, cov_i, mean_j, cov_j)
            matrix[i, j] = distance
            matrix[j, i] = distance
    return pd.DataFrame(matrix, index=range(n_clusters), columns=range(n_clusters))


def compute_cluster_covariance_log_volume(model: GaussianMixture) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for cluster_id in range(model.n_components):
        covariance = covariance_matrix_for_component(model, cluster_id)
        sign, logdet = np.linalg.slogdet(covariance)
        value = float("nan") if sign <= 0 else abs(float(logdet / math.log(10.0)))
        rows.append(
            {
                "cluster": int(cluster_id),
                "abs_log10_det_covariance": value,
            }
        )
    return pd.DataFrame(rows)


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    n_samples = distance_matrix.shape[0]
    squared = distance_matrix ** 2
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.maximum(eigenvalues[:n_components], 0.0)
    return eigenvectors[:, :n_components] * np.sqrt(positive)


def normalize_membership_probabilities(probabilities: np.ndarray) -> np.ndarray:
    normalized = np.asarray(probabilities, dtype=float)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, None)
    row_sums = normalized.sum(axis=1, keepdims=True)
    return np.divide(
        normalized,
        row_sums,
        out=np.full_like(normalized, 1.0 / normalized.shape[1]),
        where=row_sums > 0,
    )


def blended_cluster_colors(probabilities: np.ndarray, palette: Sequence[str]) -> list[str]:
    probabilities = normalize_membership_probabilities(probabilities)
    base_colors = np.array([to_rgb(color) for color in palette])
    mixed = np.clip(probabilities @ base_colors, 0.0, 1.0)
    return [to_hex(color) for color in mixed]


def build_membership_mds_plot(
    model: GaussianMixture,
    probabilities: np.ndarray,
    verbose: bool = True,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    log_progress("[7/8] Building Bhattacharyya distance matrix and MDS projection", verbose)
    distance_frame = compute_bhattacharyya_matrix(model)
    cluster_positions = classical_mds(distance_frame.to_numpy(), n_components=2)
    probabilities = normalize_membership_probabilities(probabilities)
    object_positions = probabilities @ cluster_positions
    palette = [cluster_color(index) for index in range(model.n_components)]
    colors = blended_cluster_colors(probabilities, palette)

    cluster_frame = pd.DataFrame(
        cluster_positions,
        columns=["MDS_1", "MDS_2"],
        index=[f"cluster_{index}" for index in range(model.n_components)],
    )
    object_frame = pd.DataFrame(object_positions, columns=["MDS_1", "MDS_2"])
    object_frame["color"] = colors

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(object_frame["MDS_1"], object_frame["MDS_2"], c=object_frame["color"], s=34, alpha=0.78)
    ax.scatter(cluster_frame["MDS_1"], cluster_frame["MDS_2"], c=palette, s=380, marker="X", edgecolor="black")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="X",
            linestyle="",
            markersize=12,
            markerfacecolor=palette[index],
            markeredgecolor="black",
            label=f"Cluster {index}",
        )
        for index in range(model.n_components)
    ]
    ax.set_title("MDS projection of cluster memberships")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, distance_frame, cluster_frame


def build_covariance_volume_strip(covariance_volume_frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    left = 0.0
    handles = []

    for row in covariance_volume_frame.sort_values("cluster").itertuples(index=False):
        value = 0.0 if pd.isna(row.abs_log10_det_covariance) else float(row.abs_log10_det_covariance)
        color = cluster_color(int(row.cluster))
        ax.barh(
            y=[0],
            width=[value],
            left=[left],
            height=0.7,
            color=color,
            edgecolor="black",
        )
        if value > 0:
            ax.text(
                left + value / 2.0,
                0,
                f"C{int(row.cluster)}\n{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                lw=8,
                label=f"Cluster {int(row.cluster)}",
            )
        )
        left += value

    ax.set_title(r"Cluster covariance heterogeneity strip: $|log_{10} det(\Sigma_i)|$")
    ax.set_xlabel(r"$|log_{10} det(\Sigma_i)|$")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def estimate_overlap_matrix(
    model: GaussianMixture,
    n_samples_per_cluster: int = 5000,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_clusters = model.n_components
    overlap = np.zeros((n_clusters, n_clusters), dtype=float)
    log_progress(
        f"[7/8] Estimating overlap matrix with {n_samples_per_cluster} synthetic samples per cluster",
        verbose,
    )

    for cluster_id in range(n_clusters):
        log_progress(f"  - Sampling cluster {cluster_id + 1}/{n_clusters}", verbose)
        cov = covariance_matrix_for_component(model, cluster_id)
        samples = rng.multivariate_normal(
            mean=model.means_[cluster_id],
            cov=cov,
            size=n_samples_per_cluster,
        )
        posterior = model.predict_proba(samples)
        assigned = posterior.argmax(axis=1)
        for predicted_cluster in range(n_clusters):
            overlap[predicted_cluster, cluster_id] = np.mean(assigned == predicted_cluster)

    np.fill_diagonal(overlap, 1.0 - (overlap.sum(axis=0) - np.diag(overlap)))
    frame = pd.DataFrame(
        overlap,
        index=[f"pred_{idx}" for idx in range(n_clusters)],
        columns=[f"true_{idx}" for idx in range(n_clusters)],
    )
    return frame

def estimate_overlap_matrix_map_style(
    model: GaussianMixture,
    n_samples_per_cluster: int = 5000,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)
    n_clusters = model.n_components
    overlap = np.zeros((n_clusters, n_clusters), dtype=float)

    weights = np.asarray(model.weights_, dtype=float)
    log_weights = np.log(weights)

    means = np.asarray(model.means_, dtype=float)
    covariances = [
        covariance_matrix_for_component(model, component_id)
        for component_id in range(n_clusters)
    ]

    log_progress(
        f"[7/8] Estimating MAP-style overlap matrix with {n_samples_per_cluster} synthetic samples per cluster",
        verbose,
    )

    for true_cluster in range(n_clusters):
        log_progress(f"  - Sampling component {true_cluster + 1}/{n_clusters}", verbose)

        samples = rng.multivariate_normal(
            mean=means[true_cluster],
            cov=covariances[true_cluster],
            size=n_samples_per_cluster,
        )

        component_scores = np.column_stack([
            log_weights[pred_cluster] + log_gaussian_density(
                samples,
                means[pred_cluster],
                covariances[pred_cluster],
            )
            for pred_cluster in range(n_clusters)
        ])

        predicted_clusters = np.argmax(component_scores, axis=1)

        for pred_cluster in range(n_clusters):
            overlap[true_cluster, pred_cluster] = np.mean(
                predicted_clusters == pred_cluster
            )

    frame = pd.DataFrame(
        overlap,
        index=[f"true_{i}" for i in range(n_clusters)],
        columns=[f"pred_{j}" for j in range(n_clusters)],
    )
    return frame


# def compute_overlapping_reference_style(
#     model: GaussianMixture,
#     n_samples_per_cluster: int = 5000,
#     random_state: int = 0,
#     eps: float = 1e-9,
#     lim: int = 10**6,
#     verbose: bool = True,
# ) -> pd.DataFrame:

#     _ = (eps, lim)
#     overlap_table = estimate_overlap_matrix_map_style(
#         model=model,
#         n_samples_per_cluster=n_samples_per_cluster,
#         random_state=random_state,
#         verbose=verbose,
#     )
#     cluster_labels = [f"Cluster {index + 1}" for index in range(model.n_components)]
#     overlap_table.index = cluster_labels
#     overlap_table.columns = cluster_labels
#     return overlap_table

def compute_overlapping_reference_style(
    model: GaussianMixture,
    eps: float = 1e-9,
    lim: int = 10**6,
    verbose: bool = True,
) -> pd.DataFrame:
    rscript_path = r"C:\Users\Student\Липс Екатерина\R-4.5.3\bin\x64\Rscript.exe"

    weights = np.asarray(model.weights_, dtype=float)
    means = np.asarray(model.means_, dtype=float)

    covariances = np.stack(
        [covariance_matrix_for_component(model, k) for k in range(model.n_components)],
        axis=2,  # shape: (n_features, n_features, n_components)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        pi_path = tmpdir / "Pi.csv"
        mu_path = tmpdir / "Mu.csv"
        s_path = tmpdir / "S.npy"
        omega_path = tmpdir / "OmegaMap.csv"

        np.savetxt(pi_path, weights, delimiter=",")
        np.savetxt(mu_path, means, delimiter=",")
        np.save(s_path, covariances)

        r_script = f"""
        suppressPackageStartupMessages(library(MixSim))
        suppressPackageStartupMessages(library(reticulate))

        Pi <- as.numeric(read.csv("{pi_path.as_posix()}", header=FALSE)[,1])
        Mu <- as.matrix(read.csv("{mu_path.as_posix()}", header=FALSE))

        py <- import("numpy", convert=TRUE)
        S <- py$load("{s_path.as_posix()}")

        out <- overlap(Pi=Pi, Mu=Mu, S=S, eps={eps}, lim={lim})$OmegaMap
        write.csv(out, file="{omega_path.as_posix()}", row.names=FALSE)
        """

        try:
            result = subprocess.run(
                [rscript_path, "-e", r_script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Rscript failed.\n"
                f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}"
            ) from e


        if verbose and result.stdout.strip():
            print(result.stdout)

        if not omega_path.exists():
            raise RuntimeError(
                "R script finished, but OmegaMap.csv was not created.\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

        overlap_table = pd.read_csv(omega_path)

    cluster_labels = [f"Cluster {index + 1}" for index in range(model.n_components)]
    overlap_table.index = cluster_labels
    overlap_table.columns = cluster_labels

    return overlap_table


def threshold_membership_table(
    assignments: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cluster_counts = assignments["assigned_cluster"].value_counts().sort_index()

    for threshold in thresholds:
        row: dict[str, object] = {"threshold": threshold}
        for cluster_id, total in cluster_counts.items():
            count = int(
                (
                    (assignments["assigned_cluster"] == cluster_id)
                    & (assignments["p_star"] >= threshold)
                ).sum()
            )
            row[f"cluster_{cluster_id} (n={int(total)})"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def save_interactive_figure(fig: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "write_html"):
        fig.write_html(output_path)


def select_representative_spines(
    assignments: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cluster_id in sorted(assignments["assigned_cluster"].unique()):
        subset = (
            assignments[assignments["assigned_cluster"] == cluster_id]
            .sort_values(["p_star"], ascending=[False])
            .head(top_n)
            .copy()
        )
        subset["representative_rank"] = np.arange(1, len(subset) + 1)
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0)


def build_representative_mesh_figure(
    base_dir: Path,
    cluster_id: int,
    representative_table: pd.DataFrame,
) -> object:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    try:
        if pio.renderers.default in ("", None):
            pio.renderers.default = "notebook_connected"
    except Exception:
        pass

    subset = representative_table[representative_table["assigned_cluster"] == cluster_id].copy()
    if subset.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=len(subset),
        specs=[[{"type": "scene"} for _ in range(len(subset))]],
        subplot_titles=[
            f"{spine_key.split('/', 1)[-1]}<br>p*={p_star:.3f}"
            for spine_key, p_star in zip(subset.index, subset["p_star"])
        ],
    )

    mesh_color = cluster_color(cluster_id)
    for column_index, spine_key in enumerate(subset.index, start=1):
        mesh_path = resolve_spine_mesh_path(base_dir, spine_key)
        vertices, faces = load_off_mesh(mesh_path)
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=mesh_color,
                opacity=0.8,
                name=str(spine_key),
                showscale=False,
            ),
            row=1,
            col=column_index,
        )
        scene_name = "scene" if column_index == 1 else f"scene{column_index}"
        fig.layout[scene_name].update(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        )

    fig.update_layout(
        title=f"Representative spines for cluster {cluster_id}",
        showlegend=False,
        height=360,
        width=max(420 * len(subset), 700),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def build_representative_mesh_figures(
    base_dir: Path,
    assignments: pd.DataFrame,
    top_n: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[int, object]]:
    log_progress(f"[8/8] Selecting top-{top_n} representative spines per cluster", verbose)
    representative_table = select_representative_spines(assignments, top_n=top_n)
    figures: dict[int, object] = {}
    for cluster_id in sorted(representative_table["assigned_cluster"].unique()):
        log_progress(f"  - Building representative mesh figure for cluster {cluster_id}", verbose)
        figures[int(cluster_id)] = build_representative_mesh_figure(
            base_dir=base_dir,
            cluster_id=int(cluster_id),
            representative_table=representative_table,
        )
    return representative_table, figures


def build_representative_mesh_grid_figure(
    base_dir: Path,
    assignments: pd.DataFrame,
    top_n_per_cluster: int = 4,
) -> object:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    try:
        if pio.renderers.default in ("", None):
            pio.renderers.default = "notebook_connected"
    except Exception:
        pass

    representative_table = select_representative_spines(assignments, top_n=top_n_per_cluster)
    cluster_ids = sorted(representative_table["assigned_cluster"].unique())
    if not cluster_ids:
        return None

    rows = top_n_per_cluster
    cols = len(cluster_ids)
    cluster_subsets = {
        int(cluster_id): representative_table[representative_table["assigned_cluster"] == cluster_id]
        for cluster_id in cluster_ids
    }
    subplot_titles: list[str] = []
    for row_index in range(rows):
        for cluster_id in cluster_ids:
            subset = cluster_subsets[int(cluster_id)]
            if row_index < len(subset):
                spine_key = subset.index[row_index]
                subplot_titles.append(f"Cluster {cluster_id}<br>{spine_key.split('/', 1)[-1]}")
            else:
                subplot_titles.append(f"Cluster {cluster_id}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    legend_cluster_ids: set[int] = set()
    for col_index, cluster_id in enumerate(cluster_ids, start=1):
        subset = cluster_subsets[int(cluster_id)]
        mesh_color = cluster_color(int(cluster_id))
        for row_index in range(rows):
            if row_index >= len(subset):
                continue
            spine_key = subset.index[row_index]
            p_star = float(subset.iloc[row_index]["p_star"])
            mesh_path = resolve_spine_mesh_path(base_dir, spine_key)
            vertices, faces = load_off_mesh(mesh_path)
            show_legend = int(cluster_id) not in legend_cluster_ids
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=mesh_color,
                    opacity=0.82,
                    name=f"Cluster {int(cluster_id)}",
                    legendgroup=f"cluster_{int(cluster_id)}",
                    showlegend=show_legend,
                    hovertemplate=(
                        f"Cluster {int(cluster_id)}<br>"
                        f"{spine_key}<br>"
                        f"p*={p_star:.3f}<extra></extra>"
                    ),
                ),
                row=row_index + 1,
                col=col_index,
            )
            legend_cluster_ids.add(int(cluster_id))
            scene_name = "scene" if (row_index == 0 and col_index == 1) else f"scene{(row_index) * cols + col_index}"
            fig.layout[scene_name].update(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                aspectmode="data",
            )

    fig.update_layout(
        title="Representative spines grid (top 4 per cluster)",
        height=max(230 * rows, 620),
        width=max(210 * cols, 760),
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(title="Clusters", orientation="v", x=1.02, y=1),
    )
    return fig


def run_full_experiment(
    base_dir: Path,
    dataset_names: Sequence[str] = DEFAULT_DATASETS,
    output_dir: Path | None = None,
    cluster_range: Iterable[int] = range(2, 11),
    covariance_types: Sequence[str] = DEFAULT_COVARIANCE_TYPES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    random_state: int = 0,
    include_age_analysis: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    base_dir = Path(base_dir).resolve()
    output_dir = (base_dir / "gmm_human_spine_outputs") if output_dir is None else Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not include_age_analysis:
        stale_age_outputs = (
            "cluster_distribution_by_age.csv",
            "cluster_distribution_by_group.csv",
            "hist_age.png",
            "hist_group.png",
            "hist_age_stacked.png",
            "hist_group_stacked.png",
        )
        for filename in stale_age_outputs:
            (output_dir / filename).unlink(missing_ok=True)
    log_progress("[0/8] Starting full GMM clustering experiment", verbose)

    features, metadata, diagnostics = load_all_datasets(
        base_dir=base_dir,
        dataset_names=dataset_names,
        verbose=verbose,
    )
    metadata = ensure_metadata_columns(metadata)
    log_progress("[2/8] Standardizing features", verbose)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features.to_numpy())

    bic_frame, best_model = fit_gmm_sweep(
        standardized_features=standardized,
        cluster_range=cluster_range,
        covariance_types=covariance_types,
        random_state=random_state,
        verbose=verbose,
    )
    log_progress("[3/8] Computing posterior probabilities and hard assignments", verbose)
    probabilities = best_model.predict_proba(standardized)
    assignments = responsibilities_to_frame(probabilities, features.index)
    full_table = metadata.join(assignments).join(features[["Volume"]])

    umap_embedding = build_umap_embedding(standardized, random_state=random_state, verbose=verbose)
    log_progress("[5/8] Building plots and cluster distribution tables", verbose)
    bic_fig = build_bic_plot(bic_frame)
    umap_fig = build_umap_plot(umap_embedding, assignments, metadata)
    umap_type_fig = (
        build_umap_type_plot(umap_embedding, metadata)
        if metadata["type"].nunique(dropna=True) > 1
        else None
    )
    umap_type_interactive_fig = (
        build_interactive_umap_type_plot(umap_embedding, metadata, assignments)
        if metadata["type"].nunique(dropna=True) > 1
        else None
    )

    distributions = {
        "overall": distribution_table(assignments, metadata),
        "compartment": distribution_table(assignments, metadata, "compartment"),
    }
    if include_age_analysis and metadata["age_label"].notna().any():
        distributions["age"] = distribution_table(assignments, metadata, "age_label")
    if include_age_analysis and metadata["group"].notna().any():
        distributions["group"] = distribution_table(assignments, metadata, "group")
    if metadata["type"].nunique(dropna=True) > 1:
        distributions["type"] = distribution_table(assignments, metadata, "type")

    distribution_titles = {
        "overall": "Cluster distribution: all spines",
        "compartment": "Cluster distribution by dendritic compartment",
        "age": "Cluster distribution by age",
        "group": "Cluster distribution by compartment + age",
        "type": "Cluster distribution by type",
    }
    grouped_distribution_titles = {
        "compartment": "Cluster distribution grouped by dendritic compartment",
        "age": "Cluster distribution grouped by age",
        "group": "Cluster distribution grouped by compartment + age",
        "type": "Cluster distribution grouped by type",
    }
    histogram_figures = {
        key: build_histogram(distribution, distribution_titles[key])
        for key, distribution in distributions.items()
    }
    histogram_figures.update(
        {
            f"{key}_stacked": build_grouped_type_histogram(
                distribution,
                grouped_distribution_titles[key],
            )
            for key, distribution in distributions.items()
            if key != "overall"
        }
    )

    comparison_columns = ["compartment"]
    if include_age_analysis:
        comparison_columns.extend(["age_label", "group"])
    if metadata["type"].nunique(dropna=True) > 1:
        comparison_columns.append("type")
    comparison_tests = chi_square_summary(
        assignments,
        metadata,
        group_columns=comparison_columns,
        verbose=verbose,
    )
    per_cluster_tests = per_cluster_independence_tests(
        assignments,
        metadata,
        group_columns=comparison_columns,
        verbose=verbose,
    )
    mds_fig, bhattacharyya_frame, cluster_mds_positions = build_membership_mds_plot(
        best_model,
        probabilities,
        verbose=verbose,
    )
    covariance_volume_frame = compute_cluster_covariance_log_volume(best_model)
    covariance_volume_fig = build_covariance_volume_strip(covariance_volume_frame)

    overlap_frame = estimate_overlap_matrix(
        best_model,
        random_state=random_state,
        verbose=verbose,
    )

    overlap_map_frame = estimate_overlap_matrix_map_style(
        best_model,
        random_state=random_state,
        verbose=verbose,
    )
#     overlap_reference_frame = compute_overlapping_reference_style(
#         best_model,
#         random_state=random_state,
#         verbose=verbose,
#     )
    overlap_reference_frame = compute_overlapping_reference_style(
        best_model,
        verbose=verbose,
    )

    threshold_table = threshold_membership_table(assignments, thresholds=thresholds)
    representative_table, representative_mesh_figures = build_representative_mesh_figures(
        base_dir=base_dir,
        assignments=assignments,
        top_n=5,
        verbose=verbose,
    )
    representative_mesh_grid_figure = build_representative_mesh_grid_figure(
        base_dir=base_dir,
        assignments=assignments,
        top_n_per_cluster=4,
    )
    features_preview = features.head(5)

    log_progress(f"[8/8] Saving tables and figures to {output_dir}", verbose)
    diagnostics.to_csv(output_dir / "dataset_diagnostics.csv", index=False)
    bic_frame.to_csv(output_dir / "bic_sweep.csv", index=False)
    full_table.to_csv(output_dir / "cluster_assignments.csv")
    pd.DataFrame(
        umap_embedding,
        index=features.index,
        columns=["UMAP_1", "UMAP_2", "UMAP_3"],
    ).to_csv(output_dir / "umap_embedding.csv")
    distribution_output_names = {
        "overall": "cluster_distribution_all.csv",
        "compartment": "cluster_distribution_by_compartment.csv",
        "age": "cluster_distribution_by_age.csv",
        "group": "cluster_distribution_by_group.csv",
        "type": "cluster_distribution_by_type.csv",
    }
    for key, distribution in distributions.items():
        distribution.to_csv(output_dir / distribution_output_names[key], index=False)
    comparison_tests.to_csv(output_dir / "distribution_comparison_pvalues.csv", index=False)
    per_cluster_tests.to_csv(output_dir / "per_cluster_independence_tests.csv", index=False)
    bhattacharyya_frame.to_csv(output_dir / "bhattacharyya_distances.csv")
    cluster_mds_positions.to_csv(output_dir / "cluster_mds_positions.csv")
    covariance_volume_frame.to_csv(output_dir / "cluster_covariance_log_volume.csv", index=False)
    overlap_frame.to_csv(output_dir / "overlap_matrix.csv")
    overlap_map_frame.to_csv(output_dir / "overlap_matrix_map_style.csv")
    overlap_reference_frame.to_csv(output_dir / "overlap_matrix_reference_style.csv")
    threshold_table.to_csv(output_dir / "membership_threshold_table.csv", index=False)
    representative_table.to_csv(output_dir / "representative_spines.csv")

    save_figure(bic_fig, output_dir / "bic_sweep.png")
    save_figure(umap_fig, output_dir / "umap_3d_clusters.png")
    if umap_type_fig is not None:
        save_figure(umap_type_fig, output_dir / "umap_3d_by_type.png")
    if umap_type_interactive_fig is not None:
        save_interactive_figure(
            umap_type_interactive_fig,
            output_dir / "umap_3d_by_type_interactive.html",
        )
    histogram_output_names = {
        "overall": "hist_all.png",
        "compartment": "hist_compartment.png",
        "age": "hist_age.png",
        "group": "hist_group.png",
        "type": "hist_type.png",
        "compartment_stacked": "hist_compartment_stacked.png",
        "age_stacked": "hist_age_stacked.png",
        "group_stacked": "hist_group_stacked.png",
        "type_stacked": "hist_type_stacked.png",
    }
    for key, figure in histogram_figures.items():
        save_figure(figure, output_dir / histogram_output_names[key])
    save_figure(mds_fig, output_dir / "membership_mds.png")
    save_figure(covariance_volume_fig, output_dir / "cluster_covariance_log_volume.png")

    best_row = bic_frame.loc[bic_frame["bic"].idxmax()]
    best_model_summary = pd.DataFrame(
        [
            {
                "n_clusters": int(best_model.n_components),
                "covariance_type": best_model.covariance_type,
                "bic": float(best_row["bic"]),
                "sklearn_bic": float(best_row["sklearn_bic"]),
            }
        ]
    )
    best_model_summary.to_csv(output_dir / "best_model.csv", index=False)
    log_progress("[done] Experiment finished", verbose)

    figures = {
        "bic": bic_fig,
        "umap": umap_fig,
        **histogram_figures,
        "mds": mds_fig,
        "covariance_volume": covariance_volume_fig,
    }
    if umap_type_fig is not None:
        figures["umap_by_type"] = umap_type_fig
    if umap_type_interactive_fig is not None:
        figures["umap_by_type_interactive"] = umap_type_interactive_fig

    return {
        "features": features,
        "metadata": metadata,
        "diagnostics": diagnostics,
        "bic_frame": bic_frame,
        "best_model": best_model,
        "best_model_summary": best_model_summary,
        "assignments": assignments,
        "full_table": full_table,
        "umap_embedding": umap_embedding,
        "figures": figures,
        "distributions": distributions,
        "comparison_tests": comparison_tests,
        "per_cluster_tests": per_cluster_tests,
        "bhattacharyya_distances": bhattacharyya_frame,
        "cluster_covariance_log_volume": covariance_volume_frame,
        "overlap_matrix": overlap_frame,
        "overlap_matrix_map_style": overlap_map_frame,
        "overlap_matrix_reference_style": overlap_reference_frame,
        "threshold_table": threshold_table,
        "features_preview": features_preview,
        "representative_spines": representative_table,
        "representative_mesh_figures": representative_mesh_figures,
        "representative_mesh_grid_figure": representative_mesh_grid_figure,
        "output_dir": output_dir,
    }
