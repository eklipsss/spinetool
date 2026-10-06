"""The 12 spine morphometrics (s-module-preprocessing.md §3.8), shared by preprocessing
stage 8 (real spines) and S-module evaluation (generated spines).

The spec requires generated spines to be judged with exactly the same metrics as real
ones, so both paths call :func:`compute_spine_morphometrics` - preprocessing passes the
attachment region found while sealing (stage 1), evaluation passes whatever an
attachment finder returns for a generated mesh (or nothing).

Attachment dependence (``src/spine_analysis/shape_metric``):

- ``JUNCTION_METRICS`` (8) need the attachment region: OpenAngle, CVD, AverageDistance,
  Length, LengthVolumeRatio, LengthAreaRatio measure from the attachment *centre*
  (``JunctionSpineMetric`` subclasses, via ``register_attachment_center``); JunctionArea
  and Area need its *area* (loop area / cap area). Without an attachment region they
  are ``None``.
- ``ATTACHMENT_FREE_METRICS`` (4): Volume, ConvexHullVolume, ConvexHullRatio,
  OldChordDistribution - always computable.

``OldChordDistribution`` is a 100-bin density of normalised chord length on ``[0, 1]``
(``np.histogram(..., range=(0, 1), density=True)``) and is NOT reproducible between runs:
the reference implementation draws chords with an unseeded ``random.Random()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

JUNCTION_METRICS = (
    "OpenAngle",
    "CVD",
    "AverageDistance",
    "LengthVolumeRatio",
    "LengthAreaRatio",
    "Length",
    "JunctionArea",
    "Area",
)
ATTACHMENT_FREE_METRICS = ("Volume", "ConvexHullVolume", "ConvexHullRatio", "OldChordDistribution")
ALL_METRICS = JUNCTION_METRICS + ATTACHMENT_FREE_METRICS
SCALAR_METRICS = tuple(m for m in ALL_METRICS if m != "OldChordDistribution")
HISTOGRAM_METRICS = ("OldChordDistribution",)
CHORD_BINS = 100  # OldChordDistributionSpineMetric default num_of_bins, range (0, 1)


@dataclass(frozen=True)
class AttachmentRegion:
    """Where the spine meets the dendrite, in the mesh's own coordinates.

    ``center``: attachment centre (the local-frame origin for preprocessed spines);
    ``cap_area``: area of the faces closing the attachment opening (subtracted for "Area");
    ``loop_area``: area of the attachment opening itself ("JunctionArea").
    """

    center: np.ndarray
    cap_area: Optional[float]
    loop_area: Optional[float]
    method: Optional[str] = None  # how it was found (stage-1 sealing, or a generated-mesh finder)


def _metric_value(metric: Any) -> Any:
    value = metric.value
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _polyhedron(mesh: Any, center: Optional[np.ndarray]):
    from src.spine_analysis.mesh.utils import v_f_to_mesh_isolated
    from src.spine_analysis.shape_metric.utils import register_attachment_center

    poly = v_f_to_mesh_isolated(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int))
    if center is not None:
        # the junction helpers look the centre up per Polyhedron object (keyed by id())
        register_attachment_center(poly, np.asarray(center, dtype=float))
    return poly


def build_polyhedron(mesh: Any, attachment: Optional[AttachmentRegion]) -> Any:
    """The CGAL Polyhedron the metrics run on (attachment centre registered). Building it
    costs a subprocess (``v_f_to_mesh_isolated``), so callers that compute several metric
    groups for one mesh should build it once and pass it as ``poly=``."""
    return _polyhedron(mesh, None if attachment is None else attachment.center)


def compute_scalar_metrics(mesh: Any, attachment: Optional[AttachmentRegion], *, poly: Any = None) -> Dict[str, Any]:
    """Every metric except OldChordDistribution; junction metrics are ``None`` without ``attachment``.

    ``poly``: a :func:`build_polyhedron` result for this mesh/attachment (built here if None).
    """
    from src.spine_analysis.shape_metric.float_metric import (
        ConvexHullRatioSpineMetric,
        ConvexHullVolumeSpineMetric,
        VolumeSpineMetric,
    )
    from src.spine_analysis.shape_metric.junction_metric import (
        AverageDistanceSpineMetric,
        CVDSpineMetric,
        LengthAreaRatioSpineMetric,
        LengthSpineMetric,
        LengthVolumeRatioSpineMetric,
        OpenAngleSpineMetric,
    )

    if poly is None:
        poly = build_polyhedron(mesh, attachment)
    values: Dict[str, Any] = {name: None for name in JUNCTION_METRICS}
    if attachment is not None:
        # These only use the registered attachment CENTRE (JunctionSpineMetric._calculate:
        # _junction_center = the registered point as-is), never the library's own cruder
        # "nearest face + 1-ring" attachment-patch guess - see s-module-preprocessing.md §3.8.1.
        values.update(
            {
                "OpenAngle": _metric_value(OpenAngleSpineMetric(poly)),
                "CVD": _metric_value(CVDSpineMetric(poly)),
                "AverageDistance": _metric_value(AverageDistanceSpineMetric(poly)),
                "LengthVolumeRatio": _metric_value(LengthVolumeRatioSpineMetric(poly)),
                "LengthAreaRatio": _metric_value(LengthAreaRatioSpineMetric(poly)),
                "Length": _metric_value(LengthSpineMetric(poly)),
                "JunctionArea": _safe_float(attachment.loop_area),
                # Area = total surface minus the attachment cap, from our own cap area -
                # NOT AreaSpineMetric (which subtracts its own re-guessed junction patch,
                # 40-130% off on real spines; §3.8.1).
                "Area": _safe_float(mesh.area - (attachment.cap_area or 0.0)),
            }
        )
    values.update(
        {
            "Volume": _metric_value(VolumeSpineMetric(poly)),
            "ConvexHullVolume": _metric_value(ConvexHullVolumeSpineMetric(poly)),
            "ConvexHullRatio": _metric_value(ConvexHullRatioSpineMetric(poly)),
        }
    )
    return values


def compute_chord_distribution(mesh: Any, attachment: Optional[AttachmentRegion] = None, *, poly: Any = None) -> Any:
    """OldChordDistribution (slowest metric: 3000 random chords; not seeded upstream)."""
    from src.spine_analysis.shape_metric.histogram_metric import OldChordDistributionSpineMetric

    if poly is None:
        poly = build_polyhedron(mesh, attachment)
    return _metric_value(OldChordDistributionSpineMetric(poly))


def compute_spine_morphometrics(mesh: Any, attachment: Optional[AttachmentRegion]) -> Dict[str, Any]:
    poly = build_polyhedron(mesh, attachment)
    values = compute_scalar_metrics(mesh, attachment, poly=poly)
    values["OldChordDistribution"] = compute_chord_distribution(mesh, attachment, poly=poly)
    return values
