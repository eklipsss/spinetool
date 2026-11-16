"""Forward model of confocal imaging: what a fluorescence confocal microscope would have shown of an EM-reconstructed surface.

Purpose (docs/neuron-model/confocal-imaging-model.md): LabID dendrites are confocal images (voxel 25 x 25 x 100 nm, optical
resolution ~200 nm laterally and ~600 nm axially), Minnie65 / H01 are electron-microscopy reconstructions (a few nm). A spine
of a LabID mesh is the iso-surface of a blurred fluorescence image, not the membrane. To compare the two domains, the EM
surface is passed through the same physics:

1. fill: the inside of the closed mesh is the fluorophore (occupancy 1, outside 0), computed on a fine grid by ray parity
   along z (``occupancy_grid``) and averaged into ``work_nm`` voxels (partial-volume fractions);
2. blur: convolution with the point-spread function, a 3-D Gaussian with ``sigma_xy`` / ``sigma_z`` (``gaussian_psf_sigmas``
   gives them from NA, wavelengths and refractive index, Zhang, Zerubia & Olivo-Marin 2007, Appl. Opt. 46:1819);
3. sample: the blurred intensity read at the centres of the confocal voxels (``voxel_nm``), with a random sub-voxel phase
   (the grid of the microscope does not know where the spine is);
4. optional noise (Gaussian, relative to the full-fill intensity);
5. segment: iso-surface at ``threshold`` x the intensity of a fully filled large region (Marching Cubes), largest component.

Geometry of imaging: LabID dendrites lie in the imaging plane (field 15-25 um in xy, ~2.5 um in z), so the optical axis is
perpendicular to the dendrite and the spine points at a uniformly random angle around it. In the spine's local frame
(x = dendrite tangent, +y = into the spine, z = x cross y) the optical axis is therefore ``cos(phi) y + sin(phi) z`` with
phi uniform (``optical_axis_angle``); the mesh is rotated into the microscope frame, imaged and rotated back, so the result is
again in the local frame (+y into the spine) and the usual morphometrics / base finder apply.

Units: nm everywhere. Unknown optical parameters (NA, wavelengths, threshold of the laboratory's surface software) are
parameters: the model is run on a grid of them (sensitivity) until the laboratory answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ConfocalParams:
    voxel_nm: Tuple[float, float, float] = (25.0, 25.0, 100.0)   # sampling of the LabID stacks (laboratory, 2026-10-28)
    sigma_xy_nm: float = 90.0                                      # PSF Gaussian sd; ~ 0.21 lambda / NA for NA 1.4, 600 nm
    sigma_z_nm: float = 270.0
    threshold: float = 0.5                                         # iso level, fraction of the full-fill intensity
    work_nm: float = 25.0                                          # isotropic grid of the blur (<< sigma)
    supersample: int = 2                                           # occupancy sub-voxels per work voxel and axis
    noise_sd: float = 0.0                                          # Gaussian noise, fraction of the full-fill intensity
    optical_axis_angle: Optional[float] = None                     # radians in the y-z plane of the spine; None = random
    random_phase: bool = True                                      # random sub-voxel offset of the sampling grid
    keep: str = "largest"                                          # components kept: largest | all
    trunk_radius_nm: Optional[float] = None                        # image together with a trunk cylinder of this radius, cut along it
    attached_max_gap_nm: float = 150.0                             # kept piece starting higher than this above the trunk = detached head

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["voxel_nm"] = list(self.voxel_nm)
        return d

    @classmethod
    def from_mapping(cls, m: Optional[Dict[str, Any]]) -> "ConfocalParams":
        m = dict(m or {})
        if "voxel_nm" in m:
            m["voxel_nm"] = tuple(float(v) for v in m["voxel_nm"])
        return cls(**{k: v for k, v in m.items() if k in cls.__dataclass_fields__})


def gaussian_psf_sigmas(na: float, wavelength_em_nm: float, refractive_index: float, *, wavelength_ex_nm: Optional[float] = None,
                        confocal: bool = True) -> Tuple[float, float]:
    """Gaussian approximation of the PSF (sigma_xy, sigma_z) in nm, Zhang et al. 2007 (paraxial, small pinhole for confocal).

    Wide field: sigma_xy = 0.21 lambda_em / NA, sigma_z = 0.66 lambda_em n / NA^2. Laser-scanning confocal: lambda_em replaced
    by lambda_ex lambda_em / sqrt(lambda_ex^2 + lambda_em^2) (product of excitation and detection PSFs). A wide-open pinhole
    behaves closer to wide field; the true effective PSF is to be checked against the laboratory's settings."""
    lam = float(wavelength_em_nm)
    if confocal:
        ex = float(wavelength_ex_nm if wavelength_ex_nm is not None else wavelength_em_nm)
        lam = ex * lam / np.hypot(ex, lam)
    return 0.21 * lam / float(na), 0.66 * lam * float(refractive_index) / float(na) ** 2


def occupancy_grid(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray, spacing: float, shape: Tuple[int, int, int]) -> np.ndarray:
    """Inside / outside of a closed mesh at the voxel centres ``origin + (i + 0.5) * spacing`` by ray parity along +z.

    For every triangle the columns (x, y) whose centre falls inside its xy projection get the z of the crossing; the parity
    of crossings below a voxel centre decides inside. Columns are shifted by an irrational fraction of a voxel so that rays
    do not hit edges / vertices exactly (double counts). Self-intersections only flip small pockets."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    nx, ny, nz = (int(s) for s in shape)
    eps = np.array([0.5 + 1.234567e-4, 0.5 + 7.654321e-5])
    toggles = np.zeros((nx, ny, nz + 1), dtype=np.int32)
    tri = v[f]                                                   # (F, 3, 3)
    lo = np.floor((tri[:, :, :2].min(axis=1) - origin[:2]) / spacing - eps).astype(np.int64) + 1
    hi = np.floor((tri[:, :, :2].max(axis=1) - origin[:2]) / spacing - eps).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi[:, 0] = np.minimum(hi[:, 0], nx - 1)
    hi[:, 1] = np.minimum(hi[:, 1], ny - 1)
    for t in np.flatnonzero((hi[:, 0] >= lo[:, 0]) & (hi[:, 1] >= lo[:, 1])):
        a, b, c = tri[t]
        ix = np.arange(lo[t, 0], hi[t, 0] + 1)
        iy = np.arange(lo[t, 1], hi[t, 1] + 1)
        gx, gy = np.meshgrid(origin[0] + (ix + eps[0]) * spacing, origin[1] + (iy + eps[1]) * spacing, indexing="ij")
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-18:
            continue                                             # triangle parallel to the ray
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        k = np.ceil((z - origin[2]) / spacing - 0.5).astype(np.int64)  # first voxel centre above the crossing
        k = np.clip(k, 0, nz)
        sel = np.nonzero(inside)
        np.add.at(toggles, (ix[sel[0]], iy[sel[1]], k[sel]), 1)
    return (np.cumsum(toggles, axis=2)[:, :, :nz] % 2).astype(bool)


def _rotation_to_microscope(phi: float) -> np.ndarray:
    """Rows = microscope axes in local coordinates: X = dendrite tangent, Z = optical axis cos(phi) y + sin(phi) z, Y = Z x X."""
    ex = np.array([1.0, 0.0, 0.0])
    ez = np.array([0.0, np.cos(phi), np.sin(phi)])
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez])


def _occupancy(v: np.ndarray, f: np.ndarray, lo: np.ndarray, hi: np.ndarray, p: "ConfocalParams") -> np.ndarray:
    """Inside of one closed mesh (microscope frame) on the fine grid of the box [lo, hi] (``work_nm / supersample``)."""
    s = int(p.supersample)
    shape = tuple(int(np.ceil((hi[k] - lo[k]) / p.work_nm)) * s for k in range(3))
    return occupancy_grid(v, f, lo, float(p.work_nm) / s, shape)


def _render(occ: np.ndarray, lo: np.ndarray, hi: np.ndarray, p: "ConfocalParams", phase: np.ndarray) -> Tuple[np.ndarray, list]:
    """Fine occupancy -> partial-volume fractions on the work grid -> PSF blur -> intensity at the confocal voxel centres."""
    from scipy import ndimage

    s = int(p.supersample)
    n = [d // s for d in occ.shape]
    work = occ.reshape(n[0], s, n[1], s, n[2], s).mean(axis=(1, 3, 5), dtype=np.float32)
    sigma = np.array([p.sigma_xy_nm, p.sigma_xy_nm, p.sigma_z_nm], dtype=float)
    blurred = ndimage.gaussian_filter(work, sigma=sigma / float(p.work_nm), mode="constant", truncate=4.0)
    vox = np.asarray(p.voxel_nm, dtype=float)
    axes = [np.arange(lo[k] + phase[k], hi[k], vox[k]) for k in range(3)]
    gi = np.meshgrid(*[(a - lo[k]) / p.work_nm - 0.5 for k, a in enumerate(axes)], indexing="ij")
    return ndimage.map_coordinates(blurred, gi, order=1, mode="constant", cval=0.0), axes


def trunk_cylinder(radius_nm: float, x_range: Tuple[float, float], sections: int = 48):
    """Trunk stand-in in the spine's local frame: a cylinder along x (dendrite tangent) whose surface passes through the
    attachment centre (origin), axis at y = -radius."""
    import trimesh

    length = float(x_range[1] - x_range[0])
    cyl = trimesh.creation.cylinder(radius=float(radius_nm), height=length, sections=sections)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))      # axis z -> x
    cyl.apply_translation([0.5 * (x_range[0] + x_range[1]), -float(radius_nm), 0.0])
    return cyl


def image_mesh(vertices: np.ndarray, faces: np.ndarray, params: ConfocalParams, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Confocal image of a closed spine mesh (local spine frame, nm) -> iso-surface mesh in the same frame + diagnostics.

    Without a trunk (``trunk_radius_nm`` None) the spine is imaged alone: iso-surface of its blurred image at ``threshold``.
    With a trunk the spine is imaged TOGETHER with a trunk cylinder (its fluorescence brightens the neck base, as in a real
    stack) and the spine is then cut from the image along the imaged trunk surface: the spine part is where the joint image
    exceeds the threshold AND the image of the trunk alone does not, i.e. the zero set of
    ``g = min(I_joint - t, t - I_trunk)``; its iso-surface is closed, its base follows the imaged trunk surface, like a
    LabID spine cut from the laboratory's surface (and resealed by our stage 1).

    Returns ``{"vertices", "faces", "detected", "info"}``; ``detected`` is False when nothing of the spine exceeds the
    threshold (invisible at this resolution / threshold); the mesh arrays are then empty."""
    from skimage import measure

    rng = rng or np.random.default_rng(0)
    p = params
    phi = float(rng.uniform(0.0, 2.0 * np.pi)) if p.optical_axis_angle is None else float(p.optical_axis_angle)
    R = _rotation_to_microscope(phi)
    v_local = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    sigma = np.array([p.sigma_xy_nm, p.sigma_xy_nm, p.sigma_z_nm], dtype=float)
    pad = 4.0 * sigma.max() + float(max(p.voxel_nm))
    vox = np.asarray(p.voxel_nm, dtype=float)
    phase = rng.uniform(0.0, 1.0, 3) * vox if p.random_phase else np.zeros(3)

    v = v_local @ R.T
    pts = [v]
    trunk = None
    if p.trunk_radius_nm:
        x0, x1 = v_local[:, 0].min() - pad, v_local[:, 0].max() + pad
        trunk = trunk_cylinder(p.trunk_radius_nm, (x0, x1))
        tv = np.asarray(trunk.vertices) @ R.T
        pts.append(tv)
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0) - pad, allp.max(axis=0) + pad
    if trunk is not None:
        # the box must not reach the cut ends of the cylinder: limit it along the dendrite (microscope X = local x)
        lo[0], hi[0] = v[:, 0].min() - 0.5 * pad, v[:, 0].max() + 0.5 * pad

    occ = _occupancy(v, f, lo, hi, p)
    image, axes = _render(occ, lo, hi, p, phase)
    info = {"optical_axis_angle": phi, "phase_nm": phase.tolist(), "image_shape": list(image.shape), "max_intensity": float(image.max()),
            "occupied_volume_nm3": float(occ.sum() * (float(p.work_nm) / int(p.supersample)) ** 3), "params": p.to_dict()}
    t = float(p.threshold)
    if trunk is None:
        field = image - t
    else:
        occ_trunk = _occupancy(tv, np.asarray(trunk.faces), lo, hi, p)
        trunk_only, _ = _render(occ_trunk, lo, hi, p, phase)
        joint, _ = _render(occ | occ_trunk, lo, hi, p, phase)
        field = np.minimum(joint - t, t - trunk_only)
        info["max_joint_intensity"] = float(joint.max())
    if p.noise_sd > 0:
        field = field + rng.normal(0.0, float(p.noise_sd), field.shape)
    if field.max() <= 0 or field.min() >= 0:
        return {"vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=np.int64), "detected": False, "info": info}
    # the field is HIGH inside (an SDF is low): gradient_direction="ascent" gives outward faces
    verts, tris, _, _ = measure.marching_cubes(field, level=0.0, spacing=tuple(vox), gradient_direction="ascent")
    verts = verts + np.array([a[0] for a in axes])
    if p.keep == "largest":
        verts, tris, n_comp = _largest_component(verts, tris)
        info["n_components"] = n_comp
    local = verts @ R
    # with a trunk: does the kept piece reach the trunk (neck visible) or is it a detached head? (y = 0 is the trunk surface)
    info["attached_to_trunk"] = bool(local[:, 1].min() < float(p.attached_max_gap_nm)) if trunk is not None else None
    info["base_gap_nm"] = float(local[:, 1].min())
    return {"vertices": local, "faces": tris.astype(np.int64), "detected": True, "info": info}


def _largest_component(verts: np.ndarray, tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    import trimesh

    mesh = trimesh.Trimesh(verts, tris, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return np.asarray(mesh.vertices), np.asarray(mesh.faces), len(parts)
    best = max(parts, key=lambda m: abs(m.volume) if m.is_watertight else m.area)
    return np.asarray(best.vertices), np.asarray(best.faces), len(parts)


def spine_seed(spine_key: str, seed: int) -> int:
    """Stable per-spine seed (optical-axis angle, sampling phase, noise) independent of the job order."""
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{spine_key}|{int(seed)}".encode()).digest()[:8], "little")
