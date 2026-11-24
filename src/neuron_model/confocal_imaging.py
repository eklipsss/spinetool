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
5. segment, two variants (``surface``):
   - ``isosurface``: iso-surface at ``threshold`` x the intensity of a fully filled large region (Marching Cubes), largest
     component;
   - ``spinetool`` (2026-11-25): the procedure of SpineTool 1 that built the LabID meshes (``spine_segmentation.py``,
     ``notebooks/dendrite/dendrite-segmentation.ipynb``): 8-bit image, Gaussian smoothing with sigma = 1 voxel, local
     threshold ``T = base - w (base - median_3x3x3)`` with ``base = 255 threshold``, largest connected component, boundary
     voxels of the binary image (binary minus its erosion) with normals = gradient of the negated binary image, Poisson
     reconstruction IN VOXEL COORDINATES, then scaling by the voxel size (``spinetool_surface``).

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
    surface: str = "isosurface"                                    # isosurface | spinetool (last step of SpineTool 1)
    st_weight: float = 0.05                                        # SpineTool local threshold weight w
    st_block: int = 3                                              # SpineTool median window (voxels)
    st_presmooth_voxels: float = 1.0                               # Gaussian sigma before thresholding, in voxels of each axis
    st_8bit: bool = True                                           # quantise the image to 0..255 (saturates above the full fill)
    st_poisson: str = "auto"                                       # auto (pymeshlab, then binary_mc) | pymeshlab | cgal (subprocess) | binary_mc

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
    if p.surface == "spinetool":
        bin_trunk = None
        if trunk is None:
            binary = spinetool_binary(image, p)
        else:
            bin_trunk = spinetool_binary(trunk_only, p)
            binary = spinetool_binary(joint, p) & ~bin_trunk
        binary = _largest_binary_component(binary)
        info["n_voxels"] = int(binary.sum())
        if binary.sum() < 8:
            return {"vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=np.int64), "detected": False, "info": info}
        verts, tris, backend = spinetool_surface(binary, vox, p.st_poisson)
        info["poisson_backend"] = backend
        verts = verts + np.array([a[0] for a in axes])
        if p.keep == "largest" and len(tris):
            verts, tris, n_comp = _largest_component(verts, tris)
            info["n_components"] = n_comp
        local = verts @ R
        # attached = the kept spine region touches the region of the IMAGED trunk (at a low threshold the imaged trunk is
        # a few hundred nm thicker than the membrane, so the height above y = 0 is not the right test here)
        if bin_trunk is not None:
            from scipy import ndimage

            touch = ndimage.binary_dilation(binary, structure=np.ones((3, 3, 3), dtype=bool)) & bin_trunk
            info["attached_to_trunk"] = bool(touch.any())
        else:
            info["attached_to_trunk"] = None
        info["base_gap_nm"] = float(local[:, 1].min()) if len(local) else float("nan")
        return {"vertices": local, "faces": np.asarray(tris, dtype=np.int64), "detected": bool(len(tris)), "info": info}
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


def spinetool_binary(intensity: np.ndarray, p: "ConfocalParams") -> np.ndarray:
    """SpineTool 1 segmentation of an image whose full fill is 1: optional 8-bit quantisation, Gaussian sigma in voxels,
    local threshold ``base - w (base - median)``, ``base = 255 * threshold`` (``spine_segmentation.local_threshold_3d``)."""
    from scipy import ndimage

    img = np.asarray(intensity, dtype=np.float64)
    scale = 255.0
    if p.st_8bit:
        img = np.clip(np.round(img * scale), 0, 255)
    else:
        img = img * scale
    if p.st_presmooth_voxels > 0:
        img = ndimage.gaussian_filter(img, float(p.st_presmooth_voxels))
    base = float(p.threshold) * scale
    thr = base - float(p.st_weight) * (base - ndimage.median_filter(img, size=int(p.st_block)))
    return img > thr


def _largest_binary_component(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return binary
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def spinetool_surface(binary: np.ndarray, voxel_nm: np.ndarray, backend: str = "auto") -> Tuple[np.ndarray, np.ndarray, str]:
    """Boundary voxels of the binary image with normals = gradient of the negated image, Poisson reconstruction in voxel
    coordinates, then scaling by the voxel size (as SpineTool: "small coordinates give worse reconstruction results").
    Returns (vertices in nm relative to voxel index 0, faces, backend used)."""
    from scipy import ndimage

    b = np.asarray(binary, dtype=bool)
    padded = np.pad(b, 2)
    edges = padded & ~ndimage.binary_erosion(padded)
    pts = np.argwhere(edges).astype(np.float64)
    grad = np.gradient(-padded.astype(np.float64) * 255.0)
    idx = tuple(pts.astype(int).T)
    normals = np.stack([g[idx] for g in grad], axis=1)
    ok = np.linalg.norm(normals, axis=1) > 0
    pts, normals = pts[ok], normals[ok]
    vox = np.asarray(voxel_nm, dtype=np.float64)
    order = ["pymeshlab", "binary_mc"] if backend == "auto" else [backend]
    for name in order:
        try:
            if name == "cgal":
                v, f = _poisson_cgal(pts, normals)
            elif name == "pymeshlab":
                from .spine_generation.reconstruction.screened_poisson import PoissonParams, screened_poisson

                m = screened_poisson(pts, normals, PoissonParams(depth=8, point_weight=4.0))
                v, f = np.asarray(m.vertices), np.asarray(m.faces)
            elif name == "binary_mc":
                from skimage import measure

                smooth = ndimage.gaussian_filter(padded.astype(np.float64), 1.0)
                v, f, _, _ = measure.marching_cubes(smooth, level=0.5, gradient_direction="ascent")
            else:
                raise ValueError(f"unknown Poisson backend {name!r}")
            if len(f) == 0:
                raise RuntimeError("empty reconstruction")
            return (np.asarray(v, dtype=np.float64) - 2.0) * vox, np.asarray(f, dtype=np.int64), name
        except Exception:  # noqa: BLE001 - try the next backend (CGAL without Eigen, no pymeshlab on the mac)
            if backend != "auto":
                raise
    raise RuntimeError("no Poisson backend worked")


def _poisson_cgal(points: np.ndarray, normals: np.ndarray, timeout_s: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
    """CGAL Poisson (SpineTool's mesher) in a subprocess: the binding can abort the interpreter."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import trimesh

    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.off")
        np.savez(src, points=points, normals=normals)
        res = subprocess.run([sys.executable, "-m", "src.neuron_model._cgal_poisson_worker", src, dst], cwd=str(repo),
                             capture_output=True, timeout=timeout_s)
        if res.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError(f"CGAL Poisson failed (exit {res.returncode}): {res.stderr.decode(errors='replace')[-300:]}")
        m = trimesh.load(dst, process=False, force="mesh")
    return np.asarray(m.vertices), np.asarray(m.faces)


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
