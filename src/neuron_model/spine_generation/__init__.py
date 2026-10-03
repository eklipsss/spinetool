"""S-module spine generators (docs/neuron-model/s-module-technical-spec.md).

Subpackages:

- ``experiment`` - YAML configs, seeding, run directories, metric logging,
  checkpoint/resume, sample artifacts, shared training helpers;
- ``data`` - preprocessed-dataset index (manifest.parquet), fixed group split,
  torch datasets for point clouds and SDF samples, frozen bounding box;
- ``reconstruction`` - SDF -> Marching Cubes, point cloud -> normals ->
  Screened Poisson, mesh postprocessing and validation;
- ``models.spine_vae`` / ``models.spine_mogen`` - the generators;
- ``evaluation`` - morphometric / distribution / diversity metrics.

See docs/neuron-model/s-module-implementation-plan.md for the stage plan.
"""
