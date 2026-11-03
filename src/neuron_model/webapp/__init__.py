"""Public web application of the branch generator (docs/neuron-model/webapp-architecture.md).

The core works without a web framework:

- ``bundle``  self-contained model folder ``models/webapp/<bundle_id>/`` (T0 + P0 + spine library + reference data);
- ``engine``  ``BranchGenerator(bundle).generate(request)``: T0 -> counts -> placements -> spines from the library -> assembly;
- ``metrics`` trunk / placement / per-spine characteristics of a generated branch;
- ``export``  zip archive with every mesh and table of a result.

``server.create_app(bundle_path)`` wraps it into a FastAPI app (fastapi is imported inside the function).
Units: nm everywhere; values for the page are converted explicitly (``factor`` of the metric description).
"""

from .bundle import ModelBundle, build_bundle
from .engine import BranchGenerator, GenerationResult, RequestError

__all__ = ["BranchGenerator", "GenerationResult", "ModelBundle", "RequestError", "build_bundle"]
