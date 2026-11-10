"""HTTP API + static page of the branch generator (FastAPI; imported inside ``create_app`` so the core works without it).

    GET  /api/health                              status and bundle id
    GET  /api/options                             categories with allowed values, defaults, limits, bundle description
    POST /api/generate                            JSON request (engine.REQUEST_FIELDS) -> result id + metrics + spine rows
    GET  /api/results/{id}/branch.off             assembled branch surface
    GET  /api/results/{id}/trunk.off              trunk tube alone
    GET  /api/results/{id}/spines/{k}.off?frame=local|global
    GET  /api/results/{id}/download.zip           every mesh and table (export.build_zip)
    GET  /api/results/{id}/metrics.json
    GET  /                                        the page (static/index.html; static/* served under /static)

Results are kept in a bounded in-memory LRU (``cache_size``). The result id is a hash of (bundle id, normalised
request) and generation is deterministic, so with ``cache_dir`` the request of every result is also written to disk
and an evicted result is regenerated transparently on the next download. Generation runs in the thread pool (sync
endpoint); at most ``max_concurrent`` generations run at once, further requests get 503 after ``queue_timeout_s``.
Request limits (length, soma distance, number of spines) come from the bundle manifest. Units of the API: nm.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import export
from .bundle import ModelBundle
from .engine import BranchGenerator, GenerationError, GenerationResult, RequestError, canonical_request_id

STATIC_DIR = Path(__file__).resolve().parent / "static"
LABELS_RU = {
    "species": {"mouse": "мышь", "human": "человек"},
    "cell_type_binary": {"excitatory": "возбуждающая", "inhibitory": "тормозная"},
    "compartment": {"apical": "апикальный", "basal": "базальный", "uncertain": "не определён", "inhibitory_dendrite": "дендрит тормозного нейрона"},
}
_ID = re.compile(r"^[0-9a-f]{16}$")


class ResultStore:
    """Bounded LRU of generation results (thread-safe), optional request cache on disk for regeneration."""

    def __init__(self, generator: BranchGenerator, size: int = 64, cache_dir: Optional[Path] = None) -> None:
        self.generator = generator
        self.size = max(1, int(size))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._items: "OrderedDict[str, Tuple[GenerationResult, Dict[str, bytes]]]" = OrderedDict()
        self._lock = threading.Lock()
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def put(self, result: GenerationResult) -> None:
        with self._lock:
            self._items[result.result_id] = (result, {})
            self._items.move_to_end(result.result_id)
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        if self.cache_dir:
            (self.cache_dir / f"{result.result_id}.json").write_text(export.dumps(export.request_document(result)), encoding="utf-8")

    def peek(self, result_id: str) -> Optional[Tuple[GenerationResult, Dict[str, bytes]]]:
        """Memory only (no regeneration)."""
        with self._lock:
            return self._items.get(result_id)

    def get(self, result_id: str) -> Optional[Tuple[GenerationResult, Dict[str, bytes]]]:
        with self._lock:
            item = self._items.get(result_id)
            if item is not None:
                self._items.move_to_end(result_id)
                return item
        path = self.cache_dir / f"{result_id}.json" if self.cache_dir and _ID.match(result_id) else None
        if path is None or not path.exists():
            return None
        request = json.loads(path.read_text(encoding="utf-8"))
        if request.pop("bundle_id", None) != self.generator.bundle.bundle_id:
            return None  # made with another bundle: cannot be reproduced here
        result = self.generator.generate(request)
        self.put(result)
        return self._items.get(result_id)


def options_document(bundle: ModelBundle) -> Dict[str, Any]:
    man = bundle.manifest
    return {
        "bundle": {"bundle_id": bundle.bundle_id, "created": man.get("created"), "units": man.get("units", "nm"), "models": man.get("models", {}),
                   "t0_run": Path(man["sources"]["t0_run"]).name, "p0_run": Path(man["sources"]["p0_run"]).name,
                   "spine_selection": man.get("spine_selection")},
        "categories": man["categories"], "labels": LABELS_RU, "spine_cell_type_binary": man.get("spine_cell_type_binary", {}),
        "trained_groups": man.get("trained_groups", []), "defaults": man["defaults"], "limits": man["limits"],
        "spine_library": {k: {"n_spines": v["n_spines"], "morphometrics": v.get("morphometrics", False)} for k, v in man.get("spine_library", {}).items()},
        "reference_available": bool(bundle.reference.get("trunk_placement")),
    }


def create_app(bundle_path: Path, *, cache_size: int = 64, cache_dir: Optional[Path] = None, max_concurrent: int = 2, queue_timeout_s: float = 60.0):
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    bundle = ModelBundle.load(Path(bundle_path))
    generator = BranchGenerator(bundle)
    store = ResultStore(generator, cache_size, cache_dir)
    slots = threading.BoundedSemaphore(max(1, int(max_concurrent)))
    app = FastAPI(title="Генератор ветвей дендрита с шипиками", version=str(bundle.bundle_id))

    def json_response(value: Any, status_code: int = 200) -> Response:
        return Response(content=export.dumps(value), media_type="application/json", status_code=status_code)

    def result_of(result_id: str) -> Tuple[GenerationResult, Dict[str, bytes]]:
        if not _ID.match(result_id):
            raise HTTPException(404, "результат не найден")
        item = store.get(result_id)
        if item is None:
            raise HTTPException(404, "результат не найден или устарел: повторите генерацию с теми же параметрами")
        return item

    def off_response(vertices, faces, name: str) -> Response:
        return Response(content=export.off_text(vertices, faces), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'inline; filename="{name}"'})

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "bundle_id": bundle.bundle_id}

    @app.get("/api/options")
    def options() -> Response:
        return json_response(options_document(bundle))

    @app.post("/api/generate")
    def generate(payload: Dict[str, Any] = Body(...)) -> Response:
        try:
            request = generator.normalize(payload)
        except RequestError as exc:
            return json_response({"errors": exc.errors}, 422)
        cached = store.peek(canonical_request_id(bundle.bundle_id, request))  # the same parameters were generated recently
        if cached is not None:
            return json_response(cached[0].summary())
        if not slots.acquire(timeout=queue_timeout_s):
            return json_response({"errors": [{"field": "server", "message": "сервер занят другими генерациями, повторите позже"}]}, 503)
        try:
            result = generator.generate(request)
        except GenerationError as exc:
            return json_response({"errors": [{"field": "generation", "message": str(exc)}]}, 422)
        finally:
            slots.release()
        store.put(result)
        return json_response(result.summary())

    @app.get("/api/results/{result_id}/branch.off")
    def branch_off(result_id: str) -> Response:
        result, _ = result_of(result_id)
        return off_response(*result.branch_mesh, f"branch_{result_id}.off")

    @app.get("/api/results/{result_id}/trunk.off")
    def trunk_off(result_id: str) -> Response:
        result, _ = result_of(result_id)
        return off_response(*result.trunk_mesh, f"trunk_{result_id}.off")

    @app.get("/api/results/{result_id}/spines/{index}.off")
    def spine_off(result_id: str, index: int, frame: str = Query("local")) -> Response:
        result, _ = result_of(result_id)
        if frame not in ("local", "global"):
            raise HTTPException(422, "frame: local или global")
        try:
            v, f = export.mesh_of(result, "spine", index, frame)
        except KeyError:
            raise HTTPException(404, f"шипика {index} нет на этой ветви (не поставлен или нет такого номера)")
        return off_response(v, f, f"spine_{index:03d}_{frame}.off")

    @app.get("/api/results/{result_id}/download.zip")
    def download(result_id: str) -> Response:
        result, files = result_of(result_id)
        if "zip" not in files:
            files["zip"] = export.build_zip(result)
        return Response(content=files["zip"], media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="branch_{result_id}.zip"'})

    @app.get("/api/results/{result_id}/metrics.json")
    def metrics_json(result_id: str) -> Response:
        result, _ = result_of(result_id)
        return json_response(export.metrics_document(result))

    @app.exception_handler(HTTPException)
    def http_error(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"errors": [{"field": "request", "message": str(exc.detail)}]}, status_code=exc.status_code)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
