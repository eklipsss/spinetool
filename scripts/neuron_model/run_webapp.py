"""Serve the web application of the branch generator (FastAPI + uvicorn; ``pip install fastapi uvicorn``).

    python scripts/neuron_model/run_webapp.py --bundle models/webapp/<bundle-id> [--host 127.0.0.1 --port 8000]

Open http://127.0.0.1:8000/ in a browser. One process (results live in its memory); generation is CPU-only (0.1-0.3 s for a
15 um branch with 10 spines on the mac; grows with the number and size of the spine meshes). ``--cache-dir``: keep the request of every result on disk, so a result evicted from the
in-memory cache is regenerated (bit for bit) on the next download. For a public deployment put a reverse proxy (TLS,
rate limits) in front and bind to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  (first on Windows: see generate_tp.py)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="models/webapp/<bundle-id>")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-size", type=int, default=64, help="results kept in memory")
    parser.add_argument("--cache-dir", type=Path, default=None, help="requests of the results on disk (regeneration after eviction)")
    parser.add_argument("--max-concurrent", type=int, default=2, help="generations running at once")
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is not installed: python -m pip install fastapi uvicorn")

    from src.neuron_model.webapp.server import create_app

    app = create_app(args.bundle, cache_size=args.cache_size, cache_dir=args.cache_dir, max_concurrent=args.max_concurrent)
    print(f"[webapp] bundle {args.bundle} -> http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
