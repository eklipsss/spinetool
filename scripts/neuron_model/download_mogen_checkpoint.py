"""Download a public MoGen checkpoint (gs://mogen-release/models/<name>) over HTTPS - no gsutil needed.

    python scripts/neuron_model/download_mogen_checkpoint.py --name mouse_mixed --out data/external/mogen

Result: ``data/external/mogen/mouse_mixed/{config.json, best_checkpoints/<step>/...}`` (~224 MB for
mouse_mixed). Files already present with the right size are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "mogen-release"
LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={prefix}&fields=items(name,size),nextPageToken"
FILE_URL = "https://storage.googleapis.com/{bucket}/{name}"


def list_objects(prefix: str):
    token = None
    while True:
        url = LIST_URL.format(bucket=BUCKET, prefix=urllib.parse.quote(prefix, safe=""))
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        yield from payload.get("items", [])
        token = payload.get("nextPageToken")
        if not token:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="mouse_mixed")
    parser.add_argument("--out", type=Path, default=Path("data/external/mogen"))
    args = parser.parse_args()
    prefix = f"models/{args.name}/"
    items = list(list_objects(prefix))
    if not items:
        sys.exit(f"nothing found under gs://{BUCKET}/{prefix}")
    total = sum(int(i["size"]) for i in items)
    print(f"{len(items)} files, {total / 1e6:.1f} MB")
    for item in items:
        target = args.out / item["name"][len("models/"):]
        if target.exists() and target.stat().st_size == int(item["size"]):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(FILE_URL.format(bucket=BUCKET, name=urllib.parse.quote(item["name"])), tmp)
        if tmp.stat().st_size != int(item["size"]):
            sys.exit(f"size mismatch for {item['name']}")
        tmp.replace(target)
        print("  ", target)
    print("done ->", args.out / args.name)


if __name__ == "__main__":
    main()
