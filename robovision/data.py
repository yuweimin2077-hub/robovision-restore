"""Small manifests with explicit labels, splits, and source provenance."""
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = [r["id"] for r in manifest["records"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Image IDs must be unique")
    dev_groups = {r["group"] for r in manifest["records"] if r["split"] == "dev"}
    test_groups = {r["group"] for r in manifest["records"] if r["split"] == "test"}
    if not dev_groups or not test_groups or dev_groups & test_groups:
        raise ValueError("Nonempty, disjoint development and test groups are required")
    manifest["_directory"] = str(path.parent)
    return manifest


def load_record(record, manifest, cache_dir="data/cache/images", max_side=384):
    if "path" in record:
        path = Path(manifest["_directory"]) / record["path"]
    else:
        path = Path(cache_dir) / record["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".part")
            try:
                with urllib.request.urlopen(record["url"], timeout=60) as source:
                    temporary.write_bytes(source.read())
                with Image.open(temporary) as check:
                    check.verify()
                temporary.replace(path)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Could not download image {record['id']}. Check the connection and rerun.") from exc
    with Image.open(path) as original:
        image = original.convert("RGB")
        ow, oh = image.size
        if (ow, oh) != (record["width"], record["height"]):
            raise ValueError(f"Image/annotation size mismatch for {record['id']}")
        scale = min(1.0, max_side / max(ow, oh))
        width, height = round(ow * scale), round(oh * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float64) / 255
    boxes, labels = [], []
    for annotation in record["annotations"]:
        x, y, w, h = annotation["bbox"]
        boxes.append([x * width / ow, y * height / oh, (x + w) * width / ow, (y + h) * height / oh])
        labels.append(annotation["category_id"])
    return array, {"boxes": np.asarray(boxes, dtype=float).reshape(-1, 4), "labels": np.asarray(labels, dtype=int)}
