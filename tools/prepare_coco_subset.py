"""Build the small manifest from original COCO annotations, once per project.

End users only download the selected JPEGs, not this full annotation archive.
Selection uses annotations and a fixed seed, never detector performance.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
CATEGORIES = {44, 47, 51, 62, 84}


def main():
    cache = ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "annotations_trainval2017.zip"
    if not archive.exists():
        print("Downloading official COCO annotation archive...", flush=True)
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        raw = z.read("annotations/instances_val2017.json")
    coco = json.loads(raw)
    annotations = {}
    for a in coco["annotations"]:
        if a["category_id"] in CATEGORIES:
            annotations.setdefault(a["image_id"], []).append(a)
    candidates = []
    for item in sorted(coco["images"], key=lambda x: x["id"]):
        aa = annotations.get(item["id"], [])
        scale = 384 / max(item["width"], item["height"])
        if aa and not any(a["iscrowd"] for a in aa) and any(a["category_id"] in {47, 51} and min(a["bbox"][2:]) * scale >= 30 for a in aa):
            candidates.append(item)
    rng = random.Random(2026)
    selected = rng.sample(candidates, 18)
    records = []
    for index, item in enumerate(selected):
        records.append({"id": str(item["id"]), "file_name": item["file_name"],
                        "url": "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + item["file_name"],
                        "width": item["width"], "height": item["height"],
                        "split": "dev" if index < 6 else "test", "group": str(item["id"]),
                        "license_id": item["license"], "flickr_url": item.get("flickr_url"),
                        "annotations": [{"bbox": a["bbox"], "category_id": a["category_id"],
                                         "annotation_id": a["id"]} for a in annotations[item["id"]]]})
    result = {"name": "COCO household-object pilot (not robot data)",
              "source": URL, "source_annotation_sha256": hashlib.sha256(raw).hexdigest(),
              "selection_seed": 2026, "selection_rule": "18 random images with >=1 cup or bowl >=30px at 384px long side; exclude selected-category crowds; retain all selected-category annotations",
              "categories": {str(c["id"]): c["name"] for c in coco["categories"] if c["id"] in CATEGORIES},
              "licenses": coco["licenses"], "records": records}
    path = ROOT / "data/coco_household18.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} records to {path}")


if __name__ == "__main__":
    main()
