"""Optional robot-view dataset collection for a Linux GPU runtime.

Scene IDs define the development/test split. Bounding boxes come from
instance segmentation, never from detector predictions. No learned action
policy or navigation benchmark is implemented in this collection script.
"""
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

MAPPING = {"Bottle": 44, "Mug": 47, "Cup": 47, "Bowl": 51, "Chair": 62, "Book": 84}
NAMES = {"44": "bottle", "47": "cup", "51": "bowl", "62": "chair", "84": "book"}


def annotations_from_event(event):
    """Convert visible segmentation extents into COCO-category xywh boxes."""
    objects = {o["objectId"]: o for o in event.metadata["objects"]}
    annotations = []
    for object_id, box in (event.instance_detections2D or {}).items():
        object_type = objects.get(object_id, {}).get("objectType")
        if object_type not in MAPPING:
            continue
        x0, y0, x1, y1 = map(float, box)
        if x1 > x0 and y1 > y0:
            annotations.append({"bbox": [x0, y0, x1 - x0, y1 - y0],
                                "category_id": MAPPING[object_type], "object_id": object_id})
    return annotations


def collect(output="data/ai2thor", per_scene=6, seed=2026, cloud_rendering=True):
    from ai2thor.controller import Controller
    output = Path(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    kwargs = dict(width=384, height=384, renderInstanceSegmentation=True,
                  gridSize=0.25, fieldOfView=90)
    if cloud_rendering:
        from ai2thor.platform import CloudRendering
        kwargs["platform"] = CloudRendering
    controller = None
    records = []
    rng = random.Random(seed)
    # Nearby video frames must not leak across development and test splits.
    scenes = [("FloorPlan1", "dev"), ("FloorPlan5", "dev"),
              ("FloorPlan2", "test"), ("FloorPlan6", "test")]
    try:
        controller = Controller(scene=scenes[0][0], **kwargs)
        for scene, split in scenes:
            controller.reset(scene)
            response = controller.step(action="GetReachablePositions")
            if not response.metadata["lastActionSuccess"]:
                raise RuntimeError(response.metadata["errorMessage"])
            positions = response.metadata["actionReturn"]
            count = 0
            for attempt in range(per_scene * 40):
                event = controller.step(action="Teleport", position=rng.choice(positions),
                                        rotation={"x": 0, "y": rng.randrange(0, 360, 30), "z": 0},
                                        horizon=30, standing=True)
                if not event.metadata["lastActionSuccess"]:
                    continue
                annotations = annotations_from_event(event)
                if not annotations or not any(min(a["bbox"][2:]) >= 20 for a in annotations):
                    continue
                identifier = f"{scene}_{count:03d}"
                relative = f"images/{identifier}.png"
                Image.fromarray(event.frame).save(output / relative)
                records.append({"id": identifier, "path": relative, "width": 384, "height": 384,
                                "split": split, "group": scene, "annotations": annotations,
                                "agent_pose": event.metadata["agent"]})
                count += 1
                if count >= per_scene:
                    break
            if count < per_scene:
                raise RuntimeError(f"Only collected {count}/{per_scene} usable views in {scene}; inspect scene and sampling settings.")
            print(f"Collected {count} views from {scene} ({split})", flush=True)
    finally:
        if controller is not None:
            controller.stop()
    manifest = {"name": "AI2-THOR robot-view perception pilot (no navigation policy)",
                "source": "https://github.com/allenai/ai2thor", "seed": seed,
                "label_note": "Visible instance-mask bounding boxes; Mug/Cup map to COCO cup. Label definitions differ from COCO.",
                "categories": NAMES, "records": records}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
