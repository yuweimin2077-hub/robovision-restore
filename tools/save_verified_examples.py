"""Copy completed numerical evidence into a small publishable example folder.

Source photographs and derived photo panels stay in local runs. This folder
contains tables, convergence/metric plots, provenance, and environment records.
"""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ("RESULTS.md", "summary.csv", "per_image.csv", "tuning.csv",
         "selected_parameters.json", "data_used.json", "environment.json", "config.json",
         "admm_history.json", "convergence.png", "metrics.png", "predictions.json")


def main():
    sources = {
        "quick": ROOT / "tmp/notebook_verification/robovision_colab_workspace/runs/colab_quick",
        "full": ROOT / "runs/tabletop_full",
    }
    for label, source in sources.items():
        missing = [name for name in FILES if not (source / name).is_file()]
        if missing:
            raise RuntimeError(f"Run {label} is incomplete: {missing}")
        target = ROOT / "examples/verified_run" / label
        target.mkdir(parents=True, exist_ok=True)
        config = json.loads((source / "config.json").read_text())
        categories = {int(k) for k in json.loads((ROOT / config["manifest"]).read_text())["categories"]}
        for name in FILES:
            if name == "predictions.json":
                predictions = json.loads((source / name).read_text())
                for item in predictions:
                    keep = [i for i, (label, score) in enumerate(zip(item["labels"], item["scores"]))
                            if label in categories and score >= config["score_threshold"]]
                    for key in ("boxes", "labels", "scores"):
                        item[key] = [item[key][i] for i in keep]
                (target / name).write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8", newline="\n")
            else:
                shutil.copy2(source / name, target / name)
        print(f"Saved {label}: {len(FILES)} evidence files")


if __name__ == "__main__":
    main()
