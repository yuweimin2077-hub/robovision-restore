"""Development-set tuning, held-out evaluation, and export in one command."""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import time

import numpy as np
from .data import load_manifest, load_record
from .imaging import motion_psf, degrade, wiener_restore, tv_admm
from .metrics import quality, detection_counts
from .plots import image_comparison, summary_plot, convergence_plot


@dataclass
class Config:
    manifest: str = "data/coco_household18.json"
    output: str = "runs/quick"
    cache: str = "data/cache/images"
    max_side: int = 320
    blur_lengths: tuple = (7, 17)
    angle: float = 25.0
    sigma: float = 0.01
    seed: int = 2026
    iterations: int = 60
    rho: float = 0.15
    balances: tuple = (0.001, 0.003, 0.01)
    lambdas: tuple = (0.001, 0.003, 0.01)
    score_threshold: float = 0.3
    iou_threshold: float = 0.5
    quick: bool = True
    detection: bool = True
    device: str = "auto"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        raise ValueError("No rows to export")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_seed(config, image_id, blur_length):
    return int.from_bytes(hashlib.sha256(f"{config.seed}:{image_id}:{blur_length}".encode()).digest()[:4], "little")


def run(config=None):
    config = config or Config()
    start = time.perf_counter()
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config.manifest)
    categories = {int(k) for k in manifest["categories"]}
    dev = [r for r in manifest["records"] if r["split"] == "dev"]
    test = [r for r in manifest["records"] if r["split"] == "test"]
    if config.quick:
        dev, test = dev[:2], test[:4]
    margin = max(config.blur_lengths) // 2 + 1
    save_json(out / "config.json", asdict(config))
    save_json(out / "data_used.json", {"name": manifest["name"], "dev_ids": [r["id"] for r in dev],
                                      "test_ids": [r["id"] for r in test],
                                      "manifest_sha256": hashlib.sha256(Path(config.manifest).read_bytes()).hexdigest()})
    print(f"Dataset: {manifest['name']}", flush=True)
    print(f"Tune on {len(dev)} images; evaluate on {len(test)} held-out images.", flush=True)

    # Cache development observations so every candidate sees identical noise.
    development_cases = []
    for record in dev:
        clean, _ = load_record(record, manifest, config.cache, config.max_side)
        for length in config.blur_lengths:
            kernel = motion_psf(length, config.angle)
            observation = degrade(clean, kernel, config.sigma, case_seed(config, record["id"], length))
            development_cases.append((clean, observation, kernel))
    tuning = []
    selected = {}
    for method, candidates in (("wiener", config.balances), ("tv_admm", config.lambdas)):
        for value in candidates:
            scores = []
            for clean, observation, kernel in development_cases:
                restored = (wiener_restore(observation, kernel, value) if method == "wiener" else
                            tv_admm(observation, kernel, value, config.rho, config.iterations)[0])
                scores.append(quality(clean, restored, margin)["psnr"])
            tuning.append({"method": method, "parameter": value, "dev_mean_psnr": float(np.mean(scores))})
        best = max((r for r in tuning if r["method"] == method), key=lambda r: r["dev_mean_psnr"])
        selected[method] = best["parameter"]
        print(f"Selected {method}: {best['parameter']} (development PSNR only)", flush=True)
    save_csv(out / "tuning.csv", tuning)
    save_json(out / "selected_parameters.json", selected)

    detector = None
    if config.detection:
        from .detection import Detector
        print("Loading fixed SSDLite detector...", flush=True)
        detector = Detector(config.device)
    rows, predictions_export, first = [], [], True
    for record in test:
        clean, truth = load_record(record, manifest, config.cache, config.max_side)
        clean_prediction = detector(clean) if detector else None
        for length in config.blur_lengths:
            kernel = motion_psf(length, config.angle)
            observed = degrade(clean, kernel, config.sigma, case_seed(config, record["id"], length))
            tick = time.perf_counter()
            wiener = wiener_restore(observed, kernel, selected["wiener"])
            wiener_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            tv, history = tv_admm(observed, kernel, selected["tv_admm"], config.rho, config.iterations)
            tv_seconds = time.perf_counter() - tick
            images = {"clean": clean, "degraded": observed, "wiener": wiener, "tv_admm": tv}
            metrics, predictions = {}, {}
            for method, image in images.items():
                metrics[method] = quality(clean, image, margin)
                # Clean PSNR is conceptually infinite; leave it blank in exported data.
                if method == "clean":
                    metrics[method]["psnr"] = None
                row = {"image_id": record["id"], "blur_length": length, "method": method,
                       **metrics[method], "restoration_seconds": {"wiener": wiener_seconds, "tv_admm": tv_seconds}.get(method, 0)}
                if detector:
                    prediction = clean_prediction if method == "clean" else detector(image)
                    predictions[method] = prediction
                    row.update(detection_counts(prediction, truth, categories, config.score_threshold, config.iou_threshold))
                    predictions_export.append({"image_id": record["id"], "blur_length": length, "method": method,
                                               **{k: v.tolist() for k, v in prediction.items()}})
                rows.append(row)
            gallery = out / "gallery"
            gallery.mkdir(exist_ok=True)
            image_comparison(images, metrics, gallery / f"{record['id']}_blur{length}.png")
            if detector:
                image_comparison(images, metrics, gallery / f"{record['id']}_blur{length}_detections.png",
                                 predictions, truth, detector.names, config.score_threshold, categories)
            if first:
                image_comparison(images, metrics, out / "comparison.png")
                if detector:
                    image_comparison(images, metrics, out / "detection_comparison.png", predictions, truth,
                                     detector.names, config.score_threshold, categories)
                convergence_plot(history, out / "convergence.png")
                save_json(out / "admm_history.json", history)
                first = False
            print(f"Evaluated image {record['id']}, blur length {length}", flush=True)
    summary = []
    for length in config.blur_lengths:
        for method in ("clean", "degraded", "wiener", "tv_admm"):
            subset = [r for r in rows if r["blur_length"] == length and r["method"] == method]
            item = {"blur_length": length, "method": method, "n_images": len(subset),
                    "psnr": None if method == "clean" else float(np.mean([r["psnr"] for r in subset])),
                    "ssim": float(np.mean([r["ssim"] for r in subset])),
                    "restoration_seconds": float(np.mean([r["restoration_seconds"] for r in subset]))}
            if detector:
                tp, fp, fn = (sum(r[k] for r in subset) for k in ("tp", "fp", "fn"))
                item.update({"tp": tp, "fp": fp, "fn": fn, "recall": tp / (tp + fn) if tp + fn else None,
                             "precision": tp / (tp + fp) if tp + fp else None})
            summary.append(item)
    save_csv(out / "per_image.csv", rows)
    save_csv(out / "summary.csv", summary)
    save_json(out / "predictions.json", predictions_export)
    summary_plot(summary, out / "metrics.png")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "utc": datetime.now(timezone.utc).isoformat(), "device": detector.device if detector else "no detector",
                   "elapsed_seconds": time.perf_counter() - start}
    for package in ("numpy", "scipy", "scikit-image", "matplotlib", "Pillow", "torch", "torchvision"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    save_json(out / "environment.json", environment)
    report = ["# Experiment results", "", f"Dataset: {manifest['name']}",
              f"Development images: {len(dev)}. Held-out images: {len(test)}. Seed: {config.seed}.",
              "", "| Blur | Method | PSNR (dB) | SSIM | Recall | Precision |", "|---|---|---|---|---|---|"]
    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    for r in summary:
        report.append(f"| {r['blur_length']} | {r['method']} | {fmt(r['psnr'])} | {fmt(r['ssim'])} | {fmt(r.get('recall'))} | {fmt(r.get('precision'))} |")
    report += ["", "Recall/precision use one-to-one, class-aware matching at fixed thresholds; these are NOT COCO mAP.",
               f"Score threshold = {config.score_threshold}; IoU threshold = {config.iou_threshold}.",
               f"Quality metrics exclude a {margin}-pixel border; detection uses full resized images.",
               "Known blur kernel, circular convolution, additive Gaussian noise in RGB pixel space; this is a controlled approximation.",
               "Parameters were chosen using development PSNR only. Clean test images are used only for synthetic degradation and evaluation.",
               "This small pilot does not establish statistical significance, real-robot performance, or navigation success.",
               "The first comparison image was selected by manifest order, not by the size of the improvement."]
    (out / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved figures and tables to {out}. Total time: {environment['elapsed_seconds']:.1f}s", flush=True)
    return {"summary": summary, "parameters": selected, "output": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=Config.manifest)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-detection", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    run(Config(manifest=args.manifest, output=args.output, quick=not args.full,
               detection=not args.no_detection, device=args.device))


if __name__ == "__main__":
    main()
