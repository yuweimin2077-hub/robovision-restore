"""Tests for scientific errors that can invalidate an experiment."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from robovision.imaging import motion_psf, blur, degrade, gradient, gradient_adjoint, tv_admm, wiener_restore
from robovision.metrics import detection_counts, quality
from robovision.data import load_manifest, load_record
from robovision.ai2thor_data import annotations_from_event


def test_gradient_adjoint():
    rng = np.random.default_rng(8)
    image = rng.normal(size=(19, 22, 3))
    field = rng.normal(size=(19, 22, 3, 2))
    assert np.allclose(np.sum(gradient(image) * field), np.sum(image * gradient_adjoint(field)), atol=1e-10)


def test_kernel_flux_and_center():
    image = np.zeros((31, 31, 3))
    image[15, 15] = 1
    kernel = motion_psf(9, 0)
    observed = blur(image, kernel)
    assert np.isclose(kernel.sum(), 1)
    assert np.allclose(observed[15, 11:20, 0], kernel[4, :])
    assert np.allclose(blur(image, motion_psf(1)), image)
    assert np.allclose(blur(np.ones_like(image), kernel), 1)


def test_forward_model_noise_is_reproducible_and_unclipped():
    image = np.zeros((30, 30, 3))
    first = degrade(image, motion_psf(1), 0.01, 99)
    assert np.array_equal(first, degrade(image, motion_psf(1), 0.01, 99))
    assert np.min(first) < 0


def test_tv_restores_phantom_and_satisfies_constraint():
    clean = np.full((64, 64, 3), 0.15)
    clean[15:45, 20:50] = [0.85, 0.6, 0.4]
    kernel = motion_psf(9, 20)
    observation = degrade(clean, kernel, 0.01, 4)
    restored, history = tv_admm(observation, kernel, lam=0.003, iterations=150)
    assert quality(clean, restored)["psnr"] > quality(clean, observation)["psnr"] + 3
    assert history[-1]["primal_rms"] < history[0]["primal_rms"]
    assert np.isfinite(restored).all()


def test_wiener_matches_closed_form_identity_case():
    image = np.random.default_rng(0).uniform(size=(12, 15, 3))
    assert np.allclose(wiener_restore(image, motion_psf(1), balance=0.1), image / 1.1)


def test_detection_matching_rejects_duplicates_wrong_classes_and_bad_overlap():
    truth = {"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), "labels": np.array([47, 44])}
    pred = {"boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30], [100, 100, 110, 110]]),
            "labels": np.array([47, 47, 47, 44]), "scores": np.array([0.9, 0.8, 0.7, 0.6])}
    counts = detection_counts(pred, truth, {44, 47})
    assert counts == {"tp": 1, "fp": 3, "fn": 1, "n_gt": 2}


def test_no_predictions():
    truth = {"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([47])}
    pred = {"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=int), "scores": np.empty(0)}
    assert detection_counts(pred, truth, {47})["fn"] == 1


def test_split_leakage_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"records": [{"id": "a", "split": "dev", "group": "scene1"},
                                            {"id": "b", "split": "test", "group": "scene1"}]}))
    with pytest.raises(ValueError, match="disjoint"):
        load_manifest(path)


def test_box_resize_matches_actual_integer_dimensions(tmp_path):
    Image.new("RGB", (101, 63)).save(tmp_path / "image.png")
    record = {"id": "a", "path": "image.png", "width": 101, "height": 63,
              "annotations": [{"bbox": [0, 0, 101, 63], "category_id": 47}]}
    image, truth = load_record(record, {"_directory": str(tmp_path)}, max_side=50)
    assert image.shape == (31, 50, 3)
    assert np.allclose(truth["boxes"], [[0, 0, 50, 31]])


def test_simulator_annotations_use_ground_truth_and_correct_category():
    class Event:
        metadata = {"objects": [{"objectId": "Mug|1", "objectType": "Mug"}]}
        instance_detections2D = {"Mug|1": np.array([2, 3, 12, 23])}
    assert annotations_from_event(Event())[0]["bbox"] == [2., 3., 10., 20.]
    assert annotations_from_event(Event())[0]["category_id"] == 47


def test_committed_manifest_has_disjoint_splits_and_real_annotations():
    manifest = load_manifest(Path(__file__).resolve().parents[1] / "data/coco_household18.json")
    assert len(manifest["records"]) == 18
    assert all(r["annotations"] for r in manifest["records"])
    assert all("annotation_id" in a for r in manifest["records"] for a in r["annotations"])
