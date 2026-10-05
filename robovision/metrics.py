"""Image quality and transparent fixed-threshold object detection metrics."""
import numpy as np
from skimage.metrics import structural_similarity


def quality(reference, estimate, crop=0):
    estimate = np.clip(estimate, 0, 1)
    if crop:
        if 2 * crop + 7 > min(reference.shape[:2]):
            raise ValueError("Image too small for this evaluation crop")
        reference, estimate = reference[crop:-crop, crop:-crop], estimate[crop:-crop, crop:-crop]
    mse = float(np.mean((reference - estimate) ** 2))
    return {"psnr": float(-10 * np.log10(max(mse, 1e-15))),
            "ssim": float(structural_similarity(reference, estimate, data_range=1, channel_axis=-1))}


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty(0)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(0, rb - lt), axis=1)
    a = np.prod(np.maximum(0, box[2:] - box[:2]))
    b = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    return intersection / np.maximum(a + b - intersection, 1e-12)


def detection_counts(prediction, truth, categories, score_threshold=0.3, iou_threshold=0.5):
    """Greedy score-ordered, class-aware, one-to-one matching; NOT COCO mAP."""
    gt_keep = np.isin(truth["labels"], list(categories))
    gt_boxes, gt_labels = truth["boxes"][gt_keep], truth["labels"][gt_keep]
    pred_keep = np.isin(prediction["labels"], list(categories)) & (prediction["scores"] >= score_threshold)
    boxes, labels, scores = (prediction[k][pred_keep] for k in ("boxes", "labels", "scores"))
    matched = np.zeros(len(gt_labels), dtype=bool)
    true_positives = 0
    for i in np.argsort(-scores, kind="stable"):
        eligible = np.flatnonzero((gt_labels == labels[i]) & ~matched)
        overlaps = box_iou(boxes[i], gt_boxes[eligible])
        if len(overlaps) and overlaps.max() >= iou_threshold:
            matched[eligible[np.argmax(overlaps)]] = True
            true_positives += 1
    return {"tp": true_positives, "fp": len(boxes) - true_positives,
            "fn": len(gt_labels) - true_positives, "n_gt": len(gt_labels)}
