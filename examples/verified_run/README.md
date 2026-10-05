# Recorded pilot experiments

These are actual local CPU results, not expected or fabricated outputs.

| Run | Development images | Test images | Labeled target objects in test images | Execution |
|---|---:|---:|---:|---|
| [Quick](quick/RESULTS.md) | 2 | 4 | 7 | Main notebook, executed in a local Jupyter kernel |
| [Full pilot](full/RESULTS.md) | 6 | 12 | 59 | `python -m robovision.experiment --full` |

Each run uses seed 2026, a 320-pixel maximum image side, blur lengths 7 and 17,
angle 25 degrees, noise sigma 0.01, and 60 ADMM iterations. Development PSNR
selected Wiener balance 0.01 and TV lambda 0.001. The fixed detector is SSDLite
MobileNetV3 COCO_V1 at score threshold 0.3 and IoU threshold 0.5.

## What the full pilot shows

At 17-pixel blur, the degraded images have mean PSNR 21.10 dB and 1 correct
detection out of 59 labeled objects. TV-ADMM raises mean PSNR to 27.11 dB and
recovers 7 objects. The clean-image detector finds 10 objects. The recovery is
partial, and the clean detector itself has low recall on this selected subset.

At 7-pixel blur, TV-ADMM increases true detections from 8 to 10. In the smaller
quick subset, the same method improves image quality without increasing the
number of detections at that blur level. This illustrates why small-sample
results and image-quality metrics must be interpreted carefully.

![Full-pilot image quality and detection recall](full/metrics.png)

These runs do not establish statistical significance or real-robot benefits.
The model has not been retrained, and no navigation policy was evaluated.

## Files

- `summary.csv`: aggregate quality, runtime, and pooled detection counts.
- `per_image.csv`: individual cases, including failures.
- `tuning.csv`: development parameter sweep.
- `selected_parameters.json`: frozen parameters used on test images.
- `data_used.json`: exact image IDs and the source manifest hash.
- `config.json` and `environment.json`: run configuration and tested versions.
- `predictions.json`: detector outputs retained by the five-category and confidence filters used in evaluation. These suffice to audit the recorded precision/recall. Complete unfiltered outputs remain in locally generated runs.
- `admm_history.json` and `convergence.png`: optimization diagnostics for the first case.

Original photographs and photo comparison panels are generated in local runs
and are not included here. Source URLs and image-license metadata are in the
main data manifest. The published figures above show numerical metrics only.

The recorded local wall times are approximately 93 seconds for the quick
experiment and 260 seconds for the full pilot. These include machine-dependent
work and some downloads; they are not promised Colab runtimes. Colab's hosted
runtime and AI2-THOR rendering have not been executed in this verification.
