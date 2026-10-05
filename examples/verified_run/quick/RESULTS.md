# Experiment results

Dataset: COCO household-object pilot (not robot data)
Development images: 2. Held-out images: 4. Seed: 2026.

| Blur | Method | PSNR (dB) | SSIM | Recall | Precision |
|---|---|---|---|---|---|
| 7 | clean | N/A | 1.000 | 0.571 | 0.571 |
| 7 | degraded | 24.606 | 0.793 | 0.429 | 0.429 |
| 7 | wiener | 26.647 | 0.731 | 0.429 | 0.500 |
| 7 | tv_admm | 29.892 | 0.904 | 0.429 | 0.500 |
| 17 | clean | N/A | 1.000 | 0.571 | 0.571 |
| 17 | degraded | 20.062 | 0.561 | 0.143 | 0.333 |
| 17 | wiener | 23.942 | 0.655 | 0.143 | 0.333 |
| 17 | tv_admm | 26.553 | 0.827 | 0.429 | 0.500 |

Recall/precision use one-to-one, class-aware matching at fixed thresholds; these are NOT COCO mAP.
Score threshold = 0.3; IoU threshold = 0.5.
Quality metrics exclude a 9-pixel border; detection uses full resized images.
Known blur kernel, circular convolution, additive Gaussian noise in RGB pixel space; this is a controlled approximation.
Parameters were chosen using development PSNR only. Clean test images are used only for synthetic degradation and evaluation.
This small pilot does not establish statistical significance, real-robot performance, or navigation success.
The first comparison image was selected by manifest order, not by the size of the improvement.
