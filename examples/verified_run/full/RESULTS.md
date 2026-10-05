# Experiment results

Dataset: COCO household-object pilot (not robot data)
Development images: 6. Held-out images: 12. Seed: 2026.

| Blur | Method | PSNR (dB) | SSIM | Recall | Precision |
|---|---|---|---|---|---|
| 7 | clean | N/A | 1.000 | 0.169 | 0.667 |
| 7 | degraded | 25.493 | 0.787 | 0.136 | 0.571 |
| 7 | wiener | 26.780 | 0.717 | 0.153 | 0.643 |
| 7 | tv_admm | 30.192 | 0.897 | 0.169 | 0.714 |
| 17 | clean | N/A | 1.000 | 0.169 | 0.667 |
| 17 | degraded | 21.103 | 0.566 | 0.017 | 0.167 |
| 17 | wiener | 24.476 | 0.648 | 0.085 | 0.500 |
| 17 | tv_admm | 27.108 | 0.818 | 0.119 | 0.636 |

Recall/precision use one-to-one, class-aware matching at fixed thresholds; these are NOT COCO mAP.
Score threshold = 0.3; IoU threshold = 0.5.
Quality metrics exclude a 9-pixel border; detection uses full resized images.
Known blur kernel, circular convolution, additive Gaussian noise in RGB pixel space; this is a controlled approximation.
Parameters were chosen using development PSNR only. Clean test images are used only for synthetic degradation and evaluation.
This small pilot does not establish statistical significance, real-robot performance, or navigation success.
The first comparison image was selected by manifest order, not by the size of the improvement.
