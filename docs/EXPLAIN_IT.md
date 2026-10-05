# Explain the project clearly

## Your one-sentence explanation

"I study whether solving an image-deblurring inverse problem improves object detection under simulated camera motion, using the same detector before and after restoration."

## A concrete example

A mobile robot is looking for a cup. During exposure, the camera moves, spreading the cup's edges over several pixels. A detector may then miss the cup. We ask whether image restoration can recover useful information before the detector receives the image.

In the first experiment, we control the blur ourselves. We know the original image, so we can measure reconstruction error. Later, the same experiment can use images captured inside a robot simulator. A claim about successful robot navigation would require an additional action-and-feedback experiment.

## The three images to understand

| Symbol | Meaning | Available to the restoration algorithm? |
|---|---|---|
| x | Original clean image | No, except on development data for choosing parameters |
| y | Observed blurry and noisy image | Yes |
| x_hat | Our estimate of the clean image | This is the output |

The forward model is `y = Hx + n`. H is a known blur operator; n is random noise. A point-spread function (PSF) describes where one point's light spreads. We use a line-shaped PSF to approximate uniform camera motion.

The **forward problem** creates y from x. The **inverse problem** estimates x from y. Simply dividing by the blur in the frequency domain is unstable where the blur removes information: it can amplify noise. Regularization prevents this unstable amplification by expressing a preference for plausible images.

## Method 1: the fast baseline

The Wiener-form filter applies:

`X_hat = conjugate(H) * Y / (abs(H)^2 + alpha)`

Capital letters denote Fourier-domain quantities. Alpha limits amplification where H is small. We tune alpha on development images. Because alpha is a constant, this is also a Tikhonov solution with an identity regularizer; we are not estimating a full frequency-dependent signal/noise spectrum.

Say: "This baseline is fast and easy to understand, but it may keep noise or introduce ringing."

## Method 2: total variation with ADMM

We solve:

`min_x  0.5 * ||Hx - y||_2^2 + lambda * TV(x)`

The first term is **data fidelity**: after blurring the estimated image, it should explain the observation. The second is a **prior**: large amounts of small, noisy variation are discouraged. TV can preserve strong edges while smoothing small fluctuations, but it can also erase texture or create flat regions.

Lambda balances these two preferences. Small lambda allows more noise and detail; large lambda favors smoother images. Lambda changes the optimization problem. Rho is an ADMM penalty parameter that mainly changes convergence behavior; with a finite iteration budget, it can affect the returned approximation.

ADMM introduces an auxiliary variable `z = Dx`, where D computes neighboring-pixel differences. It repeats three steps:

1. **Update x:** find an image consistent with the observed blur and the current edge estimate. FFTs make this step efficient under periodic boundaries.
2. **Update z:** shrink weak gradients with vector soft thresholding. This applies the TV prior.
3. **Update u:** accumulate disagreement between Dx and z so the next iteration encourages them to agree.

Do not say that ADMM learns a neural network. It is an iterative optimization algorithm. The detector is a separate pretrained neural network and remains frozen.

## What counts as a fair test?

We split images before selecting parameters. Alpha and lambda are selected by mean development PSNR. Those values are then frozen for the test images. For each test image and blur level, every method receives exactly the same noisy observation. The detector, confidence threshold, and IoU threshold also remain fixed.

The clean test image is used to generate the simulated observation and calculate reconstruction metrics. It is never passed into the restoration solver. The known blur kernel is passed to the solver: this assumption must be stated.

## How to read the numbers

| Metric | Plain-language meaning | Key limitation |
|---|---|---|
| PSNR | How small is the pixel reconstruction error? Higher is better. | Does not directly measure usefulness for recognition. |
| SSIM | How similar are local structures? Higher is generally better. | Still an image metric, not an action metric. |
| Recall | Of the labeled target objects, how many were detected correctly? | A low threshold may improve recall while increasing false positives. |
| Precision | Of the reported target detections, how many were correct? | Undefined if the model reports no target detections; shown as N/A. |
| IoU | How much do the predicted and labeled boxes overlap? | A box with the wrong class still fails. |

A correct detection needs the right category and IoU >= 0.5, with confidence >= 0.3. Each ground-truth box can match at most one prediction. Results aggregate counts across images before calculating precision and recall. This is micro averaging. We do not call this mAP.

The clean-image detector can make mistakes. Its result is a reference, not 100% ground truth. Ground truth comes from COCO annotations or simulator instance masks.

## What results can support

If restoration improves both PSNR and recall, that supports the proposed connection under the tested conditions. If PSNR improves but recall does not, that is also a useful finding: pixel similarity and recognition are not equivalent. If restoration hurts recall, inspect ringing, texture loss, small objects, and domain shift rather than hiding the example.

With four quick-test images, one recovered object can change recall substantially. Present counts alongside percentages. Use a larger dataset and repeated noise seeds before drawing broader conclusions. Adjacent robot frames are correlated; split by scene and estimate uncertainty at the scene level.

## The connection to embodied AI

This project develops the visual-perception front end that could support an embodied agent. The COCO pilot establishes that the implementation runs on labeled images. AI2-THOR adds first-person robot views and simulator ground truth. A future navigation extension would use the detector's output to choose actions and measure task success. Only that extension would test the full perception-action loop.

## Read the code in this order

1. `motion_psf` and `degrade` in `robovision/imaging.py`: how the observation is generated.
2. `wiener_restore`: the simple inverse filter.
3. `tv_admm`: the three alternating steps above.
4. `detection_counts` in `robovision/metrics.py`: what a correct detection means.
5. `run` in `robovision/experiment.py`: development tuning followed by held-out evaluation.
