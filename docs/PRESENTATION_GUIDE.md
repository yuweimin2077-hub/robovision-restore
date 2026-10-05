# An 18-minute presentation you can explain

Use actual figures and numbers from your latest saved run. The current outputs are a pilot, not a finished final-project result. This guide is an outline and speaking script, not a PowerPoint file.

## Slide 1 - Question and motivation (1 minute)

**Title:** Does image deblurring improve visual perception for robots?

Show a cup image before and after simulated blur.

Say: "A robot's camera can produce blurred images while the robot moves. I investigate whether restoring those images helps a fixed object detector. I evaluate reconstruction and recognition separately because a visually sharper image may not be more useful to the detector."

## Slide 2 - Scope and data (1.5 minutes)

Show the development/test counts and data source from `data_used.json`.

Say: "The initial experiment uses a small, labeled COCO subset to validate the method. An extension collects first-person views in AI2-THOR. The current study evaluates perception, and does not yet claim improvements in robot navigation."

If you have run the simulator extension, show a real exported frame and report the scene split. If you have not, label that extension as planned. Explain that selection requires a sufficiently large cup or bowl, so the subset is intentionally not representative of the full COCO dataset.

## Slide 3 - The forward model (2 minutes)

Show the line-shaped blur kernel and `y = Hx + n`.

Say: "I start with a clean image x. The known operator H spreads pixels along a short motion path. I add Gaussian noise n to obtain the observation y. Because I know x, I can evaluate reconstruction quality. The solver only receives y and H."

State the three simplifying assumptions: known blur, spatially invariant circular convolution, and Gaussian noise in normalized RGB space.

## Slide 4 - Why regularization is needed (1 minute)

Explain unstable inversion with one sentence: "If blur removes a frequency, dividing by its near-zero response also magnifies noise."

Show the Wiener-form denominator `abs(H)^2 + alpha`. Alpha prevents excessive amplification.

## Slide 5 - The optimization problem (2 minutes)

Show `0.5 ||Hx-y||^2 + lambda TV(x)` and label the two terms.

Say: "The data term requires the estimate to explain the observed image. TV encourages a compact set of meaningful edges instead of many noisy fluctuations. Lambda controls the tradeoff. Strong smoothing can remove useful object details, which is one reason we also evaluate detection."

## Slide 6 - ADMM in three steps (2 minutes)

Use three boxes: update image -> shrink gradients -> update disagreement.

Say: "I introduce z = Dx so the image update and the TV prior can be handled separately. The image update uses FFTs. The gradient update uses soft thresholding. The dual update encourages agreement. The convergence figure checks whether the optimization is settling down."

Show `convergence.png`. Do not claim the objective must decrease every iteration or that a fixed iteration count proves exact convergence.

## Slide 7 - Experimental controls (1.5 minutes)

Show a table with fixed detector, fixed thresholds, development-only tuning, identical noise, and held-out test images.

Say: "I choose hyperparameters on the development split by PSNR, then freeze them. All restoration methods see the same corruption. Ground-truth boxes come from dataset annotations, not from the clean detector's predictions."

## Slide 8 - Qualitative results (2 minutes)

Show `comparison.png` and `detection_comparison.png`. In detection figures, dashed teal boxes are ground truth and solid orange boxes are predictions.

Walk through one object from left to right. State both a success and a failure visible in the all-example gallery. The default example follows manifest order and may be a failure; do not imply it is the best case.

## Slide 9 - Quantitative results (2 minutes)

Show `metrics.png` and the relevant rows from `summary.csv`.

Say: "The left plot measures pixel recovery. The right plot measures target-object recall. I compare both because their rankings need not agree. Recall uses fixed confidence and IoU thresholds and is not mAP."

Report the number of images and objects, the noise level, and the blur lengths. Use your measured values. Do not extrapolate the pilot percentages to all scenes or all robots.

## Slide 10 - Limitations and next experiment (2 minutes)

State known-kernel blur, simplified camera noise, small sample size, circular boundaries, and the COCO-to-robot domain gap. If using AI2-THOR, also mention visible-mask box conventions and category mapping.

Say: "The next experiment tests the same pipeline on scene-disjoint robot views, with repeated noise seeds and a deliberately incorrect blur kernel. This will show whether the benefit survives more realistic uncertainty."

## Slide 11 - Takeaway (1 minute)

Complete this sentence with your actual result:

"Under [specified conditions], [method] changed image quality by [measured amount] and object recall by [measured amount]. The result suggests [limited interpretation], while [specific uncertainty] remains."

Avoid saying "our method improves robots" without an action-level evaluation.

## Likely questions and answers

**Why is this computational imaging?**
We explicitly model image formation, solve an ill-posed inverse problem using regularization and ADMM, and evaluate reconstruction. Detection is a downstream application.

**Why not just retrain the detector on blurred images?**
That is a valid alternative and a future baseline. This experiment isolates what a restoration front end contributes while keeping the detector fixed.

**Does the method know the clean test image?**
The solver does not. The evaluation pipeline uses the clean image to synthesize corruption and measure reconstruction error. The solver does know the blur kernel.

**Why choose lambda with PSNR rather than recall?**
It tests whether an imaging objective transfers to recognition. A useful follow-up is to compare development-PSNR selection against development-recall selection, keeping the test set untouched.

**Is the first method exactly Wiener filtering?**
It uses the Wiener filter's algebraic form with a constant regularizer. It is also an identity-Tikhonov inverse filter. We do not estimate frequency-dependent signal/noise power spectra.

**Is this blind deblurring?**
No. The blur kernel is known. Kernel estimation or kernel mismatch is a follow-up experiment.

**What did you implement yourself?**
The forward model, FFT-based restoration, TV-ADMM solver, development/test protocol, matching metrics, and result export. The detector, datasets, simulator, and standard metric libraries are reused and credited. Disclose coding assistance according to the course's actual policy.

**What makes the evaluation embodied?**
The simulator extension evaluates a robot-view perception module. The COCO pilot is a controlled visual experiment. Full embodied evaluation would require decisions, actions, and task-success metrics.

**What if a sharper image reduces recall?**
That is evidence of a mismatch between reconstruction quality and recognition. Inspect restoration artifacts and object scale, then test the explanation with controlled comparisons.
