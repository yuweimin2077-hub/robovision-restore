# Experiment plan

## Research question

Under controlled motion blur, does a regularized restoration front end improve household-object detection, and does the image-quality ranking agree with the detection ranking?

## Minimum complete course experiment

1. Explain and implement the forward model and TV-ADMM objective.
2. Compare clean, degraded, Wiener-form, and TV-ADMM images with one frozen detector.
3. Tune parameters only on development images and evaluate on held-out images.
4. Report PSNR, SSIM, precision, recall, counts, runtime, and a failure gallery.
5. Repeat on simulated robot views with scene-disjoint development/test sets.
6. Add at least three noise seeds and an uncertainty analysis appropriate to the sampling unit.

The implementation contains steps 1-4 and the data-collection path for step 5. Simulator rendering and larger experiments require execution on the intended GPU environment. Step 6 is future work, not a completed result.

## First run versus final evidence

| Stage | Purpose | Data |
|---|---|---|
| Quick tutorial | Learn the pipeline and catch implementation errors | 2 development + 4 test COCO images |
| Full pilot | Inspect more cases and failure modes | 6 development + 12 test COCO images |
| Robot-view study | Evaluate the perception front end in simulation | Multiple disjoint development/test scenes |
| Robustness study | Test how assumptions affect the conclusion | Repeated noise seeds and mismatched blur kernels |

Do not tune choices on held-out results after seeing them and then report those same results as an untouched test. If the project design changes substantially, document the change and use a new evaluation set for confirmatory conclusions.

## Course milestones from the supplied syllabus

| Date | Deliverable | Concrete content |
|---|---|---|
| October 15 | One-page proposal | Question, forward model, methods, data, evaluation, scope |
| November 5 | Up to two-page progress report | Working pipeline, initial figures, remaining issues |
| December 1 / 3 / 8 / 10 | Scheduled presentation sessions | 15-20 minute presentation; use the assigned slot |
| Confirm in course system | Up to four-page final report | Full method, experiments, limitations, references |

The syllabus says dates may change and does not give a final-report deadline. The instructor's LaTeX template is required for submitted reports; it is not yet available in this workspace.

## Useful extensions in priority order

1. **Kernel mismatch:** generate a 17-pixel blur and restore with slightly wrong length or angle. This tests sensitivity to the known-kernel assumption.
2. **Repeated seeds:** hold data and parameters fixed; change the Gaussian noise seed. Resample at image or scene level for uncertainty intervals.
3. **Parameter objective:** compare development-PSNR tuning with development-recall tuning.
4. **Action loop:** let an agent turn or stop based on detections and measure task success. This is a larger embodied extension.

Do not add VLA training or a physical robot until the imaging study is complete and understood.
