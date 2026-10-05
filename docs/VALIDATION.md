# Validation record

The core numerical implementation, the detection pipeline, and the main notebook have been checked locally. Recorded results and exact library versions are included in `examples/verified_run/`.

## Scientific checks

- The finite-difference adjoint satisfies the inner-product identity needed by ADMM.
- A point impulse verifies PSF alignment, and a constant image verifies flux preservation.
- Noise is deterministic for a fixed seed and remains unclipped, matching the stated forward model.
- TV-ADMM improves a synthetic piecewise-constant reconstruction and reduces the splitting residual.
- The Wiener-form identity-kernel case agrees with its analytical solution.
- Detection matching handles duplicate predictions, wrong classes, poor overlap, and empty predictions.
- Development/test group overlap is rejected.
- Bounding boxes follow the actual resized integer image dimensions.
- Simulator annotations use instance ground truth and explicit category mapping.

These checks catch implementation errors; they do not prove benefits on real robot data.

## End-to-end checks

The main pipeline downloaded real COCO images and official SSDLite weights, selected parameters on development images, evaluated held-out images, and exported figures and metric tables. The main notebook executed all 10 code cells successfully in a local Python 3.12 Jupyter kernel on Windows CPU, with dependencies preinstalled. The generated results archive and notebook output were inspected. The setup cell's package installation was skipped in this local verification because the dependencies had already been installed.

All 11 scientific unit tests passed. Both distributed notebooks passed notebook-schema and Python-syntax checks. A GitHub Actions workflow runs the unit tests and checks that the committed notebooks match their source generator.

The full 18-image pilot also completed: 6 development images, 12 held-out images, and two blur levels. Its actual results, image IDs, model predictions, and library versions are saved under `examples/verified_run/full/`.

The optional robot notebook is checked for notebook format and Python syntax. Its label conversion is unit tested. Full AI2-THOR rendering is not verified on this Windows CPU environment. A hosted Google Colab session has not been executed by this local verification. The tutorial is self-contained and uses Colab-compatible Python, but that distinction remains important.

## Reproduce validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/build_notebooks.py
python tools/verify_notebooks.py
```

The final command uses an existing local image cache where available and requires network access for missing images or weights. It saves an executed notebook under `tmp/notebook_verification/` and does not alter the distributable notebooks.
