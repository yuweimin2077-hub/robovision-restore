"""Generate upload-and-run notebooks from the canonical project source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, **metadata):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": metadata, "source": source.splitlines(keepends=True),
              "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12]}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def md(source):
    return cell("markdown", source)


def code(source, **metadata):
    return cell("code", source, **metadata)


def snapshot():
    paths = sorted((ROOT / "robovision").glob("*.py"))
    paths += [ROOT / "data/coco_household18.json", ROOT / "README.md", ROOT / "LICENSE", ROOT / "THIRD_PARTY.md"]
    paths += sorted((ROOT / "docs").glob("*.md"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            # Canonical text makes the snapshot stable across Windows/Linux Git
            # checkouts. The data manifest is byte-preserved for provenance.
            content = (path.read_bytes() if path.name == "coco_household18.json"
                       else path.read_text(encoding="utf-8").encode("utf-8"))
            archive.writestr(info, content)
    data = buffer.getvalue()
    return base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest()


INSTALL = '''
import os, sys, subprocess, importlib.util

# Colab already has a compatible torch/torchvision pair. Preserve it.
# ROBOVISION_SKIP_INSTALL is only used by the local notebook verification script.
if os.environ.get("ROBOVISION_SKIP_INSTALL") != "1":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "numpy>=1.26,<3", "Pillow>=10,<13", "scipy>=1.12,<2",
                           "scikit-image>=0.24,<1", "matplotlib>=3.8,<4"])
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
try:
    import torch, torchvision
except (ImportError, RuntimeError) as error:
    raise RuntimeError("Torch/torchvision could not load. In Colab, disconnect and delete the runtime, then rerun from the first cell.") from error
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
print("GPU available:", torch.cuda.is_available())
'''


def bootstrap(encoded, digest):
    return code('''
from pathlib import Path
import base64, hashlib, io, os, sys, zipfile

# This is a generated source snapshot, not downloaded executable code.
# The readable source lives in the project repository and is unpacked below.
PAYLOAD = "''' + encoded + '''"
EXPECTED_SHA256 = "''' + digest + '''"
archive_bytes = base64.b64decode(PAYLOAD)
assert hashlib.sha256(archive_bytes).hexdigest() == EXPECTED_SHA256
PROJECT = Path("/content/robovision_restore") if Path("/content").exists() else Path.cwd() / "robovision_colab_workspace"
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    for member in archive.infolist():
        destination = (PROJECT / member.filename).resolve()
        if not destination.is_relative_to(PROJECT.resolve()):
            raise ValueError("Archive path escapes the project directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member.filename))
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT.resolve()))
print("Project ready:", PROJECT)
''', cellView="form", collapsed=True)


def main_notebook(encoded, digest):
    return [
        md(r'''
        # RoboVision Restore
        ## Does a sharper image help object detection?

        **Start here.** This notebook explains the experiment before running it.
        Choose **Runtime -> Run all**. CPU is sufficient; a GPU is optional.

        You will create blur, recover images with two methods, and test the same
        frozen detector on all conditions. The quick run uses 2 development
        images and 4 held-out images. No model training or API key is needed.

        **Scope:** this first run uses labeled COCO images, not robot recordings.
        The separate `02_robot_views.ipynb` notebook adds simulated robot views.
        Neither notebook evaluates a navigation policy.
        '''),
        md('''
        ## 1. Prepare the environment
        **Why:** install the numerical tools and check the detector libraries.
        The next collapsed cell contains the project source, so this notebook
        can be uploaded directly without a GitHub URL.
        '''), code(INSTALL, tags=["installation"]), bootstrap(encoded, digest),
        md('''
        ## 2. Choose a small experiment
        Keep `QUICK = True` for your first run. Later, set it to `False` to use
        the entire 18-image pilot. This is still too small for broad claims.

        We test 7-pixel and 17-pixel motion at the resized image resolution.
        `sigma = 0.01` is the Gaussian noise standard deviation on a [0, 1] scale.
        The seed fixes the noise, so all methods see the same observations.
        '''),
        code('''
        from pathlib import Path
        import numpy as np
        from IPython.display import display, Image as DisplayImage, Markdown
        from robovision.experiment import Config, run
        from robovision.data import load_manifest, load_record

        QUICK = True
        config = Config(quick=QUICK, output="runs/colab_quick" if QUICK else "runs/colab_full")
        manifest = load_manifest(config.manifest)
        print("Data:", manifest["name"])
        print("Categories:", ", ".join(manifest["categories"].values()))
        print("Quick mode:", QUICK)
        '''),
        md('''
        ## 3. See how the observation is formed
        **Input:** one clean development image.
        **Operation:** a line-shaped motion kernel spreads each pixel; noise is added.
        **Output:** the observation the restoration algorithms must work with.

        The model is $y = Hx + n$. Here $x$ is the clean image, $H$ is known blur,
        $n$ is noise, and $y$ is the observation. This is a controlled approximation
        using circular boundaries and RGB-space Gaussian noise.
        '''),
        code('''
        import matplotlib.pyplot as plt
        from robovision.imaging import motion_psf, degrade

        demo_record = next(r for r in manifest["records"] if r["split"] == "dev")
        clean, truth = load_record(demo_record, manifest, config.cache, config.max_side)
        kernel = motion_psf(length=17, angle=config.angle)
        observed = degrade(clean, kernel, sigma=config.sigma, seed=config.seed)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(clean); axes[0].set_title("Clean development image: x")
        axes[1].imshow(kernel, cmap="magma"); axes[1].set_title("Known motion kernel: H")
        axes[2].imshow(np.clip(observed, 0, 1)); axes[2].set_title("Observation: y = Hx + n")
        for ax in axes: ax.axis("off")
        display(fig); plt.close(fig)
        print("Image ID:", demo_record["id"], "| Kernel sum:", kernel.sum())
        '''),
        md(r'''
        ## 4. Recover the image
        **Wiener-form baseline:** stabilize frequency-domain inversion with a
        constant regularizer. It is fast, but can retain noise or ringing.

        **TV + ADMM:** solve
        $$\hat{x}=\arg\min_x \frac12\|Hx-y\|_2^2+\lambda\operatorname{TV}(x).$$
        The first term explains the observation. TV discourages noisy changes
        while allowing strong edges. ADMM alternates an image update, a gradient
        shrinkage step, and a disagreement update.

        These illustrative parameters are used only on a development image.
        The formal experiment below selects parameters using development data.
        '''),
        code('''
        from robovision.imaging import wiener_restore, tv_admm
        from robovision.metrics import quality
        from robovision.plots import image_comparison, convergence_plot

        fast = wiener_restore(observed, kernel, balance=0.01)
        restored, history = tv_admm(observed, kernel, lam=0.003, rho=config.rho, iterations=config.iterations)
        demo_images = {"clean": clean, "degraded": observed, "wiener": fast, "tv_admm": restored}
        demo_metrics = {name: quality(clean, image, crop=9) for name, image in demo_images.items()}
        Path("runs/tutorial").mkdir(parents=True, exist_ok=True)
        image_comparison(demo_images, demo_metrics, "runs/tutorial/restoration.png")
        convergence_plot(history, "runs/tutorial/convergence.png")
        display(DisplayImage(filename="runs/tutorial/restoration.png"))
        display(DisplayImage(filename="runs/tutorial/convergence.png"))
        '''),
        md('''
        **Pause and explain:** What does the data term enforce? Why can too much
        regularization remove useful detail? What is the difference between the
        original image, the observation, and the restored estimate?

        The objective need not decrease at every ADMM iteration. The residuals
        help check convergence; a fixed iteration count does not prove an exact solution.
        '''),
        md('''
        ## 5. Run the fair comparison
        This cell selects regularization parameters using **development PSNR only**.
        It then freezes the parameters and evaluates held-out test images.

        All four conditions use the same frozen SSDLite detector and thresholds.
        The solver never receives the clean test image, but it does know the blur kernel.
        The clean test image is used to synthesize the observation and score reconstruction.

        **Outputs:** per-image CSV, summary CSV, selected parameters, full prediction
        records, figures, an all-example gallery, and environment details.
        '''), code('''
        results = run(config)
        output = Path(results["output"])
        display(Markdown((output / "RESULTS.md").read_text()))
        '''),
        md('''
        ## 6. Read the evidence
        PSNR and SSIM describe reconstruction. Recall describes how many labeled
        target objects are detected correctly. A correct box needs the right class,
        confidence >= 0.3, and IoU >= 0.5. Duplicate matches do not count twice.

        Recall and precision are calculated from pooled counts; they are **not mAP**.
        Higher PSNR need not imply higher recall. Clean-image detection can also fail.
        '''), code('''
        display(DisplayImage(filename=str(output / "metrics.png")))
        display(DisplayImage(filename=str(output / "detection_comparison.png")))
        '''),
        md('''
        ## 7. Inspect a second case, including failures
        The gallery includes every evaluated image and blur level. Examples are
        ordered by filename, not by improvement. Dashed teal boxes are labels;
        solid orange boxes are predictions.
        '''), code('''
        gallery = sorted((output / "gallery").glob("*_detections.png"))
        EXAMPLE_INDEX = min(1, len(gallery) - 1)  # Change this index to inspect another case.
        if gallery:
            print(gallery[EXAMPLE_INDEX].name)
            display(DisplayImage(filename=str(gallery[EXAMPLE_INDEX])))
        '''),
        md('''
        ## 8. Prepare a defensible interpretation
        Complete these statements using your own output:

        - "At blur length __, TV-ADMM changed PSNR from __ to __ dB."
        - "It recovered __ additional true detections out of __ labeled objects."
        - "The ranking of methods did / did not agree across image and detection metrics."
        - "These results apply to a small controlled pilot with a known blur kernel."

        The COCO experiment is a preliminary perception study. It does not establish
        real-robot performance or navigation success. Use the second notebook for
        robot-view data, and then add more scenes, repeated seeds, and kernel mismatch.

        The project includes `docs/EXPLAIN_IT.md` and `docs/PRESENTATION_GUIDE.md`.
        '''),
        code('''
        display(Markdown(Path("docs/EXPLAIN_IT.md").read_text()))
        '''),
        md('''
        ## 9. Save your work
        Download this ZIP before the Colab runtime expires. It contains the run,
        the annotation manifest, and the presentation notes. Source images retain
        their original licenses, listed in the manifest and third-party notes.
        Use **File -> Download -> Download .ipynb** to keep the notebook with its outputs.
        '''),
        code('''
        import zipfile
        archive_path = Path("robovision_results.zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in (output, Path("docs")):
                for path in folder.rglob("*"):
                    if path.is_file(): archive.write(path, path.as_posix())
            for path in (Path(config.manifest), Path("THIRD_PARTY.md")):
                archive.write(path, path.as_posix())
        print("Saved:", archive_path.resolve())
        try:
            from google.colab import files
        except ImportError:
            print("Local run: open the ZIP at the path above.")
        else:
            files.download(str(archive_path))
        '''),
        md('''
        ## References
        - [COCO dataset](https://cocodataset.org/)
        - [Torchvision SSDLite](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
        - [ADMM: Boyd et al.](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
        - [AI2-THOR](https://github.com/allenai/ai2thor)

        The full source snapshot is generated by `tools/build_notebooks.py`.
        ''')]


def robot_notebook(encoded, digest):
    return [
        md('''
        # RoboVision Restore: simulated robot views
        Run `01_start_here.ipynb` first. This optional notebook collects first-person
        RGB views and true instance boxes in AI2-THOR, then runs the same imaging experiment.

        Select **Runtime -> Change runtime type -> GPU**. AI2-THOR requires graphics
        support in addition to PyTorch CUDA support. Its simulator download is much
        larger than the main notebook's six images. GPU availability is not guaranteed.
        The hosted rendering path is not yet verified by the recorded local CPU run.

        This is a perception dataset collector, not a navigation policy.
        '''), code(INSTALL, tags=["installation"]), bootstrap(encoded, digest),
        md('''
        ## 1. Prepare simulator rendering
        The official headless API uses `CloudRendering`. If the assigned Colab runtime
        cannot provide Vulkan graphics, use the official
        [AI2-THOR Colab setup](https://github.com/allenai/ai2thor-colab) to start an X server
        and set `CLOUD_RENDERING = False` below. The main imaging notebook remains usable.
        '''), code('''
        import platform, shutil, subprocess, sys
        if platform.system() != "Linux" or shutil.which("nvidia-smi") is None:
            raise RuntimeError("This optional notebook requires a Linux GPU runtime. In Colab select a GPU runtime.")
        subprocess.run(["nvidia-smi"], check=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ai2thor==5.0.0"])
        subprocess.check_call(["apt-get", "update", "-qq"])
        subprocess.check_call(["apt-get", "install", "-y", "-qq", "libvulkan1", "vulkan-tools"])
        '''),
        md('''
        ## 2. Collect labeled views with a scene-level split
        Development scenes: FloorPlan1 and FloorPlan5.
        Test scenes: FloorPlan2 and FloorPlan6.
        This prevents adjacent views from the same scene appearing in both splits.

        Labels come from visible instance masks. Mug/Cup are mapped to COCO cup.
        These visible-mask boxes and category conventions differ from COCO annotations.
        The collector samples camera poses and retains views with a target object.
        '''), code('''
        from robovision.ai2thor_data import collect
        CLOUD_RENDERING = True
        manifest_path = collect(per_scene=6, cloud_rendering=CLOUD_RENDERING)
        print("Saved robot-view manifest:", manifest_path)
        '''),
        md('''
        ## 3. Evaluate the same restoration methods
        Use all collected views so both development and test scenes are represented.
        The COCO-pretrained detector may perform poorly in simulation even on clean
        images. Report that clean reference and the domain gap explicitly.
        '''), code('''
        from pathlib import Path
        from IPython.display import display, Markdown, Image as DisplayImage
        from robovision.experiment import Config, run
        result = run(Config(manifest=str(manifest_path), output="runs/robot_views", quick=False))
        output = Path(result["output"])
        display(Markdown((output / "RESULTS.md").read_text()))
        display(DisplayImage(filename=str(output / "metrics.png")))
        display(DisplayImage(filename=str(output / "detection_comparison.png")))
        '''),
        md('''
        ## 4. Export the views and evidence
        This saves the images, labels, run configuration, plots, and notes.
        To support broader conclusions, collect more independent scenes and repeat
        noise seeds. The present scene split is still a small pilot.
        '''), code('''
        import zipfile
        archive_path = Path("robovision_robot_views.zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in (Path("data/ai2thor"), output, Path("docs")):
                for path in folder.rglob("*"):
                    if path.is_file(): archive.write(path, path.as_posix())
            archive.write("THIRD_PARTY.md")
        from google.colab import files
        files.download(str(archive_path))
        ''')]


def write(name, cells):
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": name, "provenance": []}}, "cells": cells}
    path = ROOT / "notebooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Built {path.name}: {len(cells)} cells")


if __name__ == "__main__":
    encoded, digest = snapshot()
    write("01_start_here.ipynb", main_notebook(encoded, digest))
    write("02_robot_views.ipynb", robot_notebook(encoded, digest))
