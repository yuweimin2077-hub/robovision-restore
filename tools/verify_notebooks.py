"""Validate notebooks and execute the main one in an isolated local kernel.

Dependencies must already be installed. The local image cache is copied to
avoid repeating downloads. This validates notebook code, not hosted Colab.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        print(f"Validated structure and syntax: {path.name}", flush=True)
    work = ROOT / "tmp/notebook_verification"
    work.mkdir(parents=True, exist_ok=True)
    kernel_dir = work / "jupyter/kernels/robovision"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "RoboVision test kernel", "language": "python"}
    (kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(work / "jupyter")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "jupyter/runtime")
    os.environ["IPYTHONDIR"] = str(work / "ipython")
    os.environ["ROBOVISION_SKIP_INSTALL"] = "1"
    cache = work / "robovision_colab_workspace/data/cache/images"
    cache.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "data/cache/images").glob("*.jpg"):
        shutil.copy2(path, cache / path.name)
    nb = nbformat.read(ROOT / "notebooks/01_start_here.ipynb", as_version=4)
    def announce(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing notebook cell {cell_index + 1}/{len(nb.cells)}", flush=True)

    print("Starting local notebook kernel...", flush=True)
    client = NotebookClient(nb, timeout=600, startup_timeout=60, kernel_name="robovision",
                            on_cell_start=announce, resources={"metadata": {"path": str(work)}})
    client.execute()
    nbformat.write(nb, work / "01_start_here.executed.ipynb")
    errors = [out for c in nb.cells if c.cell_type == "code" for out in c.outputs if out.output_type == "error"]
    if errors:
        raise RuntimeError(errors)
    print(f"Executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells without errors.", flush=True)
    print("Hosted Colab and AI2-THOR rendering remain separately unverified.", flush=True)


if __name__ == "__main__":
    main()
