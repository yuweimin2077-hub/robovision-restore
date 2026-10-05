"""Build a source ZIP without local credentials, environments, or datasets."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ("README.md", "LICENSE", "THIRD_PARTY.md", "requirements.txt",
         "requirements-dev.txt", ".gitignore", ".gitattributes")
DIRECTORIES = ("robovision", "docs", "tests", "tools", "notebooks", "examples", ".github")


def main():
    paths = [ROOT / name for name in FILES]
    paths.append(ROOT / "data/coco_household18.json")
    for folder in DIRECTORIES:
        directory = ROOT / folder
        if directory.exists():
            paths.extend(path for path in directory.rglob("*")
                         if path.is_file() and "__pycache__" not in path.parts)
    output = ROOT / "dist/robovision-restore.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, "robovision-restore/" + path.relative_to(ROOT).as_posix())
    print(f"Packaged {len(set(paths))} files ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
