"""Download the NASA C-MAPSS turbofan dataset and extract its text files.

Source: PHM Society mirror of the NASA Prognostics Center of Excellence data repository.
Citation: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Usage:
    python scripts/download_data.py                    # download from the mirror
    python scripts/download_data.py --zip-path FILE    # use a zip downloaded manually
    python scripts/download_data.py --force            # overwrite existing files
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from industrial_ai.config.paths import RAW_CMAPSS_DIR
from industrial_ai.ml.data import SUBSETS

MIRROR_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
EXPECTED_FILES = frozenset(
    f"{kind}_{subset}.txt" for subset in SUBSETS for kind in ("train", "test", "RUL")
)


def download(url: str, target: Path, timeout: int = 120) -> None:
    """Stream a URL to a local file."""
    request = urllib.request.Request(url, headers={"User-Agent": "industrial-ai-copilot"})
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def extract_expected(archive: zipfile.ZipFile, wanted: set[str], destination: Path) -> set[str]:
    """Extract the wanted files from an archive, searching nested zips as well.

    Files are written by base name only, so paths inside the archive can never
    escape the destination folder (protection against "zip slip").
    """
    found: set[str] = set()
    for info in archive.infolist():
        name = Path(info.filename).name
        if name in wanted and name not in found:
            (destination / name).write_bytes(archive.read(info))
            found.add(name)
        elif info.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(info))) as inner:
                found |= extract_expected(inner, wanted - found, destination)
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(destination: Path, source: str) -> Path:
    """Record a checksum and line count per file, to detect silent data changes later."""
    files = {}
    for name in sorted(EXPECTED_FILES):
        path = destination / name
        with path.open("rb") as fh:
            n_lines = sum(1 for _ in fh)
        files[name] = {"sha256": sha256(path), "lines": n_lines}
    manifest_path = destination / "manifest.json"
    manifest = {"source": source, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip-path", type=Path, help="use a zip file downloaded manually")
    parser.add_argument("--dest", type=Path, default=RAW_CMAPSS_DIR, help="output folder")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination: Path = args.dest
    destination.mkdir(parents=True, exist_ok=True)

    if not args.force and all((destination / name).exists() for name in EXPECTED_FILES):
        print(f"C-MAPSS already present in {destination} (use --force to re-extract).")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        if args.zip_path:
            zip_path, source = args.zip_path, f"local file: {args.zip_path.name}"
        else:
            zip_path, source = Path(tmp) / "cmapss.zip", MIRROR_URL
            print(f"Downloading {MIRROR_URL} ...")
            try:
                download(MIRROR_URL, zip_path)
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"Download failed: {error}")
                print("Download the zip manually and run: --zip-path <file>")
                return 1

        with zipfile.ZipFile(zip_path) as archive:
            found = extract_expected(archive, set(EXPECTED_FILES), destination)

    missing = sorted(EXPECTED_FILES - found)
    if missing:
        print(f"Archive is missing expected files: {missing}")
        return 1

    manifest_path = write_manifest(destination, source)
    print(f"Extracted {len(found)} files to {destination}")
    print(f"Checksums written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
