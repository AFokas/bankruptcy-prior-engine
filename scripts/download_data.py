"""Download and extract the Polish Bankruptcy dataset from UCI.

Idempotent: if all five ARFF files already exist in the raw directory, the
download step is skipped and only the summary is printed.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip"
EXPECTED_FILES = [f"{i}year.arff" for i in range(1, 6)]


def _count_arff_records(path: Path) -> int:
    n = 0
    in_data = False
    with path.open("r", errors="replace") as f:
        for line in f:
            if not in_data:
                if line.strip().lower() == "@data":
                    in_data = True
                continue
            if line.strip() and not line.lstrip().startswith("%"):
                n += 1
    return n


def download_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in EXPECTED_FILES if not (raw_dir / f).exists()]
    if not missing:
        print(f"All {len(EXPECTED_FILES)} ARFF files already present in {raw_dir} — skipping download.")
        return

    print(f"Downloading {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as resp:
        payload = resp.read()
    print(f"  downloaded {len(payload):,} bytes")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in EXPECTED_FILES:
                target = raw_dir / name
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                print(f"  extracted {name}")


def summarise(raw_dir: Path) -> None:
    print(f"\n{'file':<14}{'size (MB)':>12}{'records':>12}")
    print("-" * 38)
    for name in EXPECTED_FILES:
        path = raw_dir / name
        if not path.exists():
            print(f"{name:<14}{'MISSING':>12}{'-':>12}")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        n_records = _count_arff_records(path)
        print(f"{name:<14}{size_mb:>12.2f}{n_records:>12,}")


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    download_dataset(raw_dir)
    summarise(raw_dir)


if __name__ == "__main__":
    main()
