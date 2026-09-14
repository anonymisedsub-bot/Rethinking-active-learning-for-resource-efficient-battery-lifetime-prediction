from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PACKAGE_ROOT / "figure_generation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, len(header)


def main() -> None:
    for module in sorted(path for path in FIGURE_ROOT.iterdir() if path.is_dir() and path.name != "common"):
        source_dir = module / "source_data"
        records = []
        for path in sorted(source_dir.glob("*.csv")):
            if path.name == "manifest.csv":
                continue
            n_rows, n_columns = _shape(path)
            records.append(
                {
                    "filename": path.name,
                    "sha256": _sha256(path),
                    "n_rows": n_rows,
                    "n_columns": n_columns,
                    "role": "figure source data",
                }
            )
        with (source_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "sha256", "n_rows", "n_columns", "role"])
            writer.writeheader()
            writer.writerows(records)
        print(f"{module.name}: {len(records)} tables")


if __name__ == "__main__":
    main()

