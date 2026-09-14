from __future__ import annotations

import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

PEER_MODULES = {
    "active_learning",
    "deep_learning_baseline",
    "data",
    "figure_generation",
}

ROOT_METADATA = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DATA",
    "requirements.txt",
}

DIRECT_REQUIREMENTS = {
    "matplotlib",
    "numpy",
    "pandas",
    "patsy",
    "pillow",
    "pytest",
    "pytorch-tabnet",
    "scikit-learn",
    "scipy",
    "seaborn",
    "statsmodels",
    "torch",
}

REMOVED_ROOT_DOCUMENTS = {"DATA_AVAILABILITY.md", "CODE_AVAILABILITY.md"}
ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?![nrtbfv0])"
    r"|(?<![A-Za-z0-9_.-])/(?:home|Users|tmp|var|private|mnt)/"
)
BINARY_SUFFIXES = {".mat", ".png"}

DATASETS = {
    "data_Formation.mat",
    "data_HUST.mat",
    "data_ISU_ILCC.mat",
    "data_KIT.mat",
    "data_LSD_primary.mat",
    "data_LSD_second.mat",
    "data_MIT.mat",
    "data_TRI_Tesla.mat",
}

EXCLUDED_PARTS = {
    "__pycache__",
    ".numba_cache",
    "outputs_AL_merged",
    "outputs_DL_benchmark_merged",
    "old_4_dataset_outputs",
    "parallel_job_benchmark_outputs",
}

EXCLUDED_SUFFIXES = {".pyc", ".pt", ".zip", ".pptx", ".pdf", ".svg"}


def test_required_peer_modules_exist() -> None:
    present = {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir()}
    assert PEER_MODULES <= present


def test_required_root_metadata_exists() -> None:
    missing = sorted(name for name in ROOT_METADATA if not (PACKAGE_ROOT / name).is_file())
    assert not missing, f"Missing root metadata: {missing}"


def test_direct_dependencies_are_declared() -> None:
    lines = (PACKAGE_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    declared = {
        re.split(r"[<>=!~]", line.strip(), maxsplit=1)[0].lower()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert declared == DIRECT_REQUIREMENTS


def test_readme_does_not_reference_removed_root_documents() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    stale = sorted(name for name in REMOVED_ROOT_DOCUMENTS if name in readme)
    assert not stale, f"README still references removed documents: {stale}"


def test_release_text_contains_no_absolute_paths() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() in BINARY_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if ABSOLUTE_PATH.search(line):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{line_number}")
    assert not violations, "Absolute paths found:\n" + "\n".join(violations)


def test_exact_dataset_inventory() -> None:
    data_dir = PACKAGE_ROOT / "data"
    present = {path.name for path in data_dir.glob("*.mat")}
    assert present == DATASETS


def test_excluded_artifacts_are_absent() -> None:
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        relative = path.relative_to(PACKAGE_ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            violations.append(str(relative))
        if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES:
            violations.append(str(relative))
    assert not violations, "Excluded release artifacts found:\n" + "\n".join(violations)


def test_mat_files_are_declared_for_git_lfs() -> None:
    attributes = (PACKAGE_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "data/*.mat filter=lfs diff=lfs merge=lfs -text" in attributes
