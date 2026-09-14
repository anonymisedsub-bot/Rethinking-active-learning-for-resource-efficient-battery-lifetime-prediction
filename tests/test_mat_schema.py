from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from scipy.io import whosmat


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED = {
    "data_Formation.mat": (182, 100, 21, 17, 21, 4),
    "data_HUST.mat": (77, 100, 100, 17, 21, 4),
    "data_ISU_ILCC.mat": (241, 100, 4, 11, 14, 3),
    "data_KIT.mat": (82, 100, 5, 17, 21, 4),
    "data_LSD_primary.mat": (73, 100, 20, 11, 15, 4),
    "data_LSD_second.mat": (72, 100, 20, 11, 15, 4),
    "data_MIT.mat": (124, 100, 100, 12, 15, 3),
    "data_TRI_Tesla.mat": (134, 100, 3, 18, 24, 6),
}

EXPECTED_SHA256 = {
    "data_Formation.mat": "2fc567529088bbe4eabc0f082a75b9638b1691f9209435a900129a67679fa58d",
    "data_HUST.mat": "b73f3a4d101150bd5bf2db816e727a2ab2557d4a2ed3f009716e4fe7216cb700",
    "data_ISU_ILCC.mat": "6d632a2f7e7fefab706213da16449bbe58aae989935bb3591404573299308bc9",
    "data_KIT.mat": "fbd052b77df56d8370349ce41381ae295343e1e649f057bbc541c426292edace",
    "data_LSD_primary.mat": "0d198540103612a02fd3b8d33b66bc1745757bb5406b0f523b2919d4a2ccb825",
    "data_LSD_second.mat": "f175a6719c1186a5c23f465e78cfe6b5a97b0361a674c18b53e1c927cf9c762b",
    "data_MIT.mat": "3022c33dbac7c6cd8067e7e2143a0358eed259cb3dc244b74643dddf2bb6d64f",
    "data_TRI_Tesla.mat": "5505909a66d89594e9899b3f38d0cc9b05bc65ac5ff043b9f367498ec89e6be8",
}


@pytest.mark.parametrize("filename,dimensions", EXPECTED.items())
def test_processed_mat_schema(filename: str, dimensions: tuple[int, ...]) -> None:
    n, dq, soh, engineered, metadata, protocols = dimensions
    schema = {name: shape for name, shape, _dtype in whosmat(DATA_DIR / filename)}
    expected_shapes = {
        "lifetimes": {(1, n), (n, 1)},
        "dQn_m_matrix": {(n, dq)},
        "early_soh_degradation_matrix": {(n, soh)},
        "features_matrix": {(n, engineered)},
        "features_metadata_matrix": {(n, metadata)},
        "protocols": {(n, protocols), (protocols, n)},
    }
    missing = sorted(set(expected_shapes).difference(schema))
    assert not missing, f"{filename} is missing variables: {missing}"
    for variable, allowed in expected_shapes.items():
        assert schema[variable] in allowed, (filename, variable, schema[variable], allowed)


@pytest.mark.parametrize("filename,expected_digest", EXPECTED_SHA256.items())
def test_processed_mat_sha256(filename: str, expected_digest: str) -> None:
    digest = hashlib.sha256()
    with (DATA_DIR / filename).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == expected_digest
