"""Carga perezosa del Parquet con Polars, muestras anidadas y candado sobre la prueba."""
import os

import polars as pl

from .config import ALL_CATS, PARQUET, TEST_DAY, TRAIN_DAYS, VALID_DAY

N_TRAIN = 32_377_421  # filas de los días 21 a 28 (results/eda/filas_por_dia.csv)
GLOB = str(PARQUET / "**" / "*.parquet").replace("\\", "/")


def _scan():
    return pl.scan_parquet(GLOB, hive_partitioning=True)


def load(split, n=None, extra=()):
    """Carga train, valid o test. Con `n`, muestra por sample_key: anidada entre tamaños."""
    if split == "test" and os.environ.get("CTR_ALLOW_TEST") != "1":
        raise RuntimeError("La prueba está bloqueada: solo la abre scripts/07_final_test.py")
    days = {"train": TRAIN_DAYS, "valid": [VALID_DAY], "test": [TEST_DAY]}[split]
    lf = _scan().filter(pl.col("day").is_in(days))
    if n is not None and n < N_TRAIN:
        lf = lf.filter(pl.col("sample_key") < n / N_TRAIN)
    cols = ["click", "day"] + ALL_CATS + list(extra)
    return lf.select(cols).collect(engine="streaming")
