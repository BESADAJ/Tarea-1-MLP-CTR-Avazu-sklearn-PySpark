"""Constantes compartidas: rutas, semillas, partición temporal y columnas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PARQUET = DATA / "parquet" / "avazu"          # particionado por day=21..30
RES = ROOT / "results"
FIG = ROOT / "figures"

SEED = 42
SEEDS = [42, 43, 44, 45, 46]                  # el enunciado pide al menos 5 repeticiones

TRAIN_DAYS = list(range(21, 29))              # 21..28: pasado
VALID_DAY = 29                                # elegir hiperparámetros y umbrales
TEST_DAY = 30                                 # futuro: se usa una sola vez

RARE = "__RARE__"
MIN_COUNT = 10                                # umbral de categorías raras (solo con entrenamiento)

CAT_COLS = ["C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain",
            "app_category", "device_id", "device_ip", "device_model", "device_type",
            "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
TIME_CATS = ["hod", "dow"]                    # hora del día y día de la semana (derivadas de hour)
ALL_CATS = CAT_COLS + TIME_CATS
LOW_CARD = ["C1", "banner_pos", "device_type", "device_conn_type", "site_category", "app_category"]
CROSSES = [("site_id", "device_type"), ("app_id", "device_type")]
