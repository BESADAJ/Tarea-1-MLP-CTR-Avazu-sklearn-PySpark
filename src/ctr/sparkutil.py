"""Sesión de Spark local para Windows.

Dos particularidades del equipo, documentadas en el anexo de reproducibilidad:
- Sin winutils.exe Spark no puede escribir Parquet en Windows. No se instaló ese binario de
  terceros: el Parquet lo genera DuckDB y Spark solo lee.
- Los Python workers de PySpark 4.0.1 no arrancan en este Windows (fallo al leer pyspark.zip).
  Por eso no se usan UDF de Python: todo el pipeline corre en la JVM.
"""
import os
import sys

from .config import DATA, PARQUET


def get_spark(cores=8, driver_gb=20, app="avazu", event_log=False, extra=None):
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession

    tmp = DATA / "spark_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    b = (SparkSession.builder.master(f"local[{cores}]").appName(app)
         .config("spark.driver.memory", f"{driver_gb}g")
         .config("spark.driver.maxResultSize", "4g")
         .config("spark.local.dir", str(tmp))
         .config("spark.sql.shuffle.partitions", str(max(8, cores * 4)))
         .config("spark.sql.autoBroadcastJoinThreshold", "2g")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.rdd.compress", "true")
         .config("spark.ui.showConsoleProgress", "false"))
    if event_log:
        d = DATA / "spark_events"
        d.mkdir(parents=True, exist_ok=True)
        b = b.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", d.as_uri())
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    s = b.getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    return s


def read_avazu(spark, days=None):
    """Lee el Parquet pasando rutas exactas: listar directorios exige la librería nativa de Hadoop."""
    files = [p for p in sorted(PARQUET.rglob("*.parquet"))
             if days is None or int(p.parent.name.split("=")[1]) in days]
    return spark.read.option("basePath", str(PARQUET)).parquet(*[str(f) for f in files])
