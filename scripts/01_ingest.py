"""Convierte CSV a Parquet por día y mide lecturas por motor.

El enunciado pide convertir con SparkSession. En este Windows Spark no puede escribir sin
winutils.exe, un binario de terceros que no se instaló. La conversión la hace DuckDB y Spark
ejecuta la misma transformación desde CSV hacia el destino `noop`, para medir su costo.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import duckdb
import pandas as pd
import polars as pl

from ctr.config import CAT_COLS, DATA, PARQUET, RAW, RES, SEED
from ctr.monitor import Peak, append_jsonl
from ctr.sparkutil import get_spark, read_avazu

OUT = RES / "bench" / "ingest.jsonl"
(RES / "eda").mkdir(exist_ok=True, parents=True)
csvs = [str(RAW / f).replace("\\", "/") for f in ("train.csv", "valid.csv", "test.csv")]
gl = str(PARQUET / "**" / "*.parquet").replace("\\", "/")

# 1. CSV -> Parquet particionado por día (DuckDB); se omite si ya existe
if not (PARQUET / "day=30").exists():
    if PARQUET.exists():
        shutil.rmtree(PARQUET)
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"set memory_limit='14GB'; set threads=8; set temp_directory='{str(DATA / 'duck_tmp').replace(chr(92), '/')}'")
    cats = ", ".join(CAT_COLS)
    with Peak() as m:
        con.execute(f"""
        copy (
          select id, click::int as click, {cats},
                 substr(hour,5,2)::int as day,
                 substr(hour,7,2)::int as hod,
                 dayofweek(make_date(2014,10,substr(hour,5,2)::int))::int as dow,
                 (hash(id || '{SEED}') % 1000003) / 1000003.0 as sample_key
          from read_csv({csvs}, all_varchar=true, header=true)
          order by day, hod
        ) to '{str(PARQUET).replace(chr(92), '/')}'
        (format parquet, compression zstd, partition_by (day), row_group_size 500000)
        """)
    append_jsonl(OUT, dict(step="csv_to_parquet", engine="duckdb", n=40428967, sec=m.sec, gb=m.gb))

    src = f"read_parquet('{gl}', hive_partitioning=true)"
    by = con.execute(f"select day, count(*) n, count(distinct id) ids from {src} group by 1 order by 1").df()
    print(by.to_string())
    assert by.n.sum() == 40428967, by.n.sum()
    by.to_csv(RES / "eda" / "filas_por_dia.csv", index=False)
    sz = sum(p.stat().st_size for p in PARQUET.rglob("*.parquet"))
    print("Parquet GB", sz / 1024**3)
    append_jsonl(OUT, dict(step="parquet_size", engine="disk", n=40428967, gb=sz / 1024**3, sec=0))
    con.close()

# 2. Spark: lectura del CSV, misma transformación hacia noop, y lectura del Parquet
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

s = get_spark(cores=8, app="ingest")
schema = StructType([StructField("id", StringType()), StructField("click", IntegerType()),
                     StructField("hour", StringType())] +
                    [StructField(c, StringType()) for c in CAT_COLS])
df = s.read.csv(csvs, header=True, schema=schema)
with Peak() as m:
    n = df.count()
append_jsonl(OUT, dict(step="read_csv_count", engine="spark", n=n, sec=m.sec, gb=m.gb))
tr = (df.withColumn("day", F.substring("hour", 5, 2).cast("int"))
      .withColumn("hod", F.substring("hour", 7, 2).cast("int"))
      .withColumn("sample_key", F.pmod(F.xxhash64("id", F.lit(SEED)), F.lit(1000003)) / 1000003.0)
      .drop("hour"))
with Peak() as m:
    tr.repartition("day").sortWithinPartitions("hod").write.format("noop").mode("overwrite").save()
append_jsonl(OUT, dict(step="csv_transform_noop", engine="spark", n=n, sec=m.sec, gb=m.gb))
q = read_avazu(s)
with Peak() as m:
    n2 = q.count()
    q.groupBy("site_category").agg(F.avg("click")).toPandas()
append_jsonl(OUT, dict(step="agg_parquet", engine="spark", n=n2, sec=m.sec, gb=m.gb))
with Peak() as m:
    df.groupBy("site_category").agg(F.avg("click")).toPandas()
append_jsonl(OUT, dict(step="agg_csv", engine="spark", n=n, sec=m.sec, gb=m.gb))
assert n == n2 == 40428967, (n, n2)
s.stop()

# 3. Mismo agregado (CTR por site_category) en Polars, DuckDB y pandas
for fmt in ("csv", "parquet"):
    with Peak() as m:
        lf = (pl.scan_parquet(gl, hive_partitioning=True) if fmt == "parquet" else
              pl.scan_csv(csvs, schema_overrides={"id": pl.String, "hour": pl.String}))
        lf.group_by("site_category").agg(pl.col("click").mean()).collect(engine="streaming")
    append_jsonl(OUT, dict(step=f"agg_{fmt}", engine="polars", n=n, sec=m.sec, gb=m.gb))
    q2 = f"read_parquet('{gl}', hive_partitioning=true)" if fmt == "parquet" else f"read_csv({csvs}, all_varchar=true, header=true)"
    with Peak() as m:
        duckdb.sql(f"select site_category, avg(click::double) from {q2} group by 1").df()
    append_jsonl(OUT, dict(step=f"agg_{fmt}", engine="duckdb", n=n, sec=m.sec, gb=m.gb))
with Peak() as m:   # pandas como referencia: solo 2 columnas para no agotar la RAM
    pd.read_csv(csvs[0], usecols=["click", "site_category"]).groupby("site_category").click.mean()
append_jsonl(OUT, dict(step="agg_csv_2col_train", engine="pandas", n=32343172, sec=m.sec, gb=m.gb))
print("listo")
