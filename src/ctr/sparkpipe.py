"""Preprocesamiento en Spark con la misma lógica de scikit-learn. Sin UDF: todo corre en la JVM.

El vocabulario y los conteos salen solo del entrenamiento. Una categoría con menos de MIN_COUNT
apariciones, o no vista, pasa a __RARE__. Las numéricas son las mismas: log1p de actividad de
device_ip y device_id, y hora cíclica.
"""
import math

from pyspark.ml import Pipeline
from pyspark.ml.feature import (FeatureHasher, StandardScaler, StringIndexer, VectorAssembler)
from pyspark.sql import functions as F

from .config import ALL_CATS, LOW_CARD, MIN_COUNT, RARE, TEST_DAY, TRAIN_DAYS, VALID_DAY
from .data import N_TRAIN
from .sparkutil import read_avazu

NUM = ["log_ip", "log_id", "hod_sin", "hod_cos"]


def splits(spark, n=None, test=False):
    """Devuelve entrenamiento (muestra de `n` filas si se pide) y validación o prueba."""
    tr = read_avazu(spark, TRAIN_DAYS)
    if n is not None and n < N_TRAIN:
        tr = tr.filter(F.col("sample_key") < n / N_TRAIN)
    other = read_avazu(spark, [TEST_DAY if test else VALID_DAY])
    return tr, other


class Prep:
    """Aprende conteos en entrenamiento y los aplica con joins de difusión."""

    def __init__(self, cols=ALL_CATS, min_count=MIN_COUNT):
        self.cols, self.min_count = list(cols), min_count

    def fit(self, tr):
        tr = tr.withColumn("hod", F.col("hod").cast("string")).withColumn("dow", F.col("dow").cast("string"))
        self.counts, self.rare_n = {}, {}
        for c in self.cols:
            t = tr.groupBy(c).agg(F.count("*").alias(f"n_{c}")).persist()
            self.rare_n[c] = t.filter(F.col(f"n_{c}") < self.min_count).agg(F.sum(f"n_{c}")).first()[0] or 0
            self.counts[c] = t
        return self

    def transform(self, df):
        d = df.withColumn("hod_i", F.col("hod")).withColumn("hod", F.col("hod").cast("string")) \
              .withColumn("dow", F.col("dow").cast("string"))
        for c in self.cols:
            n = F.col(f"n_{c}")
            d = d.join(F.broadcast(self.counts[c]), on=c, how="left")
            keep = n.isNotNull() & (n >= self.min_count)
            d = d.withColumn(f"{c}_r", F.when(keep, F.col(c)).otherwise(F.lit(RARE))) \
                 .withColumn(f"f_{c}", F.log1p(F.when(keep, n).otherwise(F.lit(self.rare_n[c])).cast("double")))
        ang = 2 * math.pi * F.col("hod_i").cast("double") / 24
        return (d.withColumn("log_ip", F.log1p(F.coalesce(F.col("n_device_ip"), F.lit(0)).cast("double")))
                 .withColumn("log_id", F.log1p(F.coalesce(F.col("n_device_id"), F.lit(0)).cast("double")))
                 .withColumn("hod_sin", F.sin(ang)).withColumn("hod_cos", F.cos(ang))
                 .withColumn("label", F.col("click").cast("double")))


def hash_pipeline(bits, cols=ALL_CATS):
    """Categóricas por FeatureHasher y numéricas escaladas. Vector final disperso."""
    return Pipeline(stages=[
        FeatureHasher(inputCols=[f"{c}_r" for c in cols], categoricalCols=[f"{c}_r" for c in cols],
                      outputCol="h", numFeatures=2 ** bits),
        VectorAssembler(inputCols=NUM, outputCol="num"),
        StandardScaler(inputCol="num", outputCol="num_s", withMean=True, withStd=True),
        VectorAssembler(inputCols=["h", "num_s"], outputCol="features")])


def freq_pipeline(cols=ALL_CATS):
    """Una columna por variable con log1p del conteo. Vector final denso, d = columnas + 4."""
    return Pipeline(stages=[
        VectorAssembler(inputCols=[f"f_{c}" for c in cols] + NUM, outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)])


def gbt_pipeline(cols=ALL_CATS):
    """Para árboles: frecuencia sin escalar y StringIndexer en variables de baja cardinalidad."""
    idx = [StringIndexer(inputCol=f"{c}_r", outputCol=f"i_{c}", handleInvalid="keep") for c in LOW_CARD]
    return Pipeline(stages=idx + [VectorAssembler(
        inputCols=[f"f_{c}" for c in cols if c not in LOW_CARD] + [f"i_{c}" for c in LOW_CARD] + NUM,
        outputCol="features")])
