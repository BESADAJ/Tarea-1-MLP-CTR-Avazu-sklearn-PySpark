"""Experimentos de PySpark. Uso: python 04_spark_experiments.py <etapa> [opciones].

Etapas:
  point  uno o varios ajustes con los mismos datos; ver argumentos en point()
  grid   rejilla con ParamGridBuilder, evaluada en el día 29

La validación es temporal: TrainValidationSplit y CrossValidator parten al azar, así que la
rejilla se recorre a mano con los mismos ParamMap y se evalúa en el día 29.
Sin escritura a disco desde Spark: las predicciones vuelven a Python por Arrow.
"""
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pyspark import StorageLevel
from pyspark.ml.classification import GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import functions as F

from ctr.config import DATA, RES, SEEDS, TEST_DAY
from ctr.metrics import best_threshold, scores, threshold_scores
from ctr.monitor import Peak, append_jsonl
from ctr.sparkpipe import Prep, freq_pipeline, gbt_pipeline, hash_pipeline, splits
from ctr.sparkutil import get_spark, read_avazu

F_DIR = DATA / "final"
RUNS = RES / "spark" / "runs.jsonl"


def session(cores=8, gb=14, app="exp", event_log=False):
    return get_spark(cores=cores, driver_gb=gb, app=app, event_log=event_log,
                     extra={"spark.sql.execution.arrow.pyspark.enabled": "true"})


def features(s, n, enc, valid_frac=None, level=StorageLevel.MEMORY_AND_DISK, with_test=False):
    """Devuelve (train, valid, test o None). `enc`: hashN, freq o gbt."""
    tr, va = splits(s, n)
    if valid_frac:
        va = va.filter(F.col("sample_key") < valid_frac)
    P = Prep().fit(tr)
    a, b = P.transform(tr), P.transform(va)
    pipe = {"freq": freq_pipeline, "gbt": gbt_pipeline}.get(enc, None)
    pipe = pipe() if pipe else hash_pipeline(int(enc[4:]))
    pm = pipe.fit(a)
    ft = pm.transform(a).select("features", "label").repartition(16).persist(level)
    fte = None
    if with_test:
        fte = pm.transform(P.transform(read_avazu(s, [TEST_DAY]))).select("id", "features", "label")
    return ft, pm.transform(b).select("id", "features", "label"), fte


def collect_probs(pred):
    d = pred.select("id", vector_to_array("probability")[1].alias("p"), "label").toPandas()
    return d.id.values.astype(str), d.label.values.astype(int), d.p.values


def evaluate(model, fv):
    i, y, p = collect_probs(model.transform(fv))
    thr = best_threshold(y, p)
    return {**{f"v_{k}": v for k, v in scores(y, p).items()},
            **{f"v_{k}": v for k, v in threshold_scores(y, p, thr).items()}}, (i, y, p)


class Watchdog(threading.Thread):
    """Cancela los trabajos si se pasa del tiempo o si el disco baja de 3 GB libres."""

    def __init__(self, sc, limit):
        super().__init__(daemon=True)
        self.sc, self.limit, self.t0, self.why, self.go = sc, limit, time.time(), None, True

    def run(self):
        while self.go:
            time.sleep(5)
            if time.time() - self.t0 > self.limit:
                self.why = "tiempo"
            elif shutil.disk_usage("C:/").free / 1024**3 < 3:
                self.why = "disco"
            if self.why:
                self.sc.cancelAllJobs()
                return


def estimator(model, d, layers, max_iter, solver, step, seed, block):
    if model == "mlp":
        return MultilayerPerceptronClassifier(layers=[d] + layers + [2], maxIter=max_iter, solver=solver, stepSize=step,
                                              seed=seed, blockSize=block, labelCol="label", featuresCol="features")
    if model == "lr":
        return LogisticRegression(maxIter=max_iter, regParam=1e-6, elasticNetParam=0.0, standardization=False,
                                  labelCol="label", featuresCol="features")
    if model == "gbt":
        return GBTClassifier(maxIter=max_iter, maxDepth=6, maxBins=64, stepSize=0.1, seed=seed, labelCol="label",
                             featuresCol="features", subsamplingRate=0.5)
    raise ValueError(model)


def point(a):
    """n enc model layers maxIter solver step seeds [valid_frac] [block] [exp] [limite_seg] [save]"""
    n = None if a[0] == "full" else int(a[0])
    enc, model = a[1], a[2]
    layers = [int(x) for x in a[3].split("x")] if a[3] != "-" else []
    mi, solver, step = int(a[4]), a[5], float(a[6])
    seeds = SEEDS[:int(a[7])]
    vf = float(a[8]) if len(a) > 8 and a[8] != "-" else None
    block = int(a[9]) if len(a) > 9 else 128
    exp = a[10] if len(a) > 10 else "scale"
    limit = int(a[11]) if len(a) > 11 else 86400
    save = len(a) > 12 and a[12] == "save"
    s = session(app=f"{model}_{enc}_{a[0]}")
    t0 = time.perf_counter()
    ft, fv, fte = features(s, n, enc, vf, with_test=save)
    nr = ft.count()
    prep = time.perf_counter() - t0
    d = len(ft.first().features)
    print(f"{model} {enc} n={nr} d={d} prep={prep:.0f}s", flush=True)
    base = dict(exp=exp, model=model, enc=enc, n=nr, layers=layers, maxIter=mi, solver=solver, step=step, d=d, block=block,
                dense_gb=nr * (d + 2) * 8 / 1024**3, prep_sec=prep, valid_frac=vf)
    for sd in seeds:
        est = estimator(model, d, layers, mi, solver, step, sd, block)
        w = Watchdog(s.sparkContext, limit)
        t1 = time.perf_counter()
        w.start()
        try:
            with Peak() as m:
                md = est.fit(ft)
            t2 = time.perf_counter()
            r, (i, y, p) = evaluate(md, fv)
            row = dict(base, seed=sd, status="ok", fit_sec=m.sec, pred_sec=time.perf_counter() - t2, gb=m.gb, **r)
            if save:
                F_DIR.mkdir(exist_ok=True, parents=True)
                name = f"spark_{model}_{enc}_s{sd}"
                np.savez_compressed(F_DIR / f"{name}_valid.npz", id=i, y=y.astype(np.int8), p=p.astype(np.float32))
                i2, y2, p2 = collect_probs(md.transform(fte))
                np.savez_compressed(F_DIR / f"{name}_test.npz", id=i2, y=y2.astype(np.int8), p=p2.astype(np.float32))
        except Exception as e:
            row = dict(base, seed=sd, status=f"fallo:{w.why or type(e).__name__}", fit_sec=time.perf_counter() - t1,
                       error=str(e)[:300])
        w.go = False
        append_jsonl(RUNS, row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
               if k in ("seed", "status", "fit_sec", "gb", "v_auc", "v_logloss")}, flush=True)
        if row["status"] != "ok":
            break
    s.stop()


def grid(a):
    """n enc valid_frac tag. tag=completa: rejilla del enunciado; tag=lbfgs: 4 capas con l-bfgs."""
    n, enc, vf, tag = int(a[0]), a[1], float(a[2]), a[3]
    s = session(app="grid")
    ft, fv, _ = features(s, n, enc, vf)
    ft.count()
    nr = ft.count()
    d = len(ft.first().features)
    fv = fv.persist(StorageLevel.DISK_ONLY)
    fv.count()
    base = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", seed=42, blockSize=128, layers=[d, 10, 2])
    g = ParamGridBuilder().addGrid(base.layers, [[d, 10, 2], [d, 50, 2], [d, 100, 2], [d, 128, 64, 2]])
    if tag == "completa":
        g = g.addGrid(base.stepSize, [0.1, 0.01]).addGrid(base.maxIter, [100, 200]).addGrid(base.solver, ["l-bfgs", "gd"])
    else:
        g = g.addGrid(base.stepSize, [0.03]).addGrid(base.maxIter, [50]).addGrid(base.solver, ["l-bfgs"])
    ev = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    for k, pm in enumerate(g.build()):
        p = {x.name: v for x, v in pm.items()}
        with Peak() as m:
            md = base.fit(ft, pm)
        auc = ev.evaluate(md.transform(fv))
        from ctr.monitor import append_jsonl as aj
        aj(RES / "spark" / "grid.jsonl", dict(enc=enc, n=nr, d=d, tag=tag, auc=auc, sec=m.sec, gb=m.gb, **p))
        print(k, p, round(auc, 4), round(m.sec), flush=True)
    s.stop()


if __name__ == "__main__":
    {"point": point, "grid": grid}[sys.argv[1]](sys.argv[2:])
