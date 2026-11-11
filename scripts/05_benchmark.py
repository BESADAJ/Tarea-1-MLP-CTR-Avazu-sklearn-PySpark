"""Un punto de benchmark, en proceso nuevo para medir memoria pico limpia.

Uso: python 05_benchmark.py <sk|spark> <lr|mlp|hgb> <n|full> <nucleos> <exp>
Mide por separado lectura, preprocesamiento, entrenamiento y predicción. Debe correr sin otras
cargas en el equipo. Ver run_bench.sh para la secuencia completa.

Carga de trabajo comparable y fija entre motores:
  lr   hashing 2^12, 20 pasadas (SGDClassifier frente a LogisticRegression con l-bfgs)
  mlp  frecuencia (d=27), una capa de 50 (MLPClassifier Adam, 10 épocas, frente a l-bfgs, 20 iter.)
  hgb  solo scikit-learn, para ver el paralelismo de un modelo que sí usa varios núcleos
"""
import os
import sys

eng, model, n_arg, cores, exp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
th = os.environ.get("BENCH_THREADS", str(cores if model == "hgb" else 1))  # BENCH_THREADS: control de hilos BLAS
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = th
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from ctr.config import RES
from ctr.metrics import scores
from ctr.monitor import Peak, append_jsonl

warnings.filterwarnings("ignore")
OUT = RES / "bench" / "points.jsonl"
n = None if n_arg == "full" else int(n_arg)
VF = 0.1


def sk():
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier

    from ctr.data import load
    from ctr.encoders import Base, make_encoder
    with Peak() as pk:
        t = time.perf_counter()
        tr = load("train", n)
        va = load("valid", extra=["sample_key"]).filter(pl.col("sample_key") < VF)
        t_read = time.perf_counter() - t
        t = time.perf_counter()
        E = make_encoder("hash12" if model == "lr" else "freq", Base().fit(tr))
        X, Xv = E.fit_transform(tr), E.transform(va)
        y, yv = tr["click"].to_numpy(), va["click"].to_numpy()
        t_prep = time.perf_counter() - t
        m = {"lr": SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=20, tol=None, random_state=42),
             "mlp": MLPClassifier((50,), alpha=1e-3, batch_size=1024, max_iter=10, tol=0, n_iter_no_change=99, random_state=42),
             "hgb": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=63, early_stopping=False, random_state=42)}[model]
        t = time.perf_counter()
        m.fit(X, y)
        t_fit = time.perf_counter() - t
        t = time.perf_counter()
        p = m.predict_proba(Xv)[:, 1]
        t_pred = time.perf_counter() - t
    return X.shape[0], t_read, t_prep, t_fit, t_pred, pk.gb, scores(yv, p)["auc"], len(yv)


def ui_dump(s):
    """Guarda lo que muestra Spark UI desde su API REST (el registro de eventos exige winutils en Windows)."""
    import json
    import urllib.request
    base, app = s.sparkContext.uiWebUrl, s.sparkContext.applicationId
    d = RES / "bench" / "spark_ui"; d.mkdir(parents=True, exist_ok=True)
    for name, q in [("jobs", "jobs"), ("stages", "stages?details=false"), ("executors", "allexecutors"),
                    ("environment", "environment"), ("sql", "sql?details=false")]:
        try:
            with urllib.request.urlopen(f"{base}/api/v1/applications/{app}/{q}", timeout=60) as r:
                (d / f"{name}.json").write_bytes(r.read())
        except Exception as e:
            print("ui_dump", name, e, flush=True)


def spark():
    from pyspark import StorageLevel
    from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql import functions as F

    from ctr.sparkpipe import Prep, freq_pipeline, hash_pipeline, splits
    from ctr.sparkutil import get_spark
    with Peak() as pk:
        s = get_spark(cores=cores, driver_gb=14, app=f"b_{model}_{n_arg}_{cores}", event_log=False)
        t = time.perf_counter()
        tr, va = splits(s, n)
        nr = tr.count()
        va = va.filter(F.col("sample_key") < VF)
        nv = va.count()
        t_read = time.perf_counter() - t
        t = time.perf_counter()
        P = Prep().fit(tr)
        a, b = P.transform(tr), P.transform(va)
        pm = (hash_pipeline(12) if model == "lr" else freq_pipeline()).fit(a)
        ft = pm.transform(a).select("features", "label").repartition(max(8, cores * 2)).persist(StorageLevel.MEMORY_AND_DISK)
        ft.count()
        t_prep = time.perf_counter() - t
        d = len(ft.first().features)
        est = (LogisticRegression(maxIter=20, regParam=1e-6, standardization=False, tol=0.0) if model == "lr" else
               MultilayerPerceptronClassifier(layers=[d, 50, 2], maxIter=20, seed=42, tol=0.0))
        t = time.perf_counter()
        md = est.fit(ft)
        t_fit = time.perf_counter() - t
        t = time.perf_counter()
        pv = md.transform(pm.transform(b).select("features", "label")).select("rawPrediction", "label").persist(StorageLevel.DISK_ONLY)
        pv.count()
        t_pred = time.perf_counter() - t
        auc = BinaryClassificationEvaluator(labelCol="label").evaluate(pv)
        if exp == "spark_ui":
            ui_dump(s)
        s.stop()
    return nr, t_read, t_prep, t_fit, t_pred, pk.gb, auc, nv


nr, r, pr, f, pd_, gb, auc, nv = (sk if eng == "sk" else spark)()
append_jsonl(OUT, dict(engine=eng, model=model, n=nr, cores=cores, exp=exp, read=r, prep=pr, fit=f, pred=pd_, gb=gb, auc=auc,
                       rows_s_fit=nr / f, rows_s_pred=nv / pd_))
print(eng, model, nr, cores, round(r, 1), round(pr, 1), round(f, 1), round(pd_, 1), round(gb, 1), round(auc, 4), flush=True)
