"""Evaluación final en la prueba (día 30). Es el único script que calcula métricas sobre ese día.

Lee las predicciones de scikit-learn (reconstruidas aquí con los modelos guardados) y las de Spark
(guardadas por 04_spark_experiments.py con save_test, sin mirarlas). El umbral sale de validación.
"""
import json
import os
import sys
import warnings

os.environ["CTR_ALLOW_TEST"] = "1"
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ctr.config import DATA, RES, SEEDS
from ctr.data import load
from ctr.metrics import (best_threshold, bootstrap_ci, calibration, delong, holm, lift_gain, mcnemar, scores,
                         t_ci, threshold_scores)

warnings.filterwarnings("ignore")
F = DATA / "final"
F.mkdir(exist_ok=True)
OUT = RES / "stats"
b = json.load(open(RES / "sk" / "best_params.json"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000

te = load("test", extra=["id"]).sort("id")
ids = te["id"].to_numpy()
y = te["click"].to_numpy().astype(int)
va = load("valid", extra=["id"]).sort("id")
yv = va["click"].to_numpy().astype(int)
print("CTR prueba", y.mean(), len(y), flush=True)

# Predicciones por modelo y semilla: {nombre: {semilla: (p_valid, p_test)}}
preds = {}
for key, model, enc in (("sklearn_mlp", "mlp", b["enc"]), ("sklearn_sgd", "sgd", b["enc"]), ("sklearn_hgb", "hgb", "target")):
    E = joblib.load(DATA / "cache" / f"encoder_{enc}_{n}_base.joblib")
    Xv, Xt = E.transform(va), E.transform(te)
    preds[key] = {}
    for s in SEEDS:
        m = joblib.load(DATA / "models" / f"{model}_{enc}_{n}_{s}.joblib")
        preds[key][s] = (m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1])
    print(key, flush=True)
    del Xv, Xt

for f in sorted(F.glob("spark_*_test.npz")):
    d = np.load(f, allow_pickle=True)
    k, s = f.stem.replace("_test", "").rsplit("_s", 1)
    v = np.load(str(f).replace("_test", "_valid"), allow_pickle=True)
    ot, ov = np.argsort(d["id"]), np.argsort(v["id"])
    assert (d["id"][ot] == ids.astype(d["id"].dtype)).all() and (d["y"][ot] == y).all()
    preds.setdefault(k, {})[int(s)] = (v["p"][ov], d["p"][ot])

rows, curves = [], {}
for k, ps in preds.items():
    for s, (pv, pt) in ps.items():
        thr = best_threshold(yv, pv)
        rows.append(dict(modelo=k, seed=s, **scores(y, pt), **threshold_scores(y, pt, thr)))
    print(k, rows[-1]["auc"], flush=True)
r = pd.DataFrame(rows)
r.to_csv(OUT / "test_por_semilla.csv", index=False)

mets = ["auc", "pr_auc", "logloss", "brier", "ece", "accuracy", "precision", "recall", "f1"]
res = []
for k, g in r.groupby("modelo"):
    for me in mets:
        mu, sd, lo, hi = t_ci(g[me].values)
        res.append(dict(modelo=k, metrica=me, media=mu, sd=sd, lo=lo, hi=hi, semillas=len(g)))
pd.DataFrame(res).to_csv(OUT / "test_resumen.csv", index=False)

# Curvas y comparaciones con la semilla de referencia (la primera de cada modelo)
ref = {k: ps[min(ps)][1] for k, ps in preds.items()}
refv = {k: ps[min(ps)][0] for k, ps in preds.items()}
thr = {k: best_threshold(yv, p) for k, p in refv.items()}
from sklearn.metrics import precision_recall_curve, roc_curve
cur = {}
for k, p in ref.items():
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=True)
    pr, rc, _ = precision_recall_curve(y, p)
    i, j = np.linspace(0, len(fpr) - 1, 300).astype(int), np.linspace(0, len(pr) - 1, 300).astype(int)
    a, c, m = calibration(y, p)
    fr, g, l = lift_gain(y, p)
    cur[k] = dict(fpr=fpr[i].tolist(), tpr=tpr[i].tolist(), pr=pr[j].tolist(), rc=rc[j].tolist(), cal_p=a.tolist(),
                  cal_y=c.tolist(), frac=fr.tolist(), gain=g.tolist(), lift=l.tolist())
json.dump(cur, open(OUT / "curvas.json", "w"))

# Bootstrap de AUC, PR AUC y log loss sobre el modelo de referencia (1 millón de filas por réplica)
bs = []
for k, p in ref.items():
    for name, fn in (("auc", roc_auc_score), ("logloss", log_loss)):
        lo, hi = bootstrap_ci(y, np.clip(p, 1e-7, 1 - 1e-7), fn, B=200, sub=1_000_000)
        bs.append(dict(modelo=k, metrica=name, lo=lo, hi=hi))
pd.DataFrame(bs).to_csv(OUT / "bootstrap.csv", index=False)

# DeLong y McNemar por pares, con corrección de Holm
pairs = list(combinations(ref, 2))
dl, mc = [], []
for a, c in pairs:
    dl.append(dict(a=a, c=c, **delong(y, ref[a], ref[c])))
    mc.append(dict(a=a, c=c, **mcnemar(y, (ref[a] >= thr[a]).astype(int), (ref[c] >= thr[c]).astype(int))))
dl, mc = pd.DataFrame(dl), pd.DataFrame(mc)
dl["p_holm"], mc["p_holm"] = holm(dl.p), holm(mc.p)
dl.to_csv(OUT / "delong.csv", index=False)
mc.to_csv(OUT / "mcnemar.csv", index=False)
print(dl.round(5).to_string())
print(pd.DataFrame(res).query("metrica=='auc'").round(4).to_string())
print("listo")
