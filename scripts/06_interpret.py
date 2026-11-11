"""Interpretabilidad del MLP de scikit-learn: LIME, SHAP e importancia por permutación.

Con hashing las columnas no se pueden leer. Por eso los explicadores perturban los códigos de las
23 variables originales, y `predict_codes` hace el camino código -> codificación -> MLP. Esa función
es la correspondencia entre el espacio interpretable y las columnas del modelo.

Los casos salen de validación (día 29): así la prueba sigue usándose una sola vez.
"""
import json
import os
import sys
import warnings

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score

from ctr.config import ALL_CATS, DATA, RES, SEEDS
from ctr.data import load
from ctr.metrics import best_threshold
from ctr.monitor import append_jsonl

warnings.filterwarnings("ignore")
OUT = RES / "interp"
OUT.mkdir(exist_ok=True, parents=True)
b = json.load(open(RES / "sk" / "best_params.json"))
enc, n = b["enc"], int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
E = joblib.load(DATA / "cache" / f"encoder_{enc}_{n}_base.joblib")
M = joblib.load(DATA / "models" / f"mlp_{enc}_{n}_42.joblib")
base = E.b
idx = {c: i for i, c in enumerate(base.cols)}
hod_vals = np.r_[-1, base.vocab["hod"]["v"].cast(int).to_numpy()]
ip_log = np.log1p(base.counts["device_ip"])
id_log = np.log1p(base.counts["device_id"])


def num_from_codes(Z):
    """Reconstruye las numéricas desde los códigos: mantiene la coherencia al perturbar."""
    h = hod_vals[Z[:, idx["hod"]]].astype(float)
    h = np.where(h < 0, 12.0, h)
    return np.column_stack([ip_log[Z[:, idx["device_ip"]]], id_log[Z[:, idx["device_id"]]],
                            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)])


def to_X(Z):
    return E.transform_codes(Z, num_from_codes(Z)) if hasattr(E, "transform_codes") else E.transform_parts(Z, num_from_codes(Z))


def predict_codes(Z):
    Z = np.asarray(Z).astype(np.int32)
    return M.predict_proba(to_X(Z))


# El código 0 agrupa raras y no vistas: su conteo acumulado no es la actividad de una fila.
# Se usa la actividad media real de esas filas en entrenamiento.
tr = load("train", 100_000)
Zt = base.codes(tr)
nt = base.numeric(tr)
ip_log[0] = nt[Zt[:, idx["device_ip"]] == 0, 0].mean()
id_log[0] = nt[Zt[:, idx["device_id"]] == 0, 1].mean()

va = load("valid")
Zv = base.codes(va)
yv = va["click"].to_numpy()
pv = predict_codes(Zv)[:, 1]
thr = best_threshold(yv, pv)
print("AUC valid", roc_auc_score(yv, pv), "umbral", thr, flush=True)


# Casos sistemáticos: 10 FP y 10 FN con mayor confianza errónea
fp = np.argsort(-(pv * ((yv == 0) & (pv >= thr))))[:10]
fn = np.argsort(pv + 2 * (~((yv == 1) & (pv < thr))))[:10]
casos = pd.DataFrame(dict(i=np.r_[fp, fn], tipo=["FP"] * 10 + ["FN"] * 10, p=pv[np.r_[fp, fn]], y=yv[np.r_[fp, fn]]))
casos.to_csv(OUT / "casos.csv", index=False)


def explainer(seed, kw=None):
    return LimeTabularExplainer(Zt[:5000], feature_names=base.cols, categorical_features=list(range(Zt.shape[1])),
                                class_names=["no", "click"], discretize_continuous=False, kernel_width=kw,
                                sample_around_instance=False, random_state=seed)


def explain(ex, z, ns=5000, k=8):
    e = ex.explain_instance(z, predict_codes, labels=(1,), num_features=k, num_samples=ns)
    w = {base.cols[i]: v for i, v in e.as_map()[1]}
    return w, e.score, e.local_pred[0], predict_codes(z[None])[0, 1]


def jacc(a, b):
    return len(a & b) / len(a | b)


def topk(w, k=5):
    return set(sorted(w, key=lambda c: -abs(w[c]))[:k])


# Explicaciones base, fidelidad y estabilidad con 5 semillas
rows, stab = [], []
ex0 = {s: explainer(s) for s in SEEDS}
for _, c in casos.iterrows():
    z = Zv[int(c.i)]
    ws = {}
    for s in SEEDS:
        w, r2, lp, fp_ = explain(ex0[s], z)
        ws[s] = w
        rows.append(dict(i=int(c.i), tipo=c.tipo, seed=s, r2=r2, local_pred=lp, f=fp_, **{f"w_{k}": v for k, v in w.items()}))
    for s1, s2 in combinations(SEEDS, 2):
        stab.append(dict(i=int(c.i), tipo=c.tipo, s1=s1, s2=s2, j5=jacc(topk(ws[s1]), topk(ws[s2]))))
    print("lime", c.i, flush=True)
pd.DataFrame(rows).to_csv(OUT / "lime.csv", index=False)
pd.DataFrame(stab).to_csv(OUT / "lime_estabilidad.csv", index=False)

# Sensibilidad al ancho del kernel y al número de muestras
kw0 = np.sqrt(Zt.shape[1]) * .75
sens = []
sub = casos.groupby("tipo").head(5)
for kw_f in (.25, .5, 1, 2, 4):
    for ns in (500, 2000, 5000, 15000):
        ex1, ex2 = explainer(42, kw0 * kw_f), explainer(43, kw0 * kw_f)
        for _, c in sub.iterrows():
            z = Zv[int(c.i)]
            w1, r1, *_ = explain(ex1, z, ns)
            w2, r2, *_ = explain(ex2, z, ns)
            sens.append(dict(kw=kw_f, ns=ns, i=int(c.i), r2=(r1 + r2) / 2, j5=jacc(topk(w1), topk(w2))))
    print("sens", kw_f, flush=True)
pd.DataFrame(sens).to_csv(OUT / "lime_sensibilidad.csv", index=False)

# SHAP (KernelExplainer) sobre los mismos 20 casos, en el mismo espacio de códigos
bg = Zt[np.random.default_rng(42).choice(len(Zt), 50, replace=False)]
ke = shap.KernelExplainer(lambda Z: predict_codes(Z)[:, 1], bg)
sv = ke.shap_values(Zv[casos.i.values], nsamples=2000, silent=True)
pd.DataFrame(sv, columns=base.cols).assign(i=casos.i.values, tipo=casos.tipo.values).to_csv(OUT / "shap_casos.csv", index=False)
# Muestra aleatoria de validación para importancia global de SHAP
ii = np.random.default_rng(1).choice(len(Zv), 100, replace=False)
sg = ke.shap_values(Zv[ii], nsamples=1000, silent=True)
pd.DataFrame(sg, columns=base.cols).to_csv(OUT / "shap_global.csv", index=False)
print("shap ok", flush=True)

# Importancia por permutación: se permutan columnas originales en 300 mil filas de validación
r = np.random.default_rng(42)
pick = r.choice(len(Zv), 300_000, replace=False)
Zs, ys = Zv[pick], yv[pick]
a0 = roc_auc_score(ys, predict_codes(Zs)[:, 1])
imp = []
for j, c in enumerate(base.cols):
    ds = []
    for rep in range(5):
        Zp = Zs.copy()
        Zp[:, j] = r.permutation(Zp[:, j])
        ds.append(a0 - roc_auc_score(ys, predict_codes(Zp)[:, 1]))
    imp.append(dict(col=c, caida_auc=np.mean(ds), sd=np.std(ds, ddof=1)))
pd.DataFrame(imp).sort_values("caida_auc", ascending=False).to_csv(OUT / "permutacion.csv", index=False)
print("listo", flush=True)
