"""Experimentos de scikit-learn. Uso: python 03_sk_experiments.py <etapa> [workers].

Etapas:
  enc    factor codificación (P2) con 1 M de filas, 5 semillas
  tune   búsqueda bayesiana con Optuna evaluando en el día 29
  scale  curvas de aprendizaje con 100 mil, 1 M y 10 M, 5 semillas
  abl    ablaciones: cruces, profundidad, ponderación de clases y sin identificadores

Cada ajuste escribe una línea en results/sk/runs.jsonl, y lo ya hecho se salta al reanudar.
Los tiempos aquí sirven de orientación: corren en paralelo. La medición de costo limpia está
en scripts/05_benchmark.py, sin otras cargas en el equipo.
"""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings

import joblib
import numpy as np

from ctr.config import ALL_CATS, CAT_COLS, CROSSES, DATA, RES, SEEDS, TIME_CATS
from ctr.data import load
from ctr.encoders import Base, add_crosses, make_encoder
from ctr.metrics import best_threshold, scores, threshold_scores
from ctr.monitor import Peak, append_jsonl, read_jsonl

warnings.filterwarnings("ignore")
RUNS = RES / "sk" / "runs.jsonl"
BEST = RES / "sk" / "best_params.json"
CACHE = DATA / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
MLP_DEF = dict(hidden=(100, 50), alpha=1e-3, lr=1e-3, batch=1024)


def key(c):
    return "|".join(str(c.get(k)) for k in ("exp", "model", "enc", "n", "seed", "tag", "trial"))


def cache_path(enc, n, tag):
    return CACHE / f"{enc}_{n}_{tag}.joblib"


def build(enc, n, tag="base"):
    """Codifica una vez y guarda en disco; los procesos lo abren con mmap."""
    p = cache_path(enc, n, tag)
    pe = CACHE / f"encoder_{enc}_{n}_{tag}.joblib"
    if p.exists() and pe.exists():
        return
    tr, va = load("train", n), load("valid")
    cols = ALL_CATS
    if tag == "cruces":
        tr, va = add_crosses(tr), add_crosses(va)
        cols = ALL_CATS + [f"{a}_x_{b}" for a, b in CROSSES]
    if tag == "sin_ids":
        cols = [c for c in ALL_CATS if c not in ("device_ip", "device_id")]
    t0 = time.perf_counter()
    b = Base(cols).fit(tr)
    E = make_encoder(enc, b)
    X = E.fit_transform(tr)
    Xv = E.transform(va)
    if tag == "sin_ids":                     # las numéricas de actividad también identifican
        X, Xv = X[:, :-4].tocsr() if hasattr(X, "tocsr") else X[:, :-4], Xv[:, :-4].tocsr() if hasattr(Xv, "tocsr") else Xv[:, :-4]
    joblib.dump(dict(X=X, y=tr["click"].to_numpy(), Xv=Xv, yv=va["click"].to_numpy(),
                     enc_sec=time.perf_counter() - t0), p)
    joblib.dump(E, pe)
    print("cache", p.name, X.shape, flush=True)


def get_model(c):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    w = c.get("weight")
    if c["model"] == "mlp":
        p = {**MLP_DEF, **c.get("params", {})}
        return MLPClassifier(hidden_layer_sizes=tuple(p["hidden"]), alpha=p["alpha"], learning_rate_init=p["lr"],
                             batch_size=p["batch"], early_stopping=True, n_iter_no_change=3, max_iter=30,
                             random_state=c["seed"])
    if c["model"] == "sgd":
        return SGDClassifier(loss="log_loss", alpha=c.get("params", {}).get("alpha", 1e-6), max_iter=20, tol=1e-4,
                             class_weight=w, random_state=c["seed"])
    if c["model"] == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False,
                                              class_weight=w, random_state=c["seed"])
    raise ValueError(c["model"])


def run(c):
    tag = c.get("tag", "base")
    d = joblib.load(cache_path(c["enc"], c["n"], "base" if tag.startswith("h") else tag), mmap_mode="r")
    X, y, Xv, yv = d["X"], np.asarray(d["y"]), d["Xv"], np.asarray(d["yv"])
    if c.get("oversample"):                  # MLPClassifier no admite pesos: se repiten positivos
        r = np.random.default_rng(c["seed"])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        i = r.permutation(np.r_[neg, r.choice(pos, len(neg))])
        X, y = X[i], y[i]
    m = get_model(c)
    with Peak() as t:
        m.fit(X, y)
    t0 = time.perf_counter()
    p = m.predict_proba(Xv)[:, 1]
    pred = time.perf_counter() - t0
    thr = best_threshold(yv, p)
    row = {**c, "key": key(c), "fit_sec": t.sec, "pred_sec": pred, "gb": t.gb, "d": X.shape[1], "n_real": X.shape[0],
           "iters": int(np.ravel(getattr(m, "n_iter_", 0))[0]),
           **{f"v_{k}": v for k, v in scores(yv, p).items()},
           **{f"v_{k}": v for k, v in threshold_scores(yv, p, thr).items()}}
    append_jsonl(RUNS, row)
    if c.get("save"):
        d = DATA / "models"
        d.mkdir(exist_ok=True)
        joblib.dump(m, d / f"{c['model']}_{c['enc']}_{c['n']}_{c['seed']}.joblib")
    print(row["key"], round(row["v_auc"], 4), round(t.sec), flush=True)
    return row["v_auc"]


def pending(cfgs):
    done = {r["key"] for r in read_jsonl(RUNS)}
    return [c for c in cfgs if key(c) not in done]


def par(cfgs, w):
    from joblib import Parallel, delayed
    cfgs = pending(cfgs)
    print(len(cfgs), "ajustes pendientes", flush=True)
    Parallel(n_jobs=w, backend="loky")(delayed(run)(c) for c in cfgs)


def best_enc():
    """hash12 por presupuesto, no por AUC.

    hash16 tuvo la mayor AUC en validación (0,7400 frente a 0,7382), pero cada ajuste con 1 M de
    filas tardó 37 min frente a 3 min. Con 10 M y 30 intentos de Optuna no cabe en el tiempo
    disponible. La pérdida aceptada, 0,0018 de AUC, se reporta en el capítulo 4.
    """
    return "hash12"


def tune_worker(wid, enc, n, trials):
    import json

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = f"sqlite:///{(RES / 'sk' / 'optuna.db').as_posix()}"
    st = optuna.create_study(study_name="mlp", storage=db, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=42 + wid, multivariate=True))
    H = {"50": (50,), "100": (100,), "100_50": (100, 50), "256_128": (256, 128)}

    def obj(t):
        p = dict(hidden=H[t.suggest_categorical("hidden", list(H))], alpha=t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                 lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), batch=t.suggest_categorical("batch", [512, 1024, 2048, 4096]))
        return run(dict(exp="tune", model="mlp", enc=enc, n=n, seed=42, tag="base", params=p, trial=f"{wid}-{t.number}"))

    st.optimize(obj, n_trials=trials)


def main():
    stage, w = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
    M = 1_000_000
    if stage == "enc":
        encs = ["hash12", "hash14", "hash16", "freq", "target"]
        for e in encs:
            build(e, M)
        cf = [dict(exp="enc", model="mlp", enc=e, n=M, seed=s, tag="base") for e in encs for s in SEEDS]
        cf += [dict(exp="enc", model="sgd", enc=e, n=M, seed=s, tag="base") for e in encs for s in SEEDS]
        cf += [dict(exp="enc", model="hgb", enc=e, n=M, seed=s, tag="base") for e in ("freq", "target") for s in SEEDS]
        par(sorted(cf, key=lambda c: c["model"] != "sgd"), w)
    elif stage == "tune":
        import json

        from joblib import Parallel, delayed
        enc = best_enc()
        print("codificación elegida con validación:", enc, flush=True)
        Parallel(n_jobs=w, backend="loky")(delayed(tune_worker)(i, enc, M, int(sys.argv[3])) for i in range(w))
        import optuna
        st = optuna.load_study(study_name="mlp", storage=f"sqlite:///{(RES / 'sk' / 'optuna.db').as_posix()}")
        H = {"50": (50,), "100": (100,), "100_50": (100, 50), "256_128": (256, 128)}
        bp = dict(st.best_params)
        bp["hidden"] = H[bp["hidden"]]
        json.dump(dict(enc=enc, params=bp, auc=st.best_value, trials=len(st.trials)), open(BEST, "w"), indent=1)
        st.trials_dataframe().to_csv(RES / "sk" / "optuna_trials.csv", index=False)
        print(bp, st.best_value)
    elif stage in ("scale", "abl"):
        import json
        b = json.load(open(BEST))
        enc, p = b["enc"], b["params"]
        if stage == "scale":
            sizes = [int(x) for x in sys.argv[3].split(",")]
            for n in sizes:
                from joblib import Parallel, delayed
                # en un proceso aparte: la memoria de la codificación vuelve al sistema
                for e in (enc, "target"):
                    Parallel(n_jobs=2, backend="loky")(delayed(build)(e, n) for _ in [0])
                sv = n == max(sizes)         # los modelos del tamaño mayor van a la prueba final
                cf = [dict(exp="scale", model="mlp", enc=enc, n=n, seed=s, tag="base", params=p, save=sv) for s in SEEDS]
                cf += [dict(exp="scale", model="sgd", enc=enc, n=n, seed=s, tag="base", save=sv) for s in SEEDS]
                cf += [dict(exp="scale", model="hgb", enc="target", n=n, seed=s, tag="base", save=sv) for s in SEEDS]
                par(cf, w)
        else:
            for t in ("cruces", "sin_ids"):
                build(enc, M, t)
            build(enc, M)
            build("target", M)
            cf = []
            for s in SEEDS:
                cf.append(dict(exp="abl", model="mlp", enc=enc, n=M, seed=s, tag="cruces", params=p))
                cf.append(dict(exp="abl", model="mlp", enc=enc, n=M, seed=s, tag="sin_ids", params=p))
                cf.append(dict(exp="abl_pos", model="mlp", enc=enc, n=M, seed=s, tag="base", params=p, oversample=True))
                cf.append(dict(exp="abl_pos", model="sgd", enc=enc, n=M, seed=s, tag="base", weight="balanced"))
                cf.append(dict(exp="abl_pos", model="hgb", enc="target", n=M, seed=s, tag="base", weight="balanced"))
                for h in ([50], [100], [100, 50], [128, 64, 32], [256, 128]):
                    cf.append(dict(exp="abl_depth", model="mlp", enc=enc, n=M, seed=s,
                                   tag="h" + "x".join(map(str, h)) , params={**p, "hidden": h}))
            par(cf, w)


if __name__ == "__main__":
    main()
