"""Métricas de discriminación, probabilísticas, de umbral y pruebas estadísticas."""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_recall_curve, roc_auc_score)


def ece(y, p, bins=15):
    """Error de calibración esperado con bins de igual masa (cuantiles)."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    i = np.clip(np.searchsorted(np.unique(q), p, side="right") - 1, 0, len(np.unique(q)) - 2)
    n = np.bincount(i)
    ok = n > 0
    return float(np.sum(n[ok] / len(y) * np.abs(np.bincount(i, y)[ok] / n[ok] - np.bincount(i, p)[ok] / n[ok])))


def calibration(y, p, bins=15):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    i = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    n = np.bincount(i)
    ok = n > 0
    return np.bincount(i, p)[ok] / n[ok], np.bincount(i, y)[ok] / n[ok], n[ok]


def best_threshold(y, p):
    """Umbral que maximiza F1. Se elige en validación y se aplica en prueba."""
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[np.argmax(f)])


def scores(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), logloss=log_loss(y, p),
                brier=brier_score_loss(y, p), ece=ece(y, p))


def threshold_scores(y, p, t):
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(thr=t, accuracy=(tp + tn) / len(y), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                f1=f1_score(y, yh), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def lift_gain(y, p, bins=20):
    """Por fracción contactada, de mayor a menor probabilidad: ganancia acumulada y lift."""
    o = np.argsort(-p)
    cy = np.cumsum(y[o])
    idx = (np.linspace(1 / bins, 1, bins) * len(y)).astype(int) - 1
    frac = (idx + 1) / len(y)
    gain = cy[idx] / cy[-1]
    return frac, gain, gain / frac


def _midrank(x):
    return stats.rankdata(x, method="average")


def delong(y, a, b):
    """DeLong rápido (Sun y Xu, 2014). Devuelve AUC de a y b, diferencia, z y p bilateral."""
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())   # en Windows la suma de bool es int32 y m*n se desborda
    aucs, v01, v10 = [], [], []
    for s in (a, b):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[pos] - tx) / n)
        v10.append(1 - (tz[neg] - ty) / m)
    sx, sy = np.cov(np.array(v01)), np.cov(np.array(v10))
    s = sx / m + sy / n
    d = aucs[0] - aucs[1]
    se = np.sqrt(max(s[0, 0] + s[1, 1] - 2 * s[0, 1], 1e-30))
    z = d / se
    return dict(auc_a=aucs[0], auc_b=aucs[1], dif=d, se=se, z=z, p=2 * stats.norm.sf(abs(z)))


def mcnemar(y, ya, yb):
    """McNemar con corrección de continuidad sobre los aciertos de dos clasificadores."""
    ok_a, ok_b = ya == y, yb == y
    b01, b10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    chi = (abs(b01 - b10) - 1) ** 2 / max(b01 + b10, 1)
    return dict(solo_a=b01, solo_b=b10, chi2=chi, p=float(stats.chi2.sf(chi, 1)))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[o]).clip(max=1)
    r = np.empty_like(adj)
    r[o] = adj
    return r


def bootstrap_ci(y, p, fn, B=500, seed=42, alpha=0.05, sub=None):
    """Bootstrap de percentiles. `sub` limita filas para que AUC sea viable con millones."""
    r = np.random.default_rng(seed)
    n = len(y)
    vals = []
    size = sub or n
    for _ in range(B):
        i = r.integers(0, n, size)
        vals.append(fn(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def t_ci(x, alpha=0.05):
    """Media, desviación e IC t de un vector de repeticiones por semilla."""
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
    h = stats.t.ppf(1 - alpha / 2, len(x) - 1) * s / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, s, m - h, m + h
