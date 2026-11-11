"""Preprocesamiento de scikit-learn: raras, hashing, frecuencia y target encoding.

Todo se ajusta solo con entrenamiento. Cada columna categórica se convierte a códigos enteros
(0 = __RARE__, que también recibe categorías no vistas) y luego se codifica según la estrategia.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

from .config import ALL_CATS, CROSSES, MIN_COUNT, RARE


def add_crosses(df):
    """Agrega cruces de variables como columnas de texto (ablación)."""
    return df.with_columns([(pl.col(a).cast(pl.String) + "|" + pl.col(b).cast(pl.String)).alias(f"{a}_x_{b}")
                            for a, b in CROSSES])


class Base:
    """Vocabulario por columna, con umbral de raras definido en entrenamiento."""

    def __init__(self, cols=None, min_count=MIN_COUNT):
        self.cols = list(cols or ALL_CATS)
        self.min_count = min_count

    def fit(self, df):
        self.vocab, self.counts, self.pos = {}, {}, {}
        y = df["click"].to_numpy() if "click" in df.columns else None
        self.prior = float(y.mean()) if y is not None else 0.0
        for c in self.cols:
            agg = (df.lazy().select(pl.col(c).cast(pl.String).alias("v"), pl.col("click"))
                   .group_by("v").agg(pl.len().alias("n"), pl.col("click").sum().alias("c"))
                   .filter(pl.col("n") >= self.min_count).sort("n", descending=True).collect())
            n_rare = df.height - agg["n"].sum()
            c_rare = int(y.sum()) - agg["c"].sum()
            self.vocab[c] = pl.DataFrame({"v": agg["v"], "code": np.arange(1, agg.height + 1, dtype=np.int32)})
            self.counts[c] = np.r_[n_rare, agg["n"].to_numpy()].astype(np.float64)
            self.pos[c] = np.r_[c_rare, agg["c"].to_numpy()].astype(np.float64)
        # Actividad (cola pesada) usada como numérica: conteo real en entrenamiento, antes de raras
        self.activity = {}
        for c in ("device_ip", "device_id"):
            self.activity[c] = df.group_by(c).len().rename({"len": "act"})
        return self

    def codes(self, df):
        """Matriz n x columnas con códigos; lo desconocido o raro cae en 0."""
        out = np.zeros((df.height, len(self.cols)), dtype=np.int32)
        for j, c in enumerate(self.cols):
            m = df.select(pl.col(c).cast(pl.String).alias("v")).join(self.vocab[c], on="v", how="left",
                                                                     maintain_order="left")
            out[:, j] = m["code"].fill_null(0).to_numpy()
        return out

    def numeric(self, df):
        """Numéricas: log1p de actividad y hora cíclica. Sin escalar."""
        parts = []
        for c in ("device_ip", "device_id"):
            a = df.select(c).join(self.activity[c], on=c, how="left", maintain_order="left")["act"].fill_null(0)
            parts.append(np.log1p(a.to_numpy().astype(np.float64)))
        h = df["hod"].to_numpy().astype(np.float64)
        parts += [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]
        return np.column_stack(parts)


class HashingEnc:
    """Hashing trick con dimensión 2^bits más numéricas escaladas (salida dispersa)."""

    def __init__(self, base, bits):
        self.b, self.bits, self.dim = base, bits, 2 ** bits
        fh = FeatureHasher(n_features=self.dim, input_type="string", alternate_sign=True)
        self.idx, self.sgn = [], []
        for c in base.cols:
            toks = [[f"{c}={RARE}"]] + [[f"{c}={v}"] for v in base.vocab[c]["v"].to_list()]
            H = fh.transform(toks).tocsr()
            self.idx.append(H.indices.astype(np.int32))
            self.sgn.append(H.data.astype(np.float32))

    def _hash(self, codes):
        n, k = codes.shape
        ind = np.empty(n * k, dtype=np.int32)
        dat = np.empty(n * k, dtype=np.float32)
        for j in range(k):
            ind[j::k] = self.idx[j][codes[:, j]]
            dat[j::k] = self.sgn[j][codes[:, j]]
        return sp.csr_matrix((dat, ind, np.arange(0, n * k + 1, k, dtype=np.int64)), shape=(n, self.dim))

    def fit_transform(self, df):
        num = self.b.numeric(df)
        self.sc = StandardScaler().fit(num)
        return self._stack(self.b.codes(df), num)

    def transform(self, df):
        return self._stack(self.b.codes(df), self.b.numeric(df))

    def transform_codes(self, codes, num):
        return self._stack(codes, num)

    def _stack(self, codes, num):
        return sp.hstack([self._hash(codes), sp.csr_matrix(self.sc.transform(num).astype(np.float32))],
                         format="csr", dtype=np.float32)


class FreqEnc:
    """Una columna por variable: log1p del conteo en entrenamiento (salida densa)."""

    def __init__(self, base):
        self.b = base
        self.log = [np.log1p(base.counts[c]) for c in base.cols]

    def _raw(self, codes):
        return np.column_stack([self.log[j][codes[:, j]] for j in range(codes.shape[1])])

    def fit_transform(self, df):
        X = np.column_stack([self._raw(self.b.codes(df)), self.b.numeric(df)])
        self.sc = StandardScaler().fit(X)
        return self.sc.transform(X).astype(np.float32)

    def transform(self, df):
        return self.transform_parts(self.b.codes(df), self.b.numeric(df))

    def transform_parts(self, codes, num):
        return self.sc.transform(np.column_stack([self._raw(codes), num])).astype(np.float32)


class TargetEnc:
    """Target encoding fuera de pliegue. Solo scikit-learn; ver limitación en el capítulo 3.

    En entrenamiento cada fila recibe la media calculada sin su pliegue. Validación y prueba
    usan la tabla completa de entrenamiento. Los pliegues son por día, para respetar el tiempo:
    una fila nunca se codifica con filas de su mismo día.
    """

    def __init__(self, base, smooth=20.0, folds=None):
        self.b, self.m, self.folds = base, smooth, folds

    def _table(self, n, c):
        return (c + self.m * self.b.prior) / (n + self.m)

    def fit_transform(self, df):
        y = df["click"].to_numpy().astype(np.float64)
        codes = self.b.codes(df)
        fold = df["day"].to_numpy() if self.folds is None else self.folds
        Xc = np.zeros(codes.shape, dtype=np.float64)
        for j, c in enumerate(self.b.cols):
            K = self.b.counts[c].size
            N, C = np.bincount(codes[:, j], minlength=K).astype(float), np.bincount(codes[:, j], y, minlength=K)
            for f in np.unique(fold):
                m = fold == f
                nf = np.bincount(codes[m, j], minlength=K).astype(float)
                cf = np.bincount(codes[m, j], y[m], minlength=K)
                Xc[m, j] = self._table(N - nf, C - cf)[codes[m, j]]
        self.tab = [self._table(np.bincount(codes[:, j], minlength=self.b.counts[c].size).astype(float),
                                np.bincount(codes[:, j], y, minlength=self.b.counts[c].size))
                    for j, c in enumerate(self.b.cols)]
        X = np.column_stack([Xc, self.b.numeric(df)])
        self.sc = StandardScaler().fit(X)
        return self.sc.transform(X).astype(np.float32)

    def transform(self, df):
        return self.transform_parts(self.b.codes(df), self.b.numeric(df))

    def transform_parts(self, codes, num):
        Xc = np.column_stack([self.tab[j][codes[:, j]] for j in range(codes.shape[1])])
        return self.sc.transform(np.column_stack([Xc, num])).astype(np.float32)


def make_encoder(name, base):
    if name.startswith("hash"):
        return HashingEnc(base, int(name[4:]))
    if name == "freq":
        return FreqEnc(base)
    if name == "target":
        return TargetEnc(base)
    raise ValueError(name)
