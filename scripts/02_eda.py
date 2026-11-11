"""EDA con lectura perezosa (DuckDB y Polars). Guarda tablas en results/eda.

Las etiquetas del día 30 no se miran aquí: las tablas con `click` usan días 21 a 29.
Prueba solo entra en comparaciones sin etiqueta (PSI, categorías no vistas).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import duckdb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import mutual_info_score

from ctr.config import ALL_CATS, CAT_COLS, MIN_COUNT, PARQUET, RES, TRAIN_DAYS

OUT = RES / "eda"
gl = str(PARQUET / "**" / "*.parquet").replace("\\", "/")
src = f"read_parquet('{gl}', hive_partitioning=true)"
con = duckdb.connect()
con.execute("set threads=6; set memory_limit='10GB'")
tr = f"{src} where day between {TRAIN_DAYS[0]} and {TRAIN_DAYS[-1]}"
franja = ("case when hod<6 then 'madrugada' when hod<12 then 'manana' "
          "when hod<18 then 'tarde' else 'noche' end")

# Tipos, nulos y valores centinela
sch = pl.scan_parquet(gl, hive_partitioning=True).collect_schema()
pd.DataFrame({"col": list(sch.names()), "tipo": [str(t) for t in sch.dtypes()]}).to_csv(OUT / "tipos.csv", index=False)
nul = con.execute("select " + ", ".join(f"sum(({c} is null)::int) \"{c}\"" for c in sch.names()) + f" from {src}").df().T
nul.columns = ["nulos"]
nul.to_csv(OUT / "nulos.csv")

# Tasa base por día, hora y franja (21 a 29)
con.execute(f"copy (select day, count(*) n, avg(click) ctr from {src} where day<30 group by 1 order by 1) to '{OUT.as_posix()}/ctr_dia.csv' (header)")
con.execute(f"copy (select hod, count(*) n, avg(click) ctr from {tr} group by 1 order by 1) to '{OUT.as_posix()}/ctr_hora.csv' (header)")
con.execute(f"copy (select dow, count(*) n, avg(click) ctr from {tr} group by 1 order by 1) to '{OUT.as_posix()}/ctr_dow.csv' (header)")
con.execute(f"copy (select {franja} franja, count(*) n, avg(click) ctr from {tr} group by 1 order by 1) to '{OUT.as_posix()}/ctr_franja.csv' (header)")
con.execute(f"copy (select day, hod, count(*) n, avg(click) ctr from {src} where day<30 group by 1,2 order by 1,2) to '{OUT.as_posix()}/ctr_dia_hora.csv' (header)")
con.execute(f"copy (select click, count(*) n from {src} where day<30 group by 1) to '{OUT.as_posix()}/click.csv' (header)")

# Cardinalidad, cola larga, categorías vistas una vez y raras
card, curvas, tops, unseen = [], [], [], []
for c in ALL_CATS:
    g = con.execute(f"select {c}::varchar v, count(*) n, avg(click) ctr from {tr} group by 1 order by n desc").df()
    tot, k = g.n.sum(), len(g)
    card.append(dict(col=c, categorias=k, singletons=int((g.n == 1).sum()), pct_cat_singleton=(g.n == 1).mean(),
                     pct_filas_singleton=g.n[g.n == 1].sum() / tot, pct_filas_rara=g.n[g.n < MIN_COUNT].sum() / tot,
                     pct_cat_rara=(g.n < MIN_COUNT).mean(), top1=g.n.iloc[0] / tot, top10=g.n.iloc[:10].sum() / tot))
    cum = g.n.cumsum().values / tot
    idx = np.unique(np.geomspace(1, k, 60).astype(int)) - 1
    curvas.append(pd.DataFrame(dict(col=c, frac_cat=(idx + 1) / k, n_cat=idx + 1, cum=cum[idx])))
    t = g.head(12).assign(col=c, share=lambda d: d.n / tot)
    tops.append(t)
    for d, name in ((29, "valid"), (30, "test")):
        v = con.execute(f"select {c}::varchar v, count(*) n from {src} where day={d} group by 1").df()
        known = set(g.v)
        mask = ~v.v.isin(known)
        unseen.append(dict(col=c, split=name, pct_filas_no_vistas=v.n[mask].sum() / v.n.sum(), pct_cat_no_vistas=mask.mean()))
    print(c, k, flush=True)
pd.DataFrame(card).to_csv(OUT / "cardinalidad.csv", index=False)
pd.concat(curvas).to_csv(OUT / "cola_larga.csv", index=False)
pd.concat(tops).to_csv(OUT / "top_categorias.csv", index=False)
pd.DataFrame(unseen).to_csv(OUT / "no_vistas.csv", index=False)


# Asociación sobre una muestra del 5 % del entrenamiento
def top_k(s, k=100):
    vc = s.value_counts()
    keep = vc.index[:k]
    return s.where(s.isin(keep), "__otros__")


S = con.execute(f"select click, {','.join(c + '::varchar ' + c for c in ALL_CATS)} from {tr} and sample_key<0.05").df()
y = S.click.values
rows = []
for c in ALL_CATS:
    x = top_k(S[c])
    tab = pd.crosstab(x, y).values
    chi2 = chi2_contingency(tab, correction=False)[0]
    v = np.sqrt(chi2 / (tab.sum() * min(tab.shape[0] - 1, tab.shape[1] - 1)))
    rows.append(dict(col=c, mi=mutual_info_score(x, y), cramer_v=v, n=len(y)))
pd.DataFrame(rows).sort_values("mi", ascending=False).to_csv(OUT / "asociacion.csv", index=False)
num = ["click", "hod", "dow", "banner_pos", "device_type", "device_conn_type", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
M = S[num].astype(float)
rho = spearmanr(M.values).statistic
pd.DataFrame(rho, index=num, columns=num).to_csv(OUT / "spearman.csv")

# Numéricas derivadas: actividad por dispositivo e IP dentro de entrenamiento
for c in ("device_ip", "device_id"):
    h = con.execute(f"""select floor(log2(n))::int b, count(*) filas, sum(n) impresiones, sum(c)/sum(n) ctr
                        from (select {c}, count(*) n, sum(click) c from {tr} group by 1) group by 1 order by 1""").df()
    h.to_csv(OUT / f"hist_actividad_{c}.csv", index=False)


# PSI entrenamiento vs prueba, sin etiquetas. Top 20 de entrenamiento + otros
def psi(c):
    a = con.execute(f"select {c}::varchar v, count(*) n from {tr} group by 1 order by n desc").df()
    b = con.execute(f"select {c}::varchar v, count(*) n from {src} where day=30 group by 1").df()
    keep = list(a.v[:20])
    pa = a.assign(k=a.v.where(a.v.isin(keep), "__o__")).groupby("k").n.sum()
    pb = b.assign(k=b.v.where(b.v.isin(keep), "__o__")).groupby("k").n.sum()
    i = pa.index.union(pb.index)
    p = (pa.reindex(i).fillna(0) + 1) / (pa.sum() + len(i))
    q = (pb.reindex(i).fillna(0) + 1) / (pb.sum() + len(i))
    return float(((q - p) * np.log(q / p)).sum())


pd.DataFrame([dict(col=c, psi=psi(c)) for c in ALL_CATS]).sort_values("psi", ascending=False).to_csv(OUT / "psi.csv", index=False)
print("EDA listo")
