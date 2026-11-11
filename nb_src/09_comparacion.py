# %% [markdown]
# # 9. Comparación de resultados
#
# Este capítulo reúne en una sola vista lo medido en los capítulos 4 a 8: calidad por entorno, modelo y codificación, y su costo.

# %%
import sys, json
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ctr.plots import C, style, es
from ctr.config import RES
from ctr.monitor import read_jsonl
from ctr.metrics import t_ci
style()
pd.set_option("display.max_columns", 30)
R = pd.DataFrame(read_jsonl(RES / "sk" / "runs.jsonl")); S = pd.DataFrame(read_jsonl(RES / "spark" / "runs.jsonl"))
S = S[S.status == "ok"]

# %% [markdown]
# ## 9.1 Por entorno, modelo y codificación
#
# Validación, día 29, con 1 millón de filas o el tamaño indicado. Media, desviación e intervalo t al 95 % cuando hay varias semillas.

# %%
def fila(env, mod, enc, g, col="v_auc"):
    m, s, lo, hi = t_ci(g[col].values)
    return {"Entorno": env, "Modelo": mod, "Codificación": enc, "n": int(g.n.iloc[0]), "Semillas": len(g), "AUC": m, "Desv.": s, "IC bajo": lo, "IC alto": hi,
            "Log loss": g.v_logloss.mean(), "ECE": g.v_ece.mean(), "Ajuste (s)": g.fit_sec.mean(), "GB pico": g.gb.mean()}
filas = []
e = R[R.exp == "enc"]
for (m, c), g in e.groupby(["model", "enc"]):
    filas.append(fila("scikit-learn", m.upper(), c, g))
s = R[R.exp == "scale"]
for (m, c, n), g in s[s.n > 5e6].groupby(["model", "enc", "n"]):
    filas.append(fila("scikit-learn", m.upper(), c, g))
sp = S[S.exp.isin(["scale", "full"])].copy()
sp["cap"] = sp.layers.map(lambda l: "x".join(map(str, l)))
for (m, c, ca, n, mi), g in sp.groupby(["model", "enc", "cap", "n", "maxIter"]):
    filas.append(fila("PySpark", (m + " " + ca).upper().strip(), c, g))
T = pd.DataFrame(filas).sort_values("AUC", ascending=False).round(4).reset_index(drop=True)
T

# %% [markdown]
# ## 9.2 Barras comparativas de tiempo y memoria

# %%
d = T[T.n > 5e5].sort_values("Ajuste (s)")
d = d[d["Entorno"].eq("PySpark") | d.n.gt(5e6)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
et = d.Entorno + " " + d.Modelo + " " + d["Codificación"] + " n=" + (d.n / 1e6).round(1).astype(str) + "M"
col = [C["azul"] if x == "scikit-learn" else C["naranja"] for x in d.Entorno]
ax[0].barh(et, d["Ajuste (s)"] / 60, color=col, height=.65); ax[0].set(title="Tiempo de ajuste", xlabel="minutos (con carga paralela)")
ax[1].barh(et, d["GB pico"], color=col, height=.65); ax[1].set(title="Memoria pico", xlabel="GB")
ax[0].tick_params(axis="y", labelsize=8); plt.tight_layout(); plt.show()

# %% [markdown]
# **Lectura de conjunto:**
#
# - **El orden por calidad no depende del entorno, sino de la representación.** Arriba quedan los modelos que ven la identidad de
#   las categorías con muchas filas: HGB con target, MLP y SGD con hashing en scikit-learn, y la LR dispersa de Spark. Abajo queda
#   todo lo que usa frecuencia sola.
# - **El MLP de Spark está acotado por diseño,** no por falta de ajuste: con hashing no puede usar más de 1 millón de filas, y con
#   frecuencia pierde la identidad.
# - **El costo crece mucho más rápido que la calidad.** hash16 cuesta 12 veces hash12 por 0,002 de AUC. El MLP con 10 millones cuesta
#   13 veces el de 1 millón por 0,001.
# - **Los tiempos de esta tabla tienen carga paralela** y sirven solo para comparar órdenes de magnitud. Las mediciones limpias están
#   en el capítulo 7.
