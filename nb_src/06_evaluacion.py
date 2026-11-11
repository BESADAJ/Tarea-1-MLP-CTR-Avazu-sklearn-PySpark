# %% [markdown]
# # 6. Evaluación final en la prueba
#
# La prueba es el día 30 y se usa una sola vez, en `scripts/07_final_test.py`. Las decisiones ya estaban tomadas: el umbral
# de cada modelo se fijó en validación maximizando F1. Este capítulo responde P3.
#
# | Modelo | Entorno | Datos de entrenamiento | Semillas |
# |---|---|---|---|
# | MLP `(100,)`, hash12 | scikit-learn | 10 M | 5 |
# | SGD logístico, hash12 | scikit-learn | 10 M | 5 |
# | HistGradientBoosting, target | scikit-learn | 10 M | 5 |
# | MLP `[516, 128, 64, 2]`, hash 2^9 | PySpark | 1 M | 5 |
# | MLP `[27, 50, 2]`, frecuencia | PySpark | 32,4 M | 1 |
# | Regresión logística, hash12 | PySpark | 32,4 M | 1 |
# | GBT, frecuencia y StringIndexer | PySpark | 32,4 M | 1 |

# %%
import sys, json
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ctr.plots import C, style, es
from ctr.config import RES
style()
pd.set_option("display.max_columns", 30)
ST = RES / "stats"
NOM = {"sklearn_mlp": "sklearn MLP", "sklearn_sgd": "sklearn SGD", "sklearn_hgb": "sklearn HGB", "spark_mlp_hash9": "Spark MLP hash",
       "spark_mlp_freq": "Spark MLP frec.", "spark_lr_hash12": "Spark LR", "spark_gbt_gbt": "Spark GBT"}
COL = dict(zip(NOM, [C["azul"], C["aqua"], C["violeta"], C["naranja"], C["amarillo"], C["magenta"], C["rojo"]]))
r = pd.read_csv(ST / "test_resumen.csv"); r["m"] = r.modelo.map(NOM)
cur = json.load(open(ST / "curvas.json"))

# %% [markdown]
# ## 6.1 Tabla de métricas
#
# Media entre semillas. Los modelos con una sola corrida no tienen desviación ni intervalo t.

# %%
p = r.pivot(index="m", columns="metrica", values="media")[["auc", "pr_auc", "logloss", "brier", "ece", "accuracy", "precision", "recall", "f1"]]
p.columns = ["ROC AUC", "PR AUC", "Log loss", "Brier", "ECE", "Exactitud", "Precisión", "Recall", "F1"]
p.sort_values("ROC AUC", ascending=False).round(4)

# %%
a = r[r.metrica == "auc"][["m", "media", "sd", "lo", "hi", "semillas"]].sort_values("media", ascending=False).round(4)
b = pd.read_csv(ST / "bootstrap.csv"); b = b[b.metrica == "auc"].assign(m=lambda d: d.modelo.map(NOM)).set_index("m")
a["boot_lo"], a["boot_hi"] = a.m.map(b.lo).round(4), a.m.map(b.hi).round(4)
a.rename(columns={"m": "Modelo", "media": "AUC media", "sd": "Desv.", "lo": "IC t bajo", "hi": "IC t alto", "boot_lo": "Boot. bajo", "boot_hi": "Boot. alto"}).set_index("Modelo")

# %% [markdown]
# El intervalo t mide variación por semilla. El bootstrap mide variación por muestreo de filas de la prueba: 200 réplicas de
# 1 millón de filas, para una semilla de referencia.
#
# ## 6.2 Curvas ROC y PR

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, c in cur.items():
    ax[0].plot(c["fpr"], c["tpr"], color=COL[k], lw=1.6, label=NOM[k])
    ax[1].plot(c["rc"], c["pr"], color=COL[k], lw=1.6, label=NOM[k])
ax[0].plot([0, 1], [0, 1], color=C["gris"], lw=1, ls="--")
ax[0].set(title="ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos"); ax[0].legend(fontsize=8)
ax[1].set(title="Precisión y recall", xlabel="Recall", ylabel="Precisión", ylim=(0, .8))
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6.3 Calibración

# %%
fig, ax = plt.subplots(figsize=(5.8, 4.4))
for k, c in cur.items():
    ax.plot(c["cal_p"], c["cal_y"], color=COL[k], marker="o", ms=3, lw=1.4, label=NOM[k])
ax.plot([0, .6], [0, .6], color=C["gris"], ls="--", lw=1)
ax.set(title="Curva de calibración (15 bins de igual masa)", xlabel="Probabilidad media predicha", ylabel="Tasa real de clic", xlim=(0, .6), ylim=(0, .6)); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6.4 Utilidad de negocio: lift y ganancia acumulada

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for k, c in cur.items():
    ax[0].plot(np.array(c["frac"]) * 100, np.array(c["gain"]) * 100, color=COL[k], lw=1.6, label=NOM[k])
    ax[1].plot(np.array(c["frac"]) * 100, c["lift"], color=COL[k], lw=1.6)
ax[0].plot([0, 100], [0, 100], color=C["gris"], ls="--", lw=1)
ax[0].set(title="Ganancia acumulada", xlabel="% de impresiones, de mayor a menor score", ylabel="% de clics capturados"); ax[0].legend(fontsize=8)
ax[1].set(title="Lift", xlabel="% de impresiones", ylabel="Lift"); plt.tight_layout(); plt.show()
pd.DataFrame({NOM[k]: {f"Top {int(f*100)} %": g * 100 for f, g in zip(c["frac"], c["gain"]) if round(f, 2) in (.05, .1, .2, .3)} for k, c in cur.items()}).round(1)

# %% [markdown]
# ## 6.5 Inferencia estadística
#
# DeLong compara AUC y McNemar compara aciertos con el umbral de validación, con corrección de Holm. Con 4,2 millones de
# filas casi toda diferencia resulta significativa, de modo que importa también el tamaño del efecto.

# %%
d = pd.read_csv(ST / "delong.csv"); d["A"], d["B"] = d.a.map(NOM), d.c.map(NOM)
d[["A", "B", "auc_a", "auc_b", "dif", "z", "p_holm"]].round(5).sort_values("dif", key=abs).reset_index(drop=True)

# %%
m = pd.read_csv(ST / "mcnemar.csv"); m["A"], m["B"] = m.a.map(NOM), m.c.map(NOM)
m[["A", "B", "solo_a", "solo_b", "chi2", "p_holm"]].round(5).reset_index(drop=True)

# %% [markdown]
# **Respuesta a P3: no.** El MLP de scikit-learn (AUC media 0,717) queda detrás de HGB con target encoding (0,733), de la regresión
# logística de Spark con los 32,4 millones (0,7255) y de SGD (0,721). DeLong da todas las diferencias significativas tras Holm, con
# la semilla de referencia. Es el resultado esperable con 4,2 millones de filas: la cifra que importa es la magnitud. Frente a HGB,
# el MLP pierde 0,010 a 0,016 de AUC, lo bastante para mantenerse a lo largo de la curva de ganancia.
#
# **El MLP es el más inestable y el peor calibrado de scikit-learn.** Su desviación entre semillas en la prueba es 0,0065, frente a
# 0,0009 de HGB, y su ECE de 0,029 es más del doble. Con la semilla 42 supera a SGD (0,7235 frente a 0,7209). En promedio, no.
#
# **Dentro de Spark, la regresión logística supera a los dos MLP y a GBT.** El MLP con hash 2^9 y GBT empatan en AUC (0,698), y el
# MLP con frecuencia y los datos completos es el peor modelo (0,639). Ese GBT usó 20 árboles por tiempo, y con 1 millón y 50 árboles
# daba más en validación. No es una cota de lo que GBT puede hacer.
#
# **La única diferencia no significativa en McNemar** es entre la LR de Spark y el MLP de Spark con hashing (p = 0,84 tras Holm). Con
# sus umbrales aciertan casi las mismas filas, pese a la diferencia de AUC.
