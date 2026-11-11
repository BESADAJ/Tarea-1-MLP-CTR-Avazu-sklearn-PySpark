"""Resume lo que muestra Spark UI para el punto `spark mlp 3000000 8 spark_ui` del benchmark.

En Windows sin winutils, Spark no puede escribir el registro de eventos (pide permisos POSIX a Hadoop), así que
`05_benchmark.py` guarda la API REST de la UI en vivo (`/api/v1/applications/<id>/...`) en results/bench/spark_ui/.
Este script la resume en spark_ui_stages.csv (por punto de llamada) y spark_ui_resumen.csv.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ctr.config import RES

D = RES / "bench" / "spark_ui"
S = pd.DataFrame(json.load(open(D / "stages.json")))
S = S[S.status == "COMPLETE"].copy()
t = lambda c: pd.to_datetime(S[c].str.replace("GMT", ""), format="%Y-%m-%dT%H:%M:%S.%f")
S["dur_s"] = (t("completionTime") - t("submissionTime")).dt.total_seconds()
S["llamada"] = S.name.str.split(" at ").str[0] + " @ " + S.name.str.split(" at ").str[1].str.split(":").str[0]


def fase(det):
    d = det or ""
    if "MultilayerPerceptron" in d or "LBFGS" in d or "RDDLossFunction" in d or "optimize" in d.lower():
        return "Entrenamiento (l-bfgs)"
    if "BinaryClassification" in d or "AreaUnderCurve" in d:
        return "Evaluación"
    if "StringIndexer" in d or "Prep" in d or "Pipeline" in d or "VectorAssembler" in d:
        return "Preprocesamiento"
    return "Lectura y preprocesamiento"


S["fase"] = S.details.map(fase)
MB = 2 ** 20
G = (S.groupby(["fase", "llamada"])
     .agg(etapas=("stageId", "size"), tareas=("numTasks", "sum"), dur_s=("dur_s", "sum"), run_s=("executorRunTime", lambda x: x.sum() / 1e3),
          cpu_s=("executorCpuTime", lambda x: x.sum() / 1e9), gc_s=("jvmGcTime", lambda x: x.sum() / 1e3),
          deser_s=("executorDeserializeTime", lambda x: x.sum() / 1e3), entrada_mb=("inputBytes", lambda x: x.sum() / MB),
          shuffle_lect_mb=("shuffleReadBytes", lambda x: x.sum() / MB), shuffle_escr_mb=("shuffleWriteBytes", lambda x: x.sum() / MB),
          spill_disco_mb=("diskBytesSpilled", lambda x: x.sum() / MB), mem_pico_mb=("peakExecutionMemory", lambda x: x.max() / MB))
     .reset_index().sort_values("dur_s", ascending=False))
G.to_csv(RES / "bench" / "spark_ui_stages.csv", index=False)
E = json.load(open(D / "executors.json"))[0]
J = pd.DataFrame(json.load(open(D / "jobs.json")))
Q = pd.DataFrame(json.load(open(D / "sql.json")))
res = {"jobs": len(J), "etapas_completas": len(S), "etapas_saltadas": int(J.numSkippedStages.sum()), "tareas": E["totalTasks"],
       "tareas_fallidas": E["failedTasks"], "nucleos": E["totalCores"], "tiempo_tareas_s": E["totalDuration"] / 1e3, "gc_s": E["totalGCTime"] / 1e3,
       "entrada_gb": E["totalInputBytes"] / 2 ** 30, "shuffle_lect_mb": E["totalShuffleRead"] / MB, "shuffle_escr_mb": E["totalShuffleWrite"] / MB,
       "cache_bloques": E["rddBlocks"], "cache_mem_mb": E["memoryUsed"] / MB, "cache_disco_mb": E["diskUsed"] / MB, "heap_almacen_gb": E["maxMemory"] / 2 ** 30,
       "consultas_sql": len(Q), "spill_disco_mb": S.diskBytesSpilled.sum() / MB}
pd.Series(res).to_csv(RES / "bench" / "spark_ui_resumen.csv", header=["valor"])
print(pd.Series(res).round(2).to_string())
print(S.groupby("fase")[["dur_s", "numTasks"]].sum().round(1).to_string())
print(G.head(10)[["fase", "llamada", "etapas", "tareas", "dur_s", "cpu_s", "gc_s", "shuffle_escr_mb"]].round(1).to_string())
