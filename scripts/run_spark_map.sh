#!/bin/bash
# Mapa de factibilidad del MLP de Spark con hashing, y rejillas. Cada punto en su propio proceso.
PY=${PY:-python}   # intérprete del entorno ml_venv
L=results/spark/map.log
for cfg in "100000 hash12" "30000 hash14" "300000 hash12" "100000 hash14" "1000000 hash12"; do
  set -- $cfg
  echo "=== $cfg" >> $L
  $PY -u scripts/04_spark_experiments.py point $1 $2 mlp 100 10 l-bfgs 0.03 1 0.1 128 map 1200 2>&1 | grep -E "prep=|status|Error:" >> $L
done
echo "=== grid freq" >> $L
$PY -u scripts/04_spark_experiments.py grid 200000 freq 0.1 completa 2>&1 | grep -E "^[0-9]+ |Error:" >> $L
echo "=== grid hash10" >> $L
$PY -u scripts/04_spark_experiments.py grid 100000 hash10 0.1 lbfgs 2>&1 | grep -E "^[0-9]+ |Error:" >> $L
echo MAP_DONE >> $L
