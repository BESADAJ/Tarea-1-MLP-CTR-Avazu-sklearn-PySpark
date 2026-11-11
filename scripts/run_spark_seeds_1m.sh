#!/bin/bash
# Repeticiones restantes y corridas finales de Spark, ordenadas por prioridad y costo.
# save: guarda predicciones de validación y prueba para 07_final_test.py, sin calcular métricas.
PY=${PY:-python}   # intérprete del entorno ml_venv
L=results/spark/seeds.log
run() { echo "=== $*" >> $L; $PY -u scripts/04_spark_experiments.py point "$@" 2>&1 | grep --line-buffered -E "prep=|status|Error:" >> $L; }
run 1000000 hash12 lr - 100 l-bfgs 0.1 1 - 128 scale 3000
run 1000000 gbt gbt - 50 l-bfgs 0.1 2 - 128 scale 3000
run 1000000 hash9 mlp 128x64 50 l-bfgs 0.1 1 - 128 scale 3000 save
run full hash12 lr - 30 l-bfgs 0.1 1 - 128 full 3600 save
run full gbt gbt - 20 l-bfgs 0.1 1 - 128 full 3600 save
run full freq mlp 50 50 l-bfgs 0.1 1 - 128 full 3600 save
run 100000 hash9 mlp 128x64 100 l-bfgs 0.1 5 - 128 scale 3000
echo SEEDS_DONE >> $L
