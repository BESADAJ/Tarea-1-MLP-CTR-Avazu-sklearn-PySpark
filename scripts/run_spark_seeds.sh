#!/bin/bash
# Repeticiones por semilla con muestras. Validación completa salvo hashing denso.
# point: n enc model layers maxIter solver step seeds valid_frac block exp limite
PY=${PY:-python}   # intérprete del entorno ml_venv
L=results/spark/seeds.log
run() { echo "=== $*" >> $L; $PY -u scripts/04_spark_experiments.py point "$@" 2>&1 | grep --line-buffered -E "prep=|status|Error:" >> $L; }
for n in 100000 1000000; do
  run $n freq mlp 128x64 200 l-bfgs 0.1 5 - 128 scale 3000
  run $n freq mlp 50 100 l-bfgs 0.1 5 - 128 scale 3000
  run $n hash9 mlp 128x64 200 l-bfgs 0.1 5 0.1 128 scale 3000
  run $n hash12 lr - 100 l-bfgs 0.1 1 - 128 scale 3000
  run $n gbt gbt - 50 l-bfgs 0.1 5 - 128 scale 3000
done
echo SEEDS_DONE >> $L
