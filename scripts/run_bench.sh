#!/bin/bash
# Benchmark completo, sin otras cargas en el equipo. Reanudable: salta puntos ya presentes.
PY=${PY:-python}   # intérprete del entorno ml_venv
OUT=results/bench/points.jsonl
L=results/bench/bench.log
mkdir -p results/bench; touch $OUT
b() { # motor modelo n nucleos exp
  K="$1 $2 $3 $4 $5"
  grep -qxF "$K" results/bench/done.txt 2>/dev/null && return
  BENCH_THREADS=$BENCH_THREADS $PY -u scripts/05_benchmark.py $1 $2 $3 $4 $5 2>&1 | grep -E "^(sk|spark) |Error" >> $L
  echo "$K" >> results/bench/done.txt
}
# Punto de cruce: mismo trabajo, 8 núcleos para Spark, 1 para scikit-learn (no paraleliza estos modelos)
for n in 100000 1000000 3000000 10000000; do
  b spark lr $n 8 cruce; b sk lr $n 1 cruce
  b spark mlp $n 8 cruce; b sk mlp $n 1 cruce
done
# Escalamiento fuerte: 2 M fijos
for p in 1 2 4 8; do b spark lr 2000000 $p fuerte; b spark mlp 2000000 $p fuerte; b sk hgb 2000000 $p fuerte; done
# Escalamiento débil: 500 mil por núcleo
for p in 1 2 4 8; do n=$((500000*p)); b spark lr $n $p debil; b spark mlp $n $p debil; b sk hgb $n $p debil; done
# Control: MLP de scikit-learn con 8 hilos BLAS (justifica usar 1 hilo en el cruce)
for n in 3000000 10000000; do BENCH_THREADS=8 b sk mlp $n 8 blas8; done
# Evidencia de Spark UI: registro de eventos en un punto representativo
b spark mlp 3000000 8 spark_ui
echo BENCH_DONE >> $L
