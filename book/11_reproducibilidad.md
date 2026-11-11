# 11. Anexo de reproducibilidad

## Entorno

| Elemento | Valor |
|---|---|
| Sistema | Windows 11 Home Single Language 10.0.26200 |
| CPU | Intel Core i5-13420H, 8 núcleos físicos (4 P y 4 E), 12 hilos |
| RAM | 31,7 GB, de los cuales unos 13 GB estaban ocupados por otras aplicaciones |
| GPU | NVIDIA RTX 3050, sin uso: ningún modelo la aprovecha |
| Disco libre | 13 a 19 GB durante el trabajo |
| Python | 3.9.25 en el entorno conda `ml_venv` |
| Java | Oracle JDK 21.0.10 |
| Spark | 4.0.1, modo `local[p]`, driver de 14 GB |

Las versiones exactas están en `environment.yml` y `requirements.txt`. Sobre `ml_venv` se instalaron `polars==1.36.1` y
`duckdb==1.4.5`, usados en la ingesta y el EDA.

## Configuración de Spark

| Parámetro | Valor | Razón |
|---|---|---|
| `spark.master` | `local[p]`, con `p` entre 1 y 8 | Sin clúster |
| `spark.driver.memory` | `14g` | Lo que deja libre el equipo |
| `spark.sql.shuffle.partitions` | `max(8, 4p)` | Evita las 200 particiones por defecto en modo local |
| `spark.sql.adaptive.enabled` | `true` | Ajusta particiones en ejecución |
| `spark.sql.autoBroadcastJoinThreshold` | `2g` | Las tablas de conteo van por difusión |
| `spark.rdd.compress` | `true` | Reduce lo que se derrama a disco |
| `spark.driver.maxResultSize` | `4g` | Devuelve predicciones a Python |
| `spark.sql.execution.arrow.pyspark.enabled` | `true` | Traspaso rápido a pandas, sin Python workers |

## Semillas

Semilla global 42. Repeticiones con 42, 43, 44, 45 y 46. `sample_key` es un hash de `id` con semilla 42.

## Manifiesto de datos

| Archivo | Tamaño | Verificación |
|---|---|---|
| `Avazu_x4.zip`, de `huggingface.co/datasets/reczoo/Avazu_x4` | 1.306.331.539 bytes | SHA-256 `4ba8bf46eada70abbe068e4e5bfb3b3d879e6fbd2f2fcf51760cd1388b789b65` |
| `train.csv` | 5.016.574.514 bytes | MD5 `de3a27264cdabf66adf09df82328ccaa` |
| `valid.csv` | 627.068.074 bytes | MD5 `33232931d84d6452d3f956e936cab2c9` |
| `test.csv` | 627.076.596 bytes | MD5 `3ebb774a9ca74d05919b84a3d402986d` |

Los datos no se versionan: `data/` está en `.gitignore`.

## Ejecución de extremo a extremo

```
conda env create -f environment.yml
conda activate ml_venv

# 1. Datos: colocar Avazu_x4.zip en data/raw y extraer los tres CSV allí
python scripts/01_ingest.py

# 2. EDA
python scripts/02_eda.py

# 3. scikit-learn
python scripts/03_sk_experiments.py enc 6
python scripts/03_sk_experiments.py tune 6 5
python scripts/03_sk_experiments.py scale 6 100000,1000000,10000000
python scripts/03_sk_experiments.py abl 6

# 4. Spark
bash scripts/run_spark_map.sh
bash scripts/run_spark_seeds_1m.sh

# 5. Benchmark, sin otras cargas en el equipo
bash scripts/run_bench.sh
python scripts/05b_spark_ui.py   # resume la API REST de Spark UI guardada por el punto spark_ui

# 6. Interpretabilidad y prueba final
python scripts/06_interpret.py 10000000
python scripts/07_final_test.py 10000000

# 7. Libro
python scripts/build_notebooks.py
jupyter-book build book
```

`scripts/build_notebooks.py` genera `book/*.ipynb` desde `nb_src/*.py`, y los capítulos leen las tablas de `results/`. Por eso el
libro se puede reconstruir sin repetir los experimentos. `results/sk/runs.jsonl` permite reanudar: los ajustes ya registrados no
se repiten.

Las ablaciones, la búsqueda de Optuna y la mayor parte de los ajustes con semillas corrieron en paralelo y tardan horas. Los
tiempos de los capítulos 4 y 5 dependen de esa carga. Solo los del capítulo 7 son mediciones limpias.

## Particularidades de Windows

| Problema | Causa | Solución adoptada |
|---|---|---|
| Spark no escribe Parquet | Falta `winutils.exe` y `HADOOP_HOME` | DuckDB genera el Parquet. No se instaló el binario de terceros |
| Spark no lista directorios | Falta la librería nativa de Hadoop | `read_avazu()` pasa rutas exactas de archivo, con `basePath` |
| Python workers fallan | PySpark 4.0.1 no lee `error-conditions.json` dentro de `pyspark.zip` | Sin UDF, y sin `createDataFrame` desde Python |
| No se guardan modelos Spark | Mismo problema de escritura | Las predicciones se devuelven por Arrow a `data/final` |

En Linux, estos cuatro problemas no aplican.
