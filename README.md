# Tarea 1 — MLP para predecir clics en Avazu: scikit-learn frente a PySpark

**Proyecto integrador de Deep Learning (punto 2.12, Multilayer Perceptron): clasificadores MLP que predicen si una impresión de un anuncio móvil termina en clic, comparando scikit-learn y PySpark bajo un protocolo temporal sin fuga de información, con benchmarking de escalamiento e interpretabilidad con LIME, SHAP y permutación.**

📖 **Libro publicado:** https://besadaj.github.io/Tarea-1-MLP-CTR-Avazu-sklearn-PySpark/

**Integrantes:** Jaime Andrés Besada, Ivan Berrio

---

## Descripción

Con Avazu CTR Prediction completo (40.428.967 impresiones del 21 al 30 de octubre de 2014) se entrena con los días 21 a 28, se
valida con el 29 y se evalúa una sola vez en el 30. Se comparan cinco codificaciones de las categóricas (hashing 2^12, 2^14 y
2^16, frecuencia y target encoding fuera de pliegue), MLP y líneas base en scikit-learn y en PySpark, y el costo computacional de
cada motor según el volumen de datos y el número de núcleos.

## Resultados principales

**Prueba final (día 30, 4,2 millones de filas):**

| Modelo | Entorno | Entrenamiento | ROC AUC | ECE |
|---|---|---|---|---|
| HistGradientBoosting, target encoding | scikit-learn | 10 M | **0,733** | 0,012 |
| Regresión logística, hash 2^12 | PySpark | 32,4 M | 0,7255 | 0,010 |
| SGD logístico, hash 2^12 | scikit-learn | 10 M | 0,721 | 0,015 |
| MLP `(100,)`, hash 2^12 | scikit-learn | 10 M | 0,717 | 0,029 |
| GBT, frecuencia | PySpark | 32,4 M | 0,698 | 0,026 |
| MLP `[516, 128, 64, 2]`, hash 2^9 | PySpark | 1 M | 0,698 | 0,015 |
| MLP `[27, 50, 2]`, frecuencia | PySpark | 32,4 M | 0,639 | 0,022 |

- **P1, velocidad.** Con el modelo lineal, PySpark supera a scikit-learn desde unos 5 millones de filas sin perder calidad
  (156 s frente a 225 s con 10 millones). Con el MLP no hay volumen en que Spark gane sin perder AUC. Hasta 1 millón de filas,
  scikit-learn es más rápido en todo.
- **P2, codificación.** Para el MLP, el hashing supera a la frecuencia (0,738 frente a 0,680 en validación). Pasar de 2^12 a
  2^16 gana 0,002 de AUC con 12 veces el costo.
- **P3, MLP frente a líneas base.** El MLP no las mejora: queda detrás de HGB, de la regresión logística y de SGD, y es el peor
  calibrado y el más inestable entre semillas.
- **P4, LIME.** Poco fiel (R² local cercano a 0,3) y coincide con SHAP en el 19 % del top 5. SHAP y la permutación sí coinciden
  (Spearman 0,93).
- **Hallazgo de ingeniería:** `MultilayerPerceptronClassifier` de Spark densifica la entrada; con hashing de 2^12 se queda sin
  memoria con 300 mil filas, menos del 1 % de los datos.

## Estructura del repositorio

```
book/          Jupyter Book: capítulos en Markdown y notebooks ejecutados
nb_src/        fuentes de los notebooks (formato percent), convertidas con scripts/build_notebooks.py
src/ctr/
  config.py      rutas, semillas, días y columnas
  data.py        carga perezosa con Polars, muestras anidadas y bloqueo de la prueba
  encoders.py    raras, hashing, frecuencia y target encoding fuera de pliegue
  sparkpipe.py   la misma lógica en Spark, sin UDF
  sparkutil.py   sesión local de Spark y lectura con rutas exactas
  metrics.py     métricas, calibración, lift, DeLong, McNemar y bootstrap
  monitor.py     tiempo de pared y memoria pico, incluida la JVM
  plots.py       estilo de figuras
scripts/       experimentos: 01_ingest.py a 07_final_test.py y secuencias run_*.sh
results/       tablas y registros de cada ajuste (sk/runs.jsonl, spark/runs.jsonl, bench/points.jsonl)
```

## Datos

Avazu se obtuvo de la réplica pública `reczoo/Avazu_x4` en Hugging Face, verificada con SHA-256 y MD5 (capítulo 1 del libro). Los
datos no se versionan: se colocan en `data/raw` y `scripts/01_ingest.py` los convierte a Parquet por día.

## Reproducir

```bash
conda env create -f environment.yml
conda activate ml_venv
```

La secuencia completa de experimentos está en el capítulo 11 del libro. Para reconstruir el libro desde los resultados guardados,
sin repetir experimentos:

```bash
python scripts/build_notebooks.py
jupyter-book build book
ghp-import -n -p -f book/_build/html
```

Entorno: Python 3.9.25, PySpark 4.0.1 y Java 21, en Windows 11 con Spark en modo local.
