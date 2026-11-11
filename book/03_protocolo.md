# 3. Protocolo experimental y preprocesamiento

## Partición temporal

| Conjunto | Días | Filas | Uso |
|---|---|---|---|
| Entrenamiento | 21 a 28 | 32.377.421 | Ajustar modelos, codificadores y escaladores |
| Validación | 29 | 3.832.608 | Elegir hiperparámetros, codificación y umbral |
| Prueba | 30 | 4.218.938 | Medición final, una sola vez |

La prueba está bloqueada por código: `ctr.data.load("test")` falla salvo que la llame `scripts/07_final_test.py`.
Las corridas de Spark guardan sus predicciones sobre el día 30 sin calcular ninguna métrica.

## Reglas contra la fuga de información

1. **Todo se ajusta con entrenamiento:** vocabulario, umbral de raras, conteos, tablas de target encoding y
   escaladores.
2. **Las categorías no vistas** caen en `__RARE__`.
3. **Target encoding fuera de pliegue:** cada fila de entrenamiento recibe una media calculada sin su día. Es
   anidado porque la selección ocurre afuera, sobre el día 29. Una fila nunca se codifica con su propia etiqueta.
4. **Búsqueda de hiperparámetros temporal:** Optuna evalúa cada intento sobre el día 29. No hay
   `GridSearchCV` con particiones aleatorias.
5. **Muestras anidadas:** `sample_key` es un hash de `id`, guardado en el Parquet. Filtrarlo en `sample_key < f` da
   la misma muestra en Polars y en Spark, y cada tamaño contiene al anterior.

Queda una excepción conocida. `early_stopping=True` en `MLPClassifier` aparta un 10 % **aleatorio** del
entrenamiento y detiene por **exactitud**. El enunciado pide esa opción, y aquí se cumple, pero tiene dos costos:
esa partición interna no es temporal, y con clases desbalanceadas la exactitud se estanca pronto. Se ve en los
resultados: casi todos los MLP paran entre la época 5 y la 8. La parada solo elige la época: el modelo y los
hiperparámetros se juzgan con el día 29.

## Lógica común en ambos entornos

| Paso | scikit-learn | PySpark |
|---|---|---|
| Lectura | Polars perezoso sobre Parquet | `read.parquet` con rutas exactas |
| Raras (menos de 10 apariciones) | Tabla de conteos y join | Tabla de conteos y `broadcast join` |
| Hashing | `FeatureHasher` sobre el vocabulario | `FeatureHasher` |
| Frecuencia | `log1p` del conteo | `log1p` del conteo |
| Target encoding | Fuera de pliegue por día | No se hizo, ver abajo |
| Numéricas | `log1p` de actividad de `device_ip` y `device_id`, seno y coseno de la hora | Igual |
| Escalado | `StandardScaler` | `StandardScaler` |
| Baja cardinalidad en árboles | Frecuencia | `StringIndexer` |

Las funciones de hash no son idénticas: scikit-learn y Spark usan variantes distintas de MurmurHash3. Las
colisiones concretas cambian, pero la lógica es la misma.

**Target encoding solo en scikit-learn.** El enunciado lo restringe así. Replicarlo en Spark sin UDF exigiría una
agregación por pliegue y variable, con un join por combinación, y la fuga por error de implementación es más
fácil de cometer que de detectar.

**Variables temporales.** De `hour` salen `day` (solo para partir, nunca como variable), `hod` y `dow`.

**Por qué sin UDF en Spark.** Los Python workers de PySpark 4.0.1 no arrancan en este Windows: falla la lectura
de un recurso dentro de `pyspark.zip`. El pipeline se escribió entero con operadores de la JVM. Es una restricción
del entorno, y además un buen hábito: una UDF de Python en Spark serializa cada fila.

## Semillas y repeticiones

Las semillas son 42 a 46. Cambian la inicialización y el orden de los lotes, no la muestra. Las tablas reportan
media, desviación e intervalo t al 95 % con 5 repeticiones. Las excepciones se indican donde aparecen: el MLP de
Spark con 32 millones de filas se entrena una sola vez, por costo.

## Desbalance

La tasa base ronda el 17 %. `MLPClassifier` no acepta pesos de clase, y `MultilayerPerceptronClassifier` tampoco. La
ponderación se hace así: en SGD y HGB con `class_weight="balanced"`, y en el MLP repitiendo positivos hasta
igualar clases. Ese experimento está en las ablaciones del capítulo 4. El umbral de decisión se elige en validación
maximizando F1 y se aplica sin cambios en la prueba.
