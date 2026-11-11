# 12. Correspondencia con los requisitos

Cada requisito del punto 2.12, con el capítulo donde se desarrolla y la forma en que se atendió. Las desviaciones respecto al
enunciado se indican con su justificación.

| Requisito | Capítulo | Cómo se atiende |
|---|---|---|
| **Dataset y ficha** | | |
| Criteo como principal, Avazu como secundario, o Avazu completo sin clúster | 1 | Avazu completo, la alternativa sin clúster: Criteo 1TB no cabe en el disco disponible |
| Ficha: origen, licencia, período, tasa base, disco, memoria, limitaciones | 1 | Ficha completa, con procedencia verificada por SHA-256 y MD5 |
| **Protocolo** | | |
| División temporal: 21 a 28, 29 y 30 | 3 | Partición por día; la prueba está bloqueada por código salvo en la evaluación final |
| Transformaciones ajustadas solo con entrenamiento | 3 | Vocabulario, raras, conteos, escalas y tablas de target se ajustan con los días 21 a 28 |
| Target encoding fuera de pliegue | 3 y 4 | Pliegues por día |
| Al menos 5 semillas, media, desviación e IC | 4 a 6 | Semillas 42 a 46. Los modelos de Spark con 32,4 millones de filas tienen una sola corrida, por costo |
| Métricas con y sin ponderación; umbral en validación | 4 y 6 | Umbral por F1 en el día 29, aplicado sin cambios en el día 30 |
| **EDA** | | |
| Parquet particionado y lectura perezosa (Polars, DuckDB) | 2 | Parquet por día con ZSTD, leído en modo perezoso con Polars y DuckDB |
| Click y tasa base por día, hora y franja | 2 | Tablas y figuras por día, hora y franja horaria |
| Variables temporales derivadas | 2 y 3 | Hora del día, día de la semana y codificación cíclica |
| Cardinalidad, cola larga y categorías únicas | 2 | Cardinalidad por columna, curva de cobertura acumulada y categorías vistas una sola vez |
| Información mutua, V de Cramér, Spearman | 2 | Las tres medidas, sobre el 5 % del entrenamiento |
| PSI entre entrenamiento y prueba | 2 | PSI por columna; `C14`, `C17` y `C21` superan 0,5 |
| Visualizaciones requeridas | 2 | Incluidas en el capítulo |
| **Preprocesamiento** | | |
| `__RARE__` con umbral en entrenamiento | 3 | Umbral de 10 apariciones en los días 21 a 28 |
| Hashing 2^12, 2^14 y 2^16; frecuencia; target OOF | 4 | Las cinco codificaciones, comparadas con el mismo MLP |
| log y StandardScaler en numéricas de cola pesada | 3 | log1p de la actividad de `device_ip` y `device_id`, con escalado |
| Cruces de variables como ablación | 4 | Ablación con y sin cruces |
| **scikit-learn** | | |
| 100 mil, 1 M y 10 M | 4 | Curvas de aprendizaje en los tres tamaños |
| SGDClassifier logístico y HistGradientBoosting | 4 | Ambas líneas base, con 5 semillas |
| MLPClassifier con `early_stopping=True` | 3 y 4 | Usado como pide el enunciado; su partición interna aleatoria se discute en el capítulo 3 |
| Optuna con validación temporal y el espacio indicado | 4 | 30 intentos evaluados en el día 29 |
| **PySpark** | | |
| CSV a Parquet por día con SparkSession | 1 y 7 | Spark lee el CSV y ejecuta la transformación completa; el archivo lo escribe DuckDB, porque Spark en Windows requiere `winutils.exe` para escribir |
| FeatureHasher, StringIndexer, VectorAssembler, StandardScaler | 3 y 5 | Pipelines de Spark ML sin UDF de Python |
| LogisticRegression, GBTClassifier, MultilayerPerceptronClassifier | 5 y 6 | Los tres, evaluados en validación y en la prueba final |
| ParamGridBuilder con la rejilla indicada | 5 | Rejilla completa con frecuencia; reducida con hashing, por memoria |
| `d` igual a la dimensión del hashing | 5 | Inviable con 2^12 o más: el clasificador densifica la entrada. Se demuestra con los errores de memoria medidos y con el código fuente |
| BinaryClassificationEvaluator | 5 | Usado en todas las evaluaciones de Spark |
| Limitaciones del MLP de Spark | 5 | Sección 5.4 |
| TorchDistributor (opcional) | 5 | No incluido: el entorno no tiene PyTorch y los Python workers de Spark no funcionan en este equipo |
| **Métricas e inferencia** | | |
| ROC AUC, PR AUC, log loss, Brier, ECE, calibración | 6 | Tabla de métricas y curvas de calibración |
| Accuracy, precision, recall, F1, matriz de confusión | 6 | Con el umbral fijado en validación |
| Lift y ganancia acumulada | 6 | Curvas y tabla por percentil |
| Bootstrap, DeLong y McNemar | 6 | Con corrección de Holm |
| **Benchmarking** | | |
| Lectura, preprocesamiento, entrenamiento y predicción por separado | 7 | 43 puntos, cada uno en un proceso nuevo, con el equipo dedicado |
| Escalamiento fuerte y débil, aceleración, eficiencia, Amdahl | 7 | 2 millones de filas fijas y 500 mil por núcleo, con 1 a 8 núcleos; fracción secuencial ajustada |
| Punto de cruce | 7 | Cruce por ajuste log-log, contrastado con los puntos medidos y con control de hilos BLAS |
| Spark UI y costo en la nube | 7 | Métricas de Jobs, Stages y Executors tomadas de la API REST de la UI; costo con precios de lista de AWS |
| **Interpretabilidad** | | |
| LimeTabularExplainer sobre FP y FN de mayor confianza errónea | 8 | Sobre el día 29 |
| Espacio interpretable con correspondencia documentada | 8 | Explicaciones sobre las variables originales, con la función de correspondencia al hashing |
| Fidelidad, estabilidad (Jaccard) y sensibilidad | 8 | Las tres, con varios tamaños de muestra |
| Permutación y SHAP, coherencia entre métodos | 8 | Comparación por Spearman y solapamiento del top 5 |
| **Cierre** | | |
| Tabla comparativa por entorno, modelo y codificación | 9 | Tabla y barras de tiempo y memoria |
| Ablaciones: hashing, cruces, profundidad y ponderación | 4 | Las cuatro ablaciones |
| Reflexión crítica con evidencia | 10 | Cada respuesta remite a su capítulo |
| Reproducibilidad: versiones, semillas, Spark, hardware, manifiesto | 11 | Anexo completo, con la secuencia de ejecución de extremo a extremo |
