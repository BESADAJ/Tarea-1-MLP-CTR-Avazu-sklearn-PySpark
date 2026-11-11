# 1. Ficha de datos

## Decisión sobre el dataset

El enunciado recomienda Criteo 1TB como principal y Avazu como secundario, y deja una tercera vía:
"Alternativa sin clúster: usar Avazu completo y tratar la pregunta como un experimento de punto de cruce,
sin presuponer que Spark ganará". Este trabajo sigue esa vía.

La razón es de capacidad. Cada día de Criteo 1TB ocupa cerca de 46 GB sin comprimir, y este equipo tiene
unos 19 GB libres en disco. Tampoco hay clúster. Avazu cabe completo y responde las preguntas P1 a P4.

**Lo que se pierde:** no hay un segundo dataset para evaluar si las conclusiones se generalizan entre
fuentes. Esto queda como una limitación explícita en el capítulo 10.

## Ficha

| Campo | Valor |
|---|---|
| Nombre | Avazu Click-Through Rate Prediction |
| Origen | Competencia de Kaggle `avazu-ctr-prediction`, publicada por Avazu |
| Obtención en este trabajo | Réplica pública `reczoo/Avazu_x4` en Hugging Face (ver abajo) |
| Licencia | Reglas de la competencia de Kaggle. La réplica no declara una licencia propia. Uso académico, sin redistribuir los datos |
| Período | 21 al 30 de octubre de 2014, en hora UTC (10 días con etiqueta) |
| Tamaño | 40.428.967 filas, 24 columnas |
| Disco | 6,3 GB en CSV, 1,22 GB en Parquet con ZSTD, particionado por día |
| Memoria | Unos 13 GB al convertir con DuckDB. Una carga completa en pandas no se intentó |
| Tasa base de clics | 16,9 % en los días 21 a 29 |
| Objetivo | `click` (0 o 1) |
| Variables | 21 categóricas anonimizadas o con hash, más `hour` |

## Procedencia y verificación

Kaggle exige credenciales y aceptar las reglas de la competencia, y la descarga anónima devuelve 401. Se
usó entonces una réplica sin restricción de acceso, mantenida por el proyecto RecZoo (BARS).

Esa réplica reparte las filas al azar en `train.csv`, `valid.csv` y `test.csv`, en proporción 8:1:1. Esos
archivos **no** son la partición del proyecto: se unen y se vuelven a separar por día. Conservan `id` y
`hour` sin procesar.

| Verificación | Resultado |
|---|---|
| SHA-256 del zip contra el del repositorio | `4ba8bf46…789b65`, coincide |
| MD5 de `train.csv`, `valid.csv` y `test.csv` contra los publicados | Los tres coinciden |
| Filas totales | 40.428.967, la cifra publicada de la competencia |
| `id` duplicados por día | Ninguno |

## Limitaciones

- **Anonimización:** `C1` y `C14` a `C21` no tienen significado público, y los identificadores vienen con
  hash. La interpretación de cualquier explicación llega hasta ese límite.
- **Ventana corta:** diez días no permiten estacionalidad semanal robusta. Solo hay un jueves en
  entrenamiento (día 23), y la prueba es otro jueves (día 30).
- **Sesgo de selección:** son anuncios que sí se mostraron, según la política de la plataforma en 2014.
- **`device_id`:** el valor más común cubre el 82,6 % de las filas y actúa como dato ausente.
- **Prueba original:** el día 31 de Kaggle no trae etiquetas y no se usa. La prueba aquí es el día 30.

## Por qué no se convirtió con SparkSession

El enunciado pide leer el CSV y convertirlo a Parquet con SparkSession. En este equipo, Spark sobre Windows
no puede escribir Parquet sin `winutils.exe`, un binario de terceros que no se instaló. La conversión la hace
DuckDB, y Spark ejecuta la misma transformación desde CSV hacia el destino `noop` para medir su costo
(capítulo 7). En un equipo Linux, la conversión con Spark funcionaría sin ese paso.
