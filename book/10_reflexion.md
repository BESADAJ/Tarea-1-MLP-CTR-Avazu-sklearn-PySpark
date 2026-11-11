# 10. Reflexión crítica

Cada respuesta remite a la evidencia del capítulo indicado.

## ¿Qué entorno fue más rápido, en qué rango de volumen y por qué?

Las cifras vienen del benchmark limpio del capítulo 7: un proceso nuevo por punto y el equipo dedicado.

- **Hasta 1 millón de filas, scikit-learn es más rápido en todo.** Spark arrastra unos 45 s fijos de preparación (arranque de la
  JVM, planificación y 23 agregaciones). Con 100 mil filas, el proceso completo tarda 3 a 4 s en scikit-learn y 50 a 57 s en
  Spark.
- **Con el modelo lineal, Spark gana desde unos 5 millones de filas sin perder calidad.** Su tiempo crece con exponente 0,2 y el
  de scikit-learn con 0,9 a 1,15, porque SGD corre en un solo hilo. Con 10 millones, el ajuste es 8,7 veces más rápido (21,6 s
  frente a 187,8 s) y el total 1,4 veces (156 s frente a 225 s), con el mismo AUC (0,731 frente a 0,730).
- **Con el MLP, Spark es más rápido en el ajuste desde unas 850 mil filas, pero no es comparable:** pierde 0,07 a 0,08 de AUC en
  todos los volúmenes, y en tiempo total ni siquiera gana con 10 millones.
- **En predicción y lectura gana siempre la máquina sola.** scikit-learn puntúa de 2 a 8 millones de filas por segundo, frente a
  55 a 95 mil de Spark. DuckDB y Polars agregan el Parquet en 0,1 y 0,3 s, frente a 6,5 s de Spark.
- **El paralelismo tiene techo en un portátil.** Amdahl da f = 0,23 para la LR de Spark, con una aceleración máxima de 4,3, y f
  cercano a 0,1 para el MLP de Spark y HGB. De 4 a 8 núcleos casi no se gana, porque los 4 últimos son de eficiencia.

## ¿Cuál fue más preciso y calibrado, y es significativa la diferencia?

En la prueba (día 30):

| Modelo | ROC AUC | ECE |
|---|---|---|
| HGB con target encoding, scikit-learn, 10 M | 0,733 | 0,012 |
| Regresión logística de Spark, hash12, 32,4 M | 0,7255 | 0,010 |
| SGD logístico con hash12, scikit-learn, 10 M | 0,721 | 0,015 |
| MLP `(100,)` con hash12, scikit-learn, 10 M | 0,717 | 0,029 |
| GBT de Spark, frecuencia, 32,4 M, 20 árboles | 0,698 | 0,026 |
| MLP de Spark, hash 2^9, 1 M | 0,698 | 0,015 |
| MLP de Spark, frecuencia, 32,4 M | 0,639 | 0,022 |

**P3 tiene respuesta negativa:** el MLP no mejora a las líneas base simples. Pierde frente a HGB por 0,016 de AUC, como ya pasaba en
validación (0,739 frente a 0,743). En la prueba pierde también frente a SGD, aunque en validación le ganaba (0,739 frente a 0,728). Con 4,2 millones de filas, DeLong y McNemar marcan significativas todas
las diferencias tras Holm. La magnitud importa más: entre SGD y el MLP la distancia es menor que la desviación del MLP entre
semillas (0,0065).

El MLP es además el peor calibrado. Su caída entre validación y prueba (0,739 a 0,717, es decir 0,022) es la mayor de los tres,
frente a 0,009 de HGB y 0,007 de SGD: se ajusta a
interacciones de los días 21 a 28 que el día 30 ya no cumple. El EDA lo anticipaba: `C14`, `C17` y `C21` tienen PSI mayor que 0,5.

## ¿Cuándo se justifica PySpark y cuándo es sobrecosto?

En una sola máquina con 32 GB, Avazu completo no necesita Spark: entra en memoria como CSR (hash12) o como tabla densa
(frecuencia, target). Se justifica en estos casos:

- Cuando los datos no caben: Criteo 1TB, o varios días de Avazu sin muestrear.
- Cuando el modelo es lineal y disperso, donde Spark escala casi sin costo adicional.
- Cuando se necesita tolerancia a fallos o integración con un lago de datos ya en Spark.

Es sobrecosto con 1 millón de filas o menos, y para el MLP en cualquier volumen.

## ¿Qué limitaciones impone el MLP nativo de Spark frente a una red con embeddings?

La más grave no está en la lista del enunciado: **densifica la entrada** (capítulo 5.1). Con `d = 2^12` falla con 300 mil filas,
menos del 1 % de los datos. Eso obliga a escoger entre poca identidad categórica (frecuencia, `d = 27`) o pocas filas (hash 2^9).
Una red con embeddings tiene costo de memoria proporcional a las categorías vistas, no a `n x d`, y entrenaría con todos los datos.
Además, el MLP de Spark no tiene dropout ni L2, ni pesos de clase, usa sigmoide y el solver `gd` no converge en la práctica.

## ¿Qué aporta LIME, qué no puede garantizar y cómo se compara con SHAP?

LIME aporta una explicación rápida en el lenguaje de las variables originales, aunque el modelo use hashing. No garantiza
fidelidad: el sustituto local explica cerca del 30 % de la variación. Tampoco estabilidad, que exige 15.000 muestras para llegar a
un Jaccard de 0,87. Y no coincide con SHAP: 19 % de variables comunes en el top 5. SHAP y la permutación sí coinciden entre sí
(Spearman 0,93), y apuntan al contexto de publicación: `site_category`, `app_id`, `site_domain` y `site_id`.

## Amenazas a la validez

| Amenaza | Qué se hizo | Qué queda |
|---|---|---|
| Deriva temporal | Partición por día y PSI | Un solo día de prueba, un jueves. Otro día podría ordenar distinto los modelos |
| Fuga de información | Ajustes solo con entrenamiento, target por día fuera de pliegue, prueba bloqueada | La parada temprana del MLP usa un 10 % aleatorio interno |
| Anonimización | Se respetan las variables tal como vienen | No se puede interpretar `C14` a `C21` ni validar su semántica |
| Sesgo de muestreo | Muestras por hash de `id`, anidadas | Hiperparámetros elegidos con 1 millón y usados con 10 millones |
| Selección | Todas las decisiones en el día 29 | hash12 y `(100,)` se eligieron por presupuesto o parsimonia, con pérdidas pequeñas documentadas |
| Entorno | Mediciones en proceso limpio | Windows, modo local y un portátil: los tiempos de Spark no representan un clúster |
| Generalización entre fuentes | No hecho | Sin Criteo, por disco (capítulo 1) |
