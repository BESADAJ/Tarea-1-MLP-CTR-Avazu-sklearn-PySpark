# Predicción de clics en Avazu con MLP: scikit-learn frente a PySpark

Proyecto integrador del punto 2.12 del curso de Deep Learning.

**Integrantes:** Jaime Andrés Besada e Ivan Berrio.

## Objetivo

Construir, evaluar y comparar clasificadores MLP que predicen si una impresión de un anuncio móvil termina en clic (`click = 1`). Se
usan scikit-learn y PySpark, bajo un protocolo temporal sin fuga de información, y se analiza con LIME, contrastado con SHAP, qué
explica el modelo.

## Datos

Avazu, 40,4 millones de impresiones del 21 al 30 de octubre de 2014. El enunciado permite usar Avazu completo sin clúster, y así se
hizo: Criteo 1TB no cabía en el disco disponible (capítulo 1).

## Resultados principales

| Pregunta | Respuesta corta | Capítulo |
|---|---|---|
| **P1.** ¿Desde qué volumen gana Spark en tiempo sin perder calidad? | Con el modelo lineal, desde unos 5 millones de filas contando lectura y preprocesamiento, y desde unas 700 mil en el ajuste solo, con el mismo AUC. Con 10 millones, Spark tarda 156 s y scikit-learn 225 s. Con el MLP, Spark nunca gana sin perder calidad: ajusta más rápido desde unas 850 mil filas, pero queda 0,07 de AUC por debajo | 7 |
| **P2.** ¿Cómo afecta la codificación? | Para el MLP, el hashing gana con claridad a la frecuencia (0,738 frente a 0,680), contra lo que suponía H2. De 2^12 a 2^16 se gana solo 0,002, con 12 veces el costo. Target encoding es lo mejor para árboles | 4 |
| **P3.** ¿Mejora el MLP a las líneas base? | No. En la prueba, HGB con target encoding obtiene 0,733 de AUC, la regresión logística de Spark 0,7255, SGD 0,721 y el MLP 0,717. El MLP es además el peor calibrado y el que más cae entre validación y prueba | 6 |
| **P4.** ¿Son LIME estable y fiel? | Poco fiel (R² cerca de 0,3) y estable solo con muchas muestras. Coincide con SHAP en el 19 % del top 5. SHAP y la permutación sí coinciden (Spearman 0,93) | 8 |

**El hallazgo de ingeniería más importante:** el `MultilayerPerceptronClassifier` de Spark densifica la entrada, así que su
memoria crece con `n x d` aunque `FeatureHasher` produzca vectores dispersos. Con `d = 2^12` falla con 300 mil filas. El
enunciado advierte que one-hot es inviable, y el problema persiste con hashing (capítulo 5).

## Cómo leer este libro

1. **Datos:** ficha (capítulo 1), EDA (capítulo 2) y protocolo con preprocesamiento (capítulo 3).
2. **Modelado:** scikit-learn (capítulo 4) y PySpark (capítulo 5).
3. **Resultados:** prueba final (6), benchmark (7), interpretabilidad (8), comparación (9) y reflexión crítica (10).
4. **Anexos:** reproducibilidad (11) y correspondencia con los requisitos del enunciado (12).

Los capítulos con código son notebooks ejecutados que leen las tablas de `results/`. Los experimentos están en `scripts/`.
