# 📊 Predicción de disponibilidad en parkings de Valencia

- Se plantea una solución escalable para un proyecto de análisis y predicción de disponibilidad de parkings en la ciudad de Valencia.  
- Se comparan algoritmos de deep learning específicos para tratar datos con estructura temporal (RNN, LSTM, GRU).  
- Se evalúa el rendimiento de cada modelo en su versión básica (vanilla) frente a una versión optimizada mediante ajuste de hiperparámetros con Optuna.  
- Se definen métricas de evaluación para medir su rendimiento y se incorporan técnicas de explicabilidad para entender las predicciones realizadas por cada modelo.

---

## Contenido de los notebooks

### `1_Prepare_ts.ipynb`

Inspección inicial de los datos para preparar las series temporales antes del preprocesamiento:

- Inspección de tipos de datos de cada columna.
- Análisis del número de registros por aparcamiento.
- Eliminación de columnas irrelevantes.
- Corrección de inconsistencias: identificadores duplicados, variaciones en la capacidad total, valores negativos o superiores al total de plazas disponibles.
- Eliminación de aparcamientos con datos insuficientes.
- Determinación de los rangos de inicio y fin de cada serie.
- Visualización de la ocupación por aparcamiento.
- Agrupación y promediado de registros múltiples por hora.

---

### `2_Preprocess.ipynb`

Corrección de irregularidades y preparación del dataset final:

- Identificación y caracterización de huecos en cada serie.
- Visualización y conteo de valores faltantes consecutivos.
- Filtrado de tramos irrelevantes (con huecos extensos al inicio).
- Imputación de valores faltantes mediante media ponderada de los 4 valores anteriores y el valor de la misma hora de la semana previa.

---

### `3_EDA.ipynb`

Análisis exploratorio para detectar patrones de comportamiento por aparcamiento:

- Visualización de la disponibilidad de plazas por aparcamiento.
- Análisis de ocupación por año, mes, día de la semana y hora.
- Cálculo del periodograma para identificar estacionalidades en el dominio de la frecuencia.

---

### `4_Modelling.ipynb`

Entrenamiento de modelos RNN, LSTM y GRU para la predicción de ocupación:

- Entrenamiento de versiones base (vanilla).
- Optimización de hiperparámetros con Optuna.
- Guardado automático del mejor modelo validado (sin reentrenamiento).
- Entrenamiento con early stopping y registro de métricas de validación.

---

### `5_ResultsAnalysis.ipynb`

Evaluación de resultados y visualización de predicciones:

- Comparación del MAE por aparcamiento y modelo.
- Visualización de predicciones en el conjunto de test.
- Zoom a la primera semana del conjunto de test para mayor detalle.

---

### `6_ModelInterpretability.ipynb`

Interpretabilidad del modelo con técnicas explicativas:

- Cálculo de valores SHAP para obtener explicaciones locales de cada predicción individual.

---
