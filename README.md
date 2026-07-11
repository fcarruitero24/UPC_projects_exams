# UPC Projects & Exams - Machine Learning & Data Science

Este repositorio compila los proyectos académicos y evaluaciones prácticas desarrollados en la carrera de **Ingeniería Industrial**, enfocados en la aplicación de Ciencia de Datos, Estadística Avanzada y Machine Learning para la optimización de procesos industriales y logísticos.

## 📁 Estructura del Repositorio

El repositorio se encuentra organizado en las siguientes carpetas principales:

*   **`FINAL PROJECT/`**: Proyecto enfocado en la analítica y optimización de rutas de distribución para e-commerce.
*   **`PC4/`**: Caso de estudio predictivo para la optimización del proceso de producción en la empresa MetalX.

---

## 📊 Proyecto 1: Optimización de Rutas en E-Commerce (Trabajo Parcial)

### Descripción del Problema
Análisis integral y limpieza de un dataset logístico con problemas de calidad de datos (`dirty_ecommerce_logistics_route_planning_dataset.csv`). El objetivo es auditar la información de rutas, identificar inconsistencias de proceso y preparar los datos para la toma de decisiones en ventanas de entrega, distancias y costos optimizados.

### Flujo de Desarrollo (Notebook: `TF_FINALDELFINAL (1).ipynb`)
1.  **Análisis Exploratorio de Datos (EDA):** Extracción de estadísticas descriptivas completas (Media, Mediana, Moda, Desviación Estándar, Coeficiente de Variación, Asimetría y Curtosis).
2.  **Auditoría de Calidad:** Identificación de valores nulos, registros duplicados y detección de outliers mediante el método del Rango Intercuartílico (IQR).
3.  **Limpieza y Preparación:** Eliminación de duplicados, corrección de inconsistencias físicas (como velocidades negativas o distancias extremas) e imputación de valores faltantes por medianas robustas].
4.  **Análisis de Relaciones:** Evaluación de dependencias mediante matrices de correlación no paramétricas (Coeficiente de Spearman) y mapas de calor.

### 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python
*   **Librerías Clave:** Pandas, NumPy, Matplotlib, Seaborn, SciPy (Stats).

---

## 🤖 Proyecto 2: Control de Calidad Predictivo - Caso MetalX (PC4)

### Descripción del Problema
Desarrollo de un modelo de Machine Learning y aprendizaje profundo para predecir la cantidad de piezas defectuosas por lote (`Defectos_Lote`) en un proceso de inyección de plástico, basándose en variables críticas de la maquinaria en tiempo real.

### Variables del Proceso (Dataset: `df_metalx.csv`)
*   **Características (X):** Temperatura de Matriz (°C), Presión de Inyección (Bar), Tiempo de Ciclo (s) y Vibración RMS].
*   **Variable Objetivo (y):** Defectos por Lote.

### Modelos de Machine Learning Evaluados
Para garantizar la precisión, se entrenaron y compararon tres arquitecturas distintas:
*   **Árbol de Decisión Regresor:** Optimizado mediante búsqueda en grilla (`GridSearchCV`).
*   **Random Forest Regresor:** Evaluación de ensambles, optimización de hiperparámetros y análisis de importancia de características (*Feature Importance*).
*   **Red Neuronal Artificial (RNA):** Modelo secuencial deep learning con capas densas y funciones de activación ReLU y lineales.

### Validación y Métricas
El rendimiento de los modelos se validó rigurosamente utilizando **Validación Cruzada de 5 pliegues (5-Fold Cross Validation)**, evaluando bajo las métricas:
*   Error Absoluto Medio (MAE)
*   Error Cuadrático Medio (MSE)
*   Coeficiente de Determinación ($R^2$)

### 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python
*   **Librerías Clave:** Scikit-Learn, TensorFlow / Keras, Pandas, NumPy, Seaborn

---

## 🚀 Cómo Ejecutar los Proyectos

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone [https://github.com/fcarruitero24/UPC_projects_exams.git](https://github.com/fcarruitero24/UPC_projects_exams.git)
