# UPC Projects & Exams - Machine Learning & Data Science

Este repositorio compila los proyectos académicos y evaluaciones prácticas desarrollados en la carrera de **Ingeniería Industrial**, enfocados en la aplicación de Ciencia de Datos, Estadística Avanzada y Machine Learning para la optimización de procesos industriales y logísticos.

## 📁 Estructura del Repositorio

El repositorio se encuentra organizado en las siguientes carpetas principales:

*   **`FINAL PROJECT/`**: Proyecto enfocado en la analítica y optimización de rutas de distribución para e-commerce[cite: 1].
*   **`PC4/`**: Caso de estudio predictivo para la optimización del proceso de producción en la empresa MetalX[cite: 2].

---

## 📊 Proyecto 1: Optimización de Rutas en E-Commerce (Trabajo Parcial)

### Descripción del Problema
Análisis integral y limpieza de un dataset logístico con problemas de calidad de datos (`dirty_ecommerce_logistics_route_planning_dataset.csv`)[cite: 1]. El objetivo es auditar la información de rutas, identificar inconsistencias de proceso y preparar los datos para la toma de decisiones en ventanas de entrega, distancias y costos optimizados[cite: 1].

### Flujo de Desarrollo (Notebook: `TF_FINALDELFINAL (1).ipynb`)
1.  **Análisis Exploratorio de Datos (EDA):** Extracción de estadísticas descriptivas completas (Media, Mediana, Moda, Desviación Estándar, Coeficiente de Variación, Asimetría y Curtosis)[cite: 1].
2.  **Auditoría de Calidad:** Identificación de valores nulos, registros duplicados y detección de outliers mediante el método del Rango Intercuartílico (IQR)[cite: 1].
3.  **Limpieza y Preparación:** Eliminación de duplicados, corrección de inconsistencias físicas (como velocidades negativas o distancias extremas) e imputación de valores faltantes por medianas robustas[cite: 1].
4.  **Análisis de Relaciones:** Evaluación de dependencias mediante matrices de correlación no paramétricas (Coeficiente de Spearman) y mapas de calor[cite: 1].

### 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python
*   **Librerías Clave:** Pandas, NumPy, Matplotlib, Seaborn, SciPy (Stats)[cite: 1]

---

## 🤖 Proyecto 2: Control de Calidad Predictivo - Caso MetalX (PC4)

### Descripción del Problema
Desarrollo de un modelo de Machine Learning y aprendizaje profundo para predecir la cantidad de piezas defectuosas por lote (`Defectos_Lote`) en un proceso de inyección de plástico, basándose en variables críticas de la maquinaria en tiempo real[cite: 2].

### Variables del Proceso (Dataset: `df_metalx.csv`)
*   **Características (X):** Temperatura de Matriz (°C), Presión de Inyección (Bar), Tiempo de Ciclo (s) y Vibración RMS[cite: 2].
*   **Variable Objetivo (y):** Defectos por Lote[cite: 2].

### Modelos de Machine Learning Evaluados
Para garantizar la precisión, se entrenaron y compararon tres arquitecturas distintas:
*   **Árbol de Decisión Regresor:** Optimizado mediante búsqueda en grilla (`GridSearchCV`)[cite: 2].
*   **Random Forest Regresor:** Evaluación de ensambles, optimización de hiperparámetros y análisis de importancia de características (*Feature Importance*)[cite: 2].
*   **Red Neuronal Artificial (RNA):** Modelo secuencial deep learning con capas densas y funciones de activación ReLU y lineales[cite: 2].

### Validación y Métricas
El rendimiento de los modelos se validó rigurosamente utilizando **Validación Cruzada de 5 pliegues (5-Fold Cross Validation)**, evaluando bajo las métricas[cite: 2]:
*   Error Absoluto Medio (MAE)[cite: 2]
*   Error Cuadrático Medio (MSE)[cite: 2]
*   Coeficiente de Determinación ($R^2$)[cite: 2]

### 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python
*   **Librerías Clave:** Scikit-Learn, TensorFlow / Keras, Pandas, NumPy, Seaborn[cite: 2]

---

## 🚀 Cómo Ejecutar los Proyectos

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone [https://github.com/fcarruitero24/UPC_projects_exams.git](https://github.com/fcarruitero24/UPC_projects_exams.git)
