# Informe Comparativo de Algoritmos de Frequent Itemsets Mining

## Introducción

El presente informe expone un análisis comparativo entre los algoritmos **Apriori**, **FP-Growth** y **ECLAT**, aplicados sobre el dataset `online_retail_2.xlsx` provisto en clase. Estos algoritmos son fundamentales para la minería de patrones frecuentes en conjuntos de transacciones, siendo ampliamente utilizados en tareas como análisis de mercado, recomendaciones y detección de asociaciones relevantes.

El objetivo de este análisis es evaluar tanto la **calidad de las reglas generadas** como el **rendimiento computacional** de cada enfoque, considerando métricas específicas del modelo y del sistema.

---

## 1. Metodología

Se ejecutaron los tres algoritmos utilizando implementaciones disponibles en Python:

- **Apriori** de `mlxtend`
- **FP-Growth** de `mlxtend`
- **ECLAT** (implementación personalizada basada en `transactions` y búsqueda recursiva)

Las métricas registradas incluyen:

- **Del modelo**: confianza, lift, leverage, support, conviction y cantidad de reglas generadas.
- **Del sistema**: uso de CPU, tiempo de ejecución, uso de memoria y eficiencia general del proceso.

---

## 2. Estructura del Proyecto

```
association_rules/
├── README.md                           # Este documento
├── cleaned_online_retail.csv           # Dataset preprocesado
├── online_retail_2.xlsx               # Dataset original
├── resource_monitor.py                 # Módulo de monitoreo de recursos
├── output_capture.py                   # Módulo para captura de logs
│
├── parcial/
│   ├── apriori/
│   │   ├── apriori.py                  # Implementación Apriori + MLflow
│   │   ├── ejecucion_apriori.py        # Script de recomendaciones
│   │   ├── apriori_reglas_asociacion.png
│   │   ├── apriori_analisis_detallado.png
│   │   ├── rules_apriori.pkl           # Reglas guardadas
│   │   └── apriori_execution_log.txt   # Log de ejecución
│   │
│   ├── fp_growth/
│   │   ├── fp_growth.py                # Implementación FP-Growth + MLflow
│   │   ├── ejecucion_fp_growth.py      # Script de recomendaciones
│   │   ├── fp_growth_reglas_asociacion.png
│   │   ├── fp_growth_analisis_detallado.png
│   │   ├── rules_fp_growth.pkl         # Reglas guardadas
│   │   └── fp_growth_execution_log.txt # Log de ejecución
│   │
│   └── eclat/
│       ├── eclat.py                    # Implementación ECLAT + MLflow
│       ├── ejecucion_eclat.py          # Script de recomendaciones
│       ├── eclat_reglas_asociacion.png
│       ├── eclat_analisis_detallado.png
│       ├── rules_eclat.pkl             # Reglas guardadas
│       └── eclat_execution_log.txt     # Log de ejecución
│
├── mlruns/                             # Experimentos MLflow
└── mlartifacts/                        # Artifacts MLflow
```

---

## 3. Descripción de Scripts

### 3.1 Módulos Auxiliares

#### `resource_monitor.py`
**Propósito**: Monitoreo en tiempo real de recursos del sistema.
- Mide CPU (proceso y sistema), memoria RSS/VMS
- Calcula eficiencia CPU: `(CPU time / wall time) * 100`
- Threading para monitoreo no intrusivo cada 0.5 segundos
- Integración con MLflow para logging automático

#### `output_capture.py`
**Propósito**: Captura y gestión de logs de ejecución.
- Clase `TeeOutput` para capturar stdout sin interrumpir consola
- Funciones `print_script_header()` y `print_script_footer()`
- Context manager para uso seguro
- Guardado automático en archivos `.txt`

### 3.2 Scripts Principales

#### Algoritmos de Minería (`*.py`)
**Estructura común**:
```python
# 1. Configuración MLflow
mlflow.set_experiment("Reglas_[ALGORITMO]")

# 2. Captura de salida
output_capture = setup_output_capture()

# 3. Monitoreo de recursos
monitor = ResourceMonitor(monitoring_interval=0.5)

# 4. Preprocesamiento del dataset
# - Limpieza de datos
# - Creación de matriz de transacciones

# 5. Ejecución del algoritmo
# - Generación de frequent itemsets
# - Creación de reglas de asociación

# 6. Visualizaciones
# - 6 gráficos principales (distribuciones + scatter + top rules)
# - 4 gráficos detallados (conviction, antecedents, stats)

# 7. Logging MLflow
# - Métricas de rendimiento y calidad
# - Artifacts (gráficos, reglas, logs)
```

**Características específicas**:
- **`apriori.py`**: Usa `mlxtend.frequent_patterns.apriori`
- **`fp_growth.py`**: Usa `mlxtend.frequent_patterns.fpgrowth`
- **`eclat.py`**: Implementación manual con TID-sets y recursión

#### Scripts de Recomendaciones (`ejecucion_*.py`)
**Propósito**: Sistema de recomendaciones basado en reglas generadas.

**Funcionalidades**:
```python
# Cargar modelo entrenado
rules = load_model('parcial/[algoritmo]/rules_[algoritmo].pkl')

# Buscar productos disponibles
search_products(rules, 'CHRISTMAS')

# Generar recomendaciones
get_product_recommendations('WOODEN TREE', rules)

# Estadísticas del modelo
show_general_stats(rules)
```

### 3.3 Parámetros de Configuración

Todos los algoritmos usan los mismos umbrales para comparación justa:
```python
min_support = 0.01      # 1% soporte mínimo
min_confidence = 0.6    # 60% confianza mínima
```

### 3.4 Integración MLflow

**Tracking Server**: `http://localhost:5000`

**Experimentos separados**:
- `Reglas_Apriori`
- `Reglas_FP_Growth`  
- `Reglas_ECLAT`

**Métricas registradas**:
- Rendimiento: `execution_time_seconds`, `cpu_efficiency_percent`
- Memoria: `memory_peak_process_MB`, `memory_max_additional_MB`
- Calidad: `avg_confidence`, `avg_lift`, `rules_total`

**Artifacts automáticos**:
- Gráficos PNG de alta resolución
- Reglas en formato pickle
- Logs completos de ejecución

---

## 4. Resultados

### 2.1 Métricas de Calidad del Modelo

| Métrica           | FP-Growth | Apriori | ECLAT  |
|-------------------|-----------|---------|--------|
| Avg. Confidence   | 0.701     | 0.701   | **0.707** |
| Avg. Conviction   | 3.677     | 3.677   | **3.899** |
| Avg. Lift         | 21.5      | 21.5    | **23.01** |
| Avg. Leverage     | 0.013     | 0.013   | 0.013  |
| Avg. Support      | 0.014     | 0.014   | 0.014  |
| Total de Reglas   | 138       | 138     | **147** |

> **Observación**: ECLAT muestra una leve superioridad en la calidad de las reglas, con mayor confianza, lift y cantidad de reglas generadas.

---

### 2.2 Rendimiento Computacional

#### 2.2.1 Tiempo y CPU

| Métrica                      | FP-Growth | Apriori   | ECLAT    |
|-----------------------------|-----------|-----------|----------|
| Tiempo de CPU (s)           | **22.45** | 87.86     | 29.3     |
| Tiempo total ejecución (s)  | **29.67** | 290.3     | 47.98    |
| % Prom. de CPU usada        | **76.19** | 29.75     | 61.27    |
| Eficiencia CPU (%)          | **75.66** | 30.27     | 61.06    |

> **Observación**: FP-Growth es claramente el más eficiente en términos de procesamiento y tiempo, siendo 10 veces más rápido que Apriori.

#### 2.2.2 Memoria

| Métrica                        | FP-Growth | Apriori       | ECLAT      |
|-------------------------------|-----------|---------------|------------|
| Memoria pico proceso (MB)     | 2870.8    | **31844.3**   | **1443.2** |
| Memoria adicional máxima (MB) | 1636.4    | **30610.3**   | **208.1**  |
| Memoria liberada post-uso (MB)| 1507.4    | **31584.6**   | 119.4      |

> **Observación**: Apriori muestra un uso excesivo de memoria, con picos superiores a los 31 GB, mientras que ECLAT es el más liviano. FP-Growth se mantiene intermedio.

---

## 5. Instrucciones de Uso

### 5.1 Requisitos Previos
```bash
pip install pandas matplotlib numpy mlxtend mlflow psutil
```

### 5.2 Ejecución de Algoritmos
```bash
# Ejecutar cada algoritmo (genera reglas y métricas)
python parcial/apriori/apriori.py
python parcial/fp_growth/fp_growth.py  
python parcial/eclat/eclat.py

# Visualizar resultados en MLflow
mlflow ui --host localhost --port 5000
```

### 5.3 Sistema de Recomendaciones
```bash
# Cargar reglas y hacer recomendaciones
python parcial/apriori/ejecucion_apriori.py
python parcial/fp_growth/ejecucion_fp_growth.py
python parcial/eclat/ejecucion_eclat.py
```

---

## 6. Análisis Comparativo

| Criterio             | Apriori       | FP-Growth     | ECLAT          |
|----------------------|---------------|---------------|----------------|
| Velocidad            | ❌ Muy lento  | ✅ Más rápido | ⚠️ Aceptable   |
| Uso de CPU           | ❌ Bajo       | ✅ Alto       | ⚠️ Medio       |
| Eficiencia Memoria   | ❌ Muy alto   | ⚠️ Moderado   | ✅ Bajo        |
| Calidad de Reglas    | ⚠️ Promedio   | ⚠️ Promedio   | ✅ Alta        |
| Reglas Generadas     | ⚠️ 138        | ⚠️ 138        | ✅ 147         |
| Escalabilidad        | ❌ Mala       | ✅ Alta       | ⚠️ Moderada    |

---

## 7. Conclusión

Cada algoritmo presenta fortalezas y limitaciones particulares:

- **Apriori**, aunque didácticamente valioso, es ineficiente en entornos reales con datasets medianos o grandes, debido a su elevado consumo de memoria y tiempo.
- **FP-Growth** se posiciona como el más **eficiente en términos de CPU y tiempo**, ideal para aplicaciones en tiempo real o entornos de producción.
- **ECLAT** se destaca por la **calidad y cantidad de reglas generadas**, con un uso de memoria notablemente menor al de Apriori y un rendimiento aceptable en tiempo.

En base a los resultados, **FP-Growth** es recomendable cuando se prioriza velocidad y eficiencia, mientras que **ECLAT** resulta valioso cuando el foco está en obtener reglas más significativas sin comprometer demasiado el rendimiento.
