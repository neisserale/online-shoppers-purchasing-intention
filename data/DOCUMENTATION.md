# Documentación de Datos — Online Shoppers Purchasing Intention

---

## 1. Descripción del dataset

| Característica | Valor |
|---|---|
| Registros | 12,330 sesiones |
| Variables | 17 predictoras + 1 objetivo |
| Variables numéricas | 10 |
| Variables categóricas | 7 |
| Valores faltantes | Mínimos |
| Periodo cubierto | 1 año (sin especificar) |
| Granularidad | Una fila = una sesión de usuario único |

El dataset fue construido de modo que cada sesión pertenece a un usuario distinto, evitando tendencias hacia campañas específicas, días especiales o perfiles de usuario concretos.

---

## 2. Diccionario de variables

### 2.1 Variables de comportamiento de navegación (numéricas)

Estas seis variables capturan cuántas páginas visitó el usuario y cuánto tiempo dedicó a cada categoría de contenido del sitio.

| Variable | Tipo | Rango en datos | Descripción |
|---|---|---|---|
| `Administrative` | Entero | 0 – 27 | Número de páginas administrativas visitadas (ej. login, cuenta, configuración) |
| `Administrative_Duration` | Float | 0 – ~3,399 s | Tiempo total (segundos) en páginas administrativas |
| `Informational` | Entero | 0 – 24 | Número de páginas informativas visitadas (ej. sobre nosotros, FAQ, políticas) |
| `Informational_Duration` | Float | 0 – ~2,550 s | Tiempo total (segundos) en páginas informativas |
| `ProductRelated` | Entero | 0 – 705 | Número de páginas de producto visitadas |
| `ProductRelated_Duration` | Float | 0 – ~63,974 s | Tiempo total (segundos) en páginas de producto |

> **Nota:** `ProductRelated` tiene el rango más amplio (0–705) y el mayor peso en la predicción, dado que refleja directamente el interés en productos.

### 2.2 Variables de métricas web (numéricas, provenientes de Google Analytics)

| Variable | Tipo | Rango en datos | Descripción |
|---|---|---|---|
| `BounceRates` | Float | 0.000 – 0.200 | Porcentaje de visitantes que entran a una página y la abandonan sin interactuar más con el sitio. Se calcula como promedio de las páginas visitadas en la sesión. Un valor alto indica bajo engagement. |
| `ExitRates` | Float | 0.000 – 0.200 | Porcentaje de salidas del sitio desde una página específica sobre el total de visitas a esa página. Se promedia por las páginas vistas en la sesión. Siempre ≥ `BounceRates` para la misma página. |
| `PageValues` | Float | 0.000 – 361.764 | Valor monetario promedio (en unidades de Google Analytics) de las páginas visitadas antes de completar una transacción. Páginas cercanas al checkout reciben mayor valor. Un `PageValues` alto sugiere que el usuario llegó a páginas con alta relevancia transaccional. |

> **Relación BounceRates vs ExitRates:** Un "bounce" es un caso especial de "exit" (la sesión tuvo una sola página). `ExitRates` es siempre ≥ `BounceRates`.

### 2.3 Variable de contexto temporal (numérica)

| Variable | Tipo | Rango en datos | Descripción |
|---|---|---|---|
| `SpecialDay` | Float | 0.0 – 1.0 | Proximidad de la fecha de la sesión a un día especial de compras (ej. San Valentín, Día de la Madre, Black Friday). 0 = lejos de cualquier fecha especial; 1 = exactamente en la fecha especial. El valor aumenta gradualmente en los días previos y decae después. |

### 2.4 Variables categóricas de contexto

| Variable | Tipo | Valores únicos | Descripción |
|---|---|---|---|
| `Month` | Texto | Feb, Mar, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec | Mes en que ocurrió la sesión. **Nota:** Enero y Abril no están presentes en los datos. |
| `OperatingSystems` | Entero (código) | 1 – 8 | Sistema operativo del visitante (codificado, sin mapeo público disponible) |
| `Browser` | Entero (código) | 1 – 13 | Navegador utilizado (codificado) |
| `Region` | Entero (código) | 1 – 9 | Región geográfica del visitante (codificada) |
| `TrafficType` | Entero (código) | 1 – 20 | Canal o fuente de tráfico (ej. búsqueda orgánica, campañas de pago, referido, directo; codificado) |
| `VisitorType` | Texto | New_Visitor, Returning_Visitor, Other | Tipo de visitante según historial en el sitio |
| `Weekend` | Booleano | TRUE / FALSE | Indica si la sesión ocurrió en fin de semana (sábado o domingo) |

### 2.5 Variable objetivo

| Variable | Tipo | Valores | Descripción |
|---|---|---|---|
| `Revenue` | Booleano | TRUE / FALSE | **Variable objetivo.** TRUE indica que la sesión terminó en una compra efectiva (transacción completada). FALSE indica que el usuario abandonó sin comprar. |

**Distribución:**

| Clase | Registros | Porcentaje |
|---|---|---|
| FALSE (sin compra) | 10,422 | 84.5 % |
| TRUE (compra) | 1,908 | 15.5 % |

---

## 3. Consideraciones para el modelado ML

### Desbalance de clases

Con solo el 15.5 % de ejemplos positivos, el modelo puede alcanzar 84.5 % de accuracy simplemente prediciendo siempre "no compra". Estrategias recomendadas:

- **Métricas:** Usar F1-score, Precision-Recall AUC, ROC-AUC en lugar de accuracy
- **Remuestreo:** SMOTE (oversampling de la clase minoritaria) o undersampling de la mayoría
- **Ajuste de umbral:** Calibrar el threshold de decisión según el costo de negocio (falso negativo vs falso positivo)
- **`class_weight='balanced'`** en algoritmos que lo soporten (Logistic Regression, Random Forest, SVM)

### Variables codificadas

`OperatingSystems`, `Browser`, `Region` y `TrafficType` son códigos numéricos sin semántica ordinal. Deben tratarse como **categóricas nominales** (one-hot encoding o target encoding) y no como variables continuas.

### Correlación entre variables de navegación

`BounceRates` y `ExitRates` tienen alta correlación positiva entre sí. Considerar análisis de multicolinealidad antes de modelos lineales.

### Ingeniería de features sugerida

| Feature derivada | Lógica |
|---|---|
| `Total_Pages` | `Administrative + Informational + ProductRelated` |
| `Total_Duration` | Suma de las tres duraciones |
| `Product_Ratio` | `ProductRelated / Total_Pages` (% de páginas de producto) |
| `Avg_Time_Per_Page` | `Total_Duration / Total_Pages` |
| `Is_SpecialDay` | Binarización de `SpecialDay > 0` |

### Algoritmos recomendados

| Algoritmo | Justificación |
|---|---|
| **Gradient Boosting (XGBoost / LightGBM)** | Robusto ante desbalance, maneja categorías, alta performance en tabular data |
| **Random Forest** | Baseline sólido, interpretable vía feature importance |
| **Logistic Regression** | Interpretable, útil como baseline y para calibración de probabilidades |
| **Redes neuronales (MLP / LSTM)** | Arquitectura de la publicación original |

### Validación

Dado que los datos cubren un año completo con estacionalidad (meses sin Enero/Abril, concentración en Nov-Dic por compras navideñas), se recomienda **validación cruzada estratificada** para preservar la proporción de clases en cada fold.
