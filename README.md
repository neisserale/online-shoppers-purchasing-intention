# Online Shoppers Purchasing Intention

## Fuentes

- [UCI Machine Learning Repository — Dataset #468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [Kaggle — Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset/data)

---

## 1. Descripción del caso de uso

### Contexto de negocio

Una empresa de e-commerce desea anticipar si la sesión de navegación de un visitante terminará en una **compra efectiva**. Hoy en día, solo el **15.5 %** de las sesiones concluyen con una transacción; el 84.5 % restante abandona el sitio sin comprar.

Detectar de forma temprana qué usuarios tienen intención real de compra permite:

- **Personalizar la experiencia en tiempo real** (ofertas, descuentos, recomendaciones)
- **Priorizar recursos de marketing** (retargeting, email, notificaciones push)
- **Optimizar el funnel de conversión** identificando páginas críticas de abandono
- **Medir el ROI de campañas** vinculando el origen del tráfico con la conversión

### Problema ML

| Aspecto | Detalle |
|---|---|
| Tipo de tarea | Clasificación binaria supervisada |
| Variable objetivo | `Revenue` (TRUE = compra realizada, FALSE = sin compra) |
| Distribución de clases | 15.5 % positivos / 84.5 % negativos (dataset desbalanceado) |
| Unidad de análisis | Una sesión de navegación web |
| Horizonte temporal | Predicción en tiempo de sesión (no histórico por usuario) |

### Publicación original

> Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019).
> **Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.**
> *Neural Computing & Applications*, 31, 6893–6908.
