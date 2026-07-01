# 🚀 GUÍA RÁPIDA: Cómo Ejecutar los Modelos

## 1️⃣ OPCIÓN MÁS RÁPIDA: Ejecutar TODO de una vez

```bash
cd /workspaces/Prueba_Modelo
bash ejecutar_todos.sh
```

**Resultado:**
- Se entrenan los 3 modelos
- Se generan los modelos guardados (.pkl)
- Verás métricas de cada uno

---

## 2️⃣ EJECUTAR MODELOS INDIVIDUALES

### XGBoost
```bash
cd /workspaces/Prueba_Modelo/Practica_modelo
python entrenamiento.py
```

**Archivos generados:**
- `modelo_xgboost_casas.pkl` - Modelo entrenado
- Resultado: **Accuracy 90%**

---

### Random Forest
```bash
cd /workspaces/Prueba_Modelo/Practica_RandomForest
python entrenamiento.py
```

**Archivos generados:**
- `modelo_randomforest_casas.pkl` - Modelo entrenado
- Resultado: **Accuracy 95%** ← MEJOR

---

### Regresión Logística
```bash
cd /workspaces/Prueba_Modelo/Practica_RegresionLogistica
python entrenamiento.py
```

**Archivos generados:**
- `modelo_logistico_casas.pkl` - Modelo entrenado
- `scaler_casas.pkl` - Escalador (necesario para predicciones)
- Resultado: **AUC-ROC 0.98** ← MEJOR

---

## 3️⃣ VER LOS RESULTADOS

Después de ejecutar los entrenamientos, lee:

```bash
# Resultados detallados con casas.csv
cat RESULTADOS_CASAS.md

# Comparativa de los 3 modelos
cat COMPARATIVA_MODELOS.md

# Documentación de cada modelo
cat Practica_modelo/README.md
cat Practica_RandomForest/README.md
cat Practica_RegresionLogistica/README.md
```

---

## 📊 ¿QUÉ VERÁS EN LA CONSOLA?

### Procesamiento de Variables
```
No hay variables de texto en el dataset
Mediana de precio: $2,275,390.51
Precio alto (≥ mediana): 50 casas
Precio bajo (< mediana): 50 casas
```

### Validación Cruzada
```
AUC-ROC promedio: 0.97 (±0.03)
```

### Resultados Finales
```
========================================
RESULTADOS - XGBOOST (Casas)
========================================
Accuracy:  0.9000 (90.00%)
AUC-ROC:   0.9500

Matriz de Confusión:
[[9 1]
 [1 9]]

Coeficientes/Importancia: [mostrado según modelo]
```

---

## 🎯 RESUMEN RÁPIDO DE RESULTADOS

```
┌─────────────────────┬──────────┬─────────┐
│ MODELO              │ ACCURACY │ AUC-ROC │
├─────────────────────┼──────────┼─────────┤
│ XGBoost             │ 90%      │ 0.95    │
│ Random Forest 🏆    │ 95%      │ 0.96    │
│ Regresión Logística │ 90%      │ 0.98🏆  │
└─────────────────────┴──────────┴─────────┘
```

---

## 💡 ¿QUÉ SIGNIFICAN LOS RESULTADOS?

### Accuracy
- Porcentaje de predicciones correctas
- Random Forest: 95% = Predice correctamente 19 de 20 casos

### AUC-ROC
- Capacidad del modelo para distinguir entre clases
- Rango: 0 a 1 (1 = perfecto)
- Regresión Logística: 0.98 = Excelente discriminación

### Matriz de Confusión
```
[[9  1]    = Fila "Precio Bajo"
 [1  9]]   = Fila "Precio Alto"

Columnas: [Predicción Bajo, Predicción Alto]
```

- **9** = Aciertos en Precio Bajo (predijo bajo y era bajo)
- **1** = Error (predijo alto pero era bajo)
- **1** = Error (predijo bajo pero era alto)
- **9** = Aciertos en Precio Alto

---

## 🔍 CARACTERÍSTICAS DE CADA MODELO

### XGBoost
- **Ventaja:** Buen balance
- **Desventaja:** Menos interpretable
- **Uso:** Cuando necesitas buena precisión

### Random Forest 🏆
- **Ventaja:** Mejor accuracy (95%)
- **Desventaja:** Más memoria
- **Uso:** Cuando la precisión es crítica

### Regresión Logística 🏆
- **Ventaja:** Mejor AUC-ROC, muy interpretable
- **Desventaja:** Más simple
- **Uso:** Cuando necesitas explicar el modelo

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### "ModuleNotFoundError: No module named 'xgboost'"
```bash
pip install xgboost
```

### "FileNotFoundError: casas.csv"
Asegúrate de estar en la carpeta correcta:
```bash
cd /workspaces/Prueba_Modelo/Practica_modelo
```

### El script .sh no ejecuta
```bash
chmod +x ejecutar_todos.sh
bash ejecutar_todos.sh
```

---

## 📁 ESTRUCTURA FINAL

```
Prueba_Modelo/
├── casas.csv                      ← Dataset
├── README.md                      ← Este archivo
├── ejecutar_todos.sh              ← Script para ejecutar todo
├── RESULTADOS_CASAS.md            ← Resultados detallados
├── COMPARATIVA_MODELOS.md         ← Comparativa
├── GUÍA_RÁPIDA.md                 ← Este archivo
│
├── Practica_modelo/               (XGBoost)
│   ├── casas.csv
│   ├── entrenamiento.py           ← Ejecutar esto
│   ├── prediccion.py
│   └── modelo_xgboost_casas.pkl   ← Se genera
│
├── Practica_RandomForest/         (Random Forest)
│   ├── casas.csv
│   ├── entrenamiento.py           ← Ejecutar esto
│   ├── prediccion.py
│   └── modelo_randomforest_casas.pkl ← Se genera
│
└── Practica_RegresionLogistica/   (Regresión Logística)
    ├── casas.csv
    ├── entrenamiento.py           ← Ejecutar esto
    ├── prediccion.py
    ├── modelo_logistico_casas.pkl ← Se genera
    └── scaler_casas.pkl           ← Se genera
```

---

## ✅ CHECKLIST

- [ ] Instalé las librerías: `pip install xgboost scikit-learn pandas joblib`
- [ ] Ejecuté los modelos: `bash ejecutar_todos.sh`
- [ ] Leí los resultados: `cat RESULTADOS_CASAS.md`
- [ ] Entiendo qué variable predice cada modelo
- [ ] Conozco el mejor modelo para cada métrica

¡**Listo! Ahora estás listo para usar los modelos** 🚀
