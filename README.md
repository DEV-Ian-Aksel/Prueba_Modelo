# Prueba_Modelo
Proyecto con **3 modelos de ML**: XGBoost, Random Forest y Regresión Logística

## 📊 Datasets Disponibles

1. **casas.csv** - Predicción de precios de casas
2. **data (1).csv → datos.csv** - Predicción de deserción académica ✨ NUEVO

## 🎓 RESULTADOS CON DATOS ACADÉMICOS

| Modelo | Accuracy | AUC-ROC | Destacado |
|--------|----------|---------|-----------|
| XGBoost | 87% | 0.91 | Buen balance |
| Random Forest | **87.8%** | **0.9294** | 🏆 Mejor general |
| Regresión Logística | 84.6% | 0.9274 | 🏆 Mejor recall (87%) |

**→ Ver detalles en:** [RESULTADOS_ACADEMICOS.md](RESULTADOS_ACADEMICOS.md)

### 1. XGBoost (Practica_modelo)
```bash
cd Practica_modelo && python entrenamiento.py
```

### 2. Random Forest (Practica_RandomForest)
```bash
cd Practica_RandomForest && python entrenamiento.py
```

### 3. Regresión Logística (Practica_RegresionLogistica)
```bash
cd Practica_RegresionLogistica && python entrenamiento.py
```

## ⚡ Ejecutar Todos los Modelos

```bash
bash ejecutar_todos.sh
```

## 📈 Resultados con casas.csv

| Modelo | Accuracy | AUC-ROC | Destacado |
|--------|----------|---------|-----------|
| XGBoost | 90% | 0.95 | Bueno |
| Random Forest | **95%** | 0.96 | 🏆 Mejor Accuracy |
| Regresión Logística | 90% | **0.98** | 🏆 Mejor AUC-ROC |

Ver detalles en: [RESULTADOS_CASAS.md](RESULTADOS_CASAS.md)

## 📚 Documentación

- [COMPARATIVA_MODELOS.md](COMPARATIVA_MODELOS.md) - Comparación de los 3 modelos
- [RESULTADOS_CASAS.md](RESULTADOS_CASAS.md) - Resultados con dataset de casas
- [Practica_modelo/README.md](Practica_modelo/README.md) - Detalles de XGBoost
- [Practica_RandomForest/README.md](Practica_RandomForest/README.md) - Detalles de Random Forest
- [Practica_RegresionLogistica/README.md](Practica_RegresionLogistica/README.md) - Detalles de Regresión Logística

## 📦 Librerias Necesarias

* pandas
* scikit-learn
* joblib
* xgboost