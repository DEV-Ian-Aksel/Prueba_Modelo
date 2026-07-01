# 🏠 RESULTADOS: Predicción de Precios de Casas

## Dataset: casas.csv
- **Registros:** 100 casas
- **Variable de entrada:** metros_cuadrados
- **Variable objetivo:** precio_alto (clasificación binaria)
  - Precio alto: ≥ mediana ($2,275,390.51)
  - Precio bajo: < mediana

---

## 📊 COMPARATIVA DE RESULTADOS

### Métricas de los 3 Modelos

| Métrica | XGBoost | Random Forest | Regresión Logística |
|---------|---------|---------------|-----------------|
| **Accuracy** | 90.00% | **95.00%** ✅ | 90.00% |
| **AUC-ROC** | 0.9500 | 0.9600 | **0.9800** ✅ |
| **Verdaderos Positivos (TP)** | 9 | **9** | 9 |
| **Verdaderos Negativos (TN)** | 9 | **10** ✅ | 9 |
| **Falsos Positivos (FP)** | 1 | **0** ✅ | 1 |
| **Falsos Negativos (FN)** | 1 | **1** | 1 |

---

## 🎯 ANÁLISIS POR MODELO

### 1️⃣ **XGBoost**

```
Accuracy:  0.9000 (90.00%)
AUC-ROC:   0.9500

Matriz de Confusión:
[[9 1]
 [1 9]]
```

**Interpretación:**
- Predice correctamente el 90% de casos
- AUC-ROC: 0.95 (excelente discriminación)
- 1 falso positivo (dijo precio alto pero era bajo)
- 1 falso negativo (dijo precio bajo pero era alto)

**Coeficientes:** No disponibles directamente (modelo "caja negra")

---

### 2️⃣ **Random Forest** 🏆 MEJOR ACCURACY

```
Accuracy:  0.9500 (95.00%)  ← MEJOR
AUC-ROC:   0.9600

Matriz de Confusión:
[[10  0]   ← Sin falsos positivos
 [ 1  9]]
```

**Interpretación:**
- ✅ **MEJOR ACCURACY**: 95% vs 90%
- Detecta TODOS los precios bajos (TN=10, FP=0)
- Detecta correctamente 9 de 10 precios altos
- 1 falso negativo (precio alto clasificado como bajo)

**Importancia de variables:**
```
metros_cuadrados: 100% (solo variable)
```

**Conclusión:** Random Forest es el mejor modelo para este dataset.

---

### 3️⃣ **Regresión Logística** 🏆 MEJOR AUC-ROC

```
Accuracy:  0.9000 (90.00%)
AUC-ROC:   0.9800  ← MEJOR

Matriz de Confusión:
[[9 1]
 [1 9]]
```

**Interpretación:**
- ✅ **MEJOR AUC-ROC**: 0.98 (mejor discriminación)
- Accuracy: 90% (igual a XGBoost)
- AUC-ROC más alto → mejor capacidad de distinguir clases

**Coeficientes (INTERPRETABLES):**
```
metros_cuadrados: 2.7943 (POSITIVO)
Intercept: 0.0488

Significado:
- Cada 1 unidad de variación en metros_cuadrados → 
  aumenta la probabilidad de precio alto en 2.79 unidades
```

**Conclusión:** Mejor capacidad discriminativa y altamente interpretable.

---

## 📈 RESUMEN GANADORES

| Criterio | Ganador | Por qué |
|----------|--------|--------|
| **Mejor Accuracy** | Random Forest 🥇 | 95% vs 90% |
| **Mejor AUC-ROC** | Regresión Logística 🥇 | 0.98 vs 0.96 |
| **Mejor Interpretabilidad** | Regresión Logística 🥇 | Coeficientes directos |
| **Mejor equilibrio** | Random Forest 🥇 | Accuracy + simplicidad |
| **Velocidad** | Regresión Logística 🥇 | Entrena en ms |

---

## 🚀 CÓMO EJECUTAR LOS MODELOS

### Opción 1: Ejecutar un modelo a la vez

#### XGBoost
```bash
cd Practica_modelo
python entrenamiento.py
python prediccion.py
```

#### Random Forest
```bash
cd Practica_RandomForest
python entrenamiento.py
python prediccion.py
```

#### Regresión Logística
```bash
cd Practica_RegresionLogistica
python entrenamiento.py
python prediccion.py
```

---

### Opción 2: Ejecutar todos a la vez (desde la carpeta raíz)

```bash
# Terminal 1: XGBoost
cd Practica_modelo && python entrenamiento.py

# Terminal 2: Random Forest
cd Practica_RandomForest && python entrenamiento.py

# Terminal 3: Regresión Logística
cd Practica_RegresionLogistica && python entrenamiento.py
```

---

## 📁 ARCHIVOS GENERADOS

Después de ejecutar los entrenamientos, cada carpeta contendrá:

### Practica_modelo/
- `modelo_xgboost_casas.pkl` - Modelo XGBoost entrenado
- `predicciones.csv` - Predicciones en nuevos datos

### Practica_RandomForest/
- `modelo_randomforest_casas.pkl` - Modelo Random Forest
- `predicciones_rf.csv` - Predicciones

### Practica_RegresionLogistica/
- `modelo_logistico_casas.pkl` - Modelo Logístico
- `scaler_casas.pkl` - Escalador de variables
- `predicciones_logistica.csv` - Predicciones

---

## 💡 RECOMENDACIONES

### Para **máxima precisión**:
→ Usa **Random Forest** (95% accuracy)

### Para **mejor discriminación**:
→ Usa **Regresión Logística** (0.98 AUC-ROC)

### Para **explicabilidad** (stakeholders):
→ Usa **Regresión Logística** (coeficientes interpretables)

### Para **producción** (balance todo):
→ Usa **Random Forest** (mejor accuracy + simple)

### Para **ensemble** (máxima robustez):
→ Promedia predicciones de los 3 modelos

---

## 🔧 PROCESAMIENTO DE VARIABLES

Todos los modelos incluyen código para:

```python
# 1. Detectar variables de texto
columnas_texto = df.select_dtypes(include=['object']).columns

# 2. Convertir a binario con LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[col] = le.fit_transform(df[col].astype(str))

# 3. Crear variable objetivo
df['precio_alto'] = (df['precio'] >= mediana_precio).astype(int)
```

---

## 📊 MATRIZ DE CONFUSIÓN EXPLICADA

```
              Predicción
            Bajo    Alto
Real  Bajo   TN     FP
      Alto   FN     TP
```

- **TN (Verdaderos Negativos):** Predijo bajo y era bajo ✓
- **TP (Verdaderos Positivos):** Predijo alto y era alto ✓
- **FP (Falsos Positivos):** Predijo alto pero era bajo ✗
- **FN (Falsos Negativos):** Predijo bajo pero era alto ✗

---

## 📚 DOCUMENTACIÓN ADICIONAL

Ver también:
- [COMPARATIVA_MODELOS.md](COMPARATIVA_MODELOS.md) - Comparativa XGBoost vs RF vs Logística
- [Practica_modelo/README.md](Practica_modelo/README.md) - Detalles XGBoost
- [Practica_RandomForest/README.md](Practica_RandomForest/README.md) - Detalles Random Forest
- [Practica_RegresionLogistica/README.md](Practica_RegresionLogistica/README.md) - Detalles Regresión Logística

