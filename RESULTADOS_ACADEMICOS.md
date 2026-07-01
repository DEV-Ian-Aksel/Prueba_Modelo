# 🎓 RESULTADOS: Predicción de Deserción Académica

## Dataset: data (1).csv → datos.csv
- **Registros:** 4,425 estudiantes
- **Variables:** 36 características académicas y personales
- **Variable objetivo:** Target (Dropout, Graduate, Enrolled)
- **Clasificación:** Desertor (1) vs No Desertor (0)

---

## 📊 COMPARATIVA DE RESULTADOS - TRES MODELOS

### Métricas Generales

| Métrica | XGBoost | Random Forest | Regresión Logística |
|---------|---------|---------------|-----------------|
| **Accuracy** | 87% | **87.8%** ✅ | 84.6% |
| **AUC-ROC** | 0.91 | **0.9294** ✅ | 0.9274 |
| **Precisión (Desertor)** | 0.80 | **0.82** | 0.72 |
| **Recall (Desertor)** | 0.79 | **0.80** | **0.87** ✅ |
| **F1-Score (Desertor)** | 0.80 | **0.81** | 0.78 |

---

## 🎯 ANÁLISIS DETALLADO POR MODELO

### 1️⃣ **XGBoost**

```
Accuracy:  87.00%
AUC-ROC:   0.9100

Matriz de Confusión:
[[544  57]
 [ 60 224]]

Verdaderos Positivos (TP): 224 desertores detectados
Falsos Negativos (FN): 60 desertores no detectados
```

**Top 5 Variables Más Importantes:**
1. Curricular units 2nd sem (approved) - 21.7%
2. Curricular units 1st sem (approved) - 11.7%
3. Curricular units 2nd sem (grade) - 8.1%
4. Tuition fees up to date - 7.1%
5. Curricular units 1st sem (enrolled) - 5.3%

**Conclusión:** Buen modelo. Las unidades aprobadas son el mejor predictor.

---

### 2️⃣ **Random Forest** 🏆 MEJOR ACCURACY

```
Accuracy:  87.80%  ← MEJOR
AUC-ROC:   0.9294

Matriz de Confusión:
[[551  50]   ← Menos falsos positivos
 [ 58 226]]
```

**Top 5 Variables Más Importantes:**
1. Curricular units 2nd sem (approved) - 22.1%
2. Curricular units 2nd sem (grade) - 13.4%
3. Curricular units 1st sem (approved) - 9.9%
4. Tuition fees up to date - 8.0%
5. Curricular units 1st sem (grade) - 7.6%

**Ventajas:**
- ✅ Mejor accuracy (87.8% vs 87%)
- ✅ Menos falsos positivos (50 vs 57)
- ✅ Mejor balance general
- ✅ Variables importantes muy claras

**Conclusión:** **MEJOR MODELO PARA ESTE PROBLEMA**

---

### 3️⃣ **Regresión Logística** 🏆 MEJOR RECALL

```
Accuracy:  84.63%
AUC-ROC:   0.9274

Matriz de Confusión:
[[503  98]
 [ 38 246]]
     ↑         ↑
Detecta bien desertores (Recall 87%)
```

**Top 5 Variables con Mayor Impacto POSITIVO (aumentan deserción):**
1. Curricular units 2nd sem (enrolled) - 0.853
2. Curricular units 2nd sem (credited) - 0.510
3. Age at enrollment - 0.353
4. Nacionality - 0.302
5. Course - 0.264

**Top 5 Variables con Mayor Impacto NEGATIVO (reducen deserción):**
1. Curricular units 2nd sem (approved) - **-1.834** (MÁS IMPORTANTE)
2. Curricular units 1st sem (approved) - -0.938
3. Tuition fees up to date - -0.725
4. Mother's occupation - -0.337
5. International - -0.357

**Ventajas:**
- ✅ Mejor recall (87% detecta desertores)
- ✅ Coeficientes interpretables
- ✅ Explicable a stakeholders
- ✅ Menor accuracy pero mejor para detectar desertores

**Conclusión:** Mejor para **identificar a los que van a desertar** (detecta 87% vs 80%)

---

## 🔑 INTERPRETACIÓN DE RESULTADOS

### ¿Qué es Accuracy?
Porcentaje de predicciones correctas en total.
- Random Forest: 87.8% = acierta en 777 de 885 casos

### ¿Qué es AUC-ROC?
Capacidad del modelo para distinguir entre desertores y no desertores.
- Random Forest: 0.9294 (excelente, cercano a 1.0)

### ¿Qué es Recall?
De los estudiantes que VAN a desertar, ¿cuántos identifica?
- Regresión Logística: 87% = detecta a 87 de 100 desertores
- Random Forest: 80% = detecta a 80 de 100 desertores

### ¿Qué es Precisión?
De los que predice que van a desertar, ¿cuántos realmente desertan?
- Random Forest: 82% = de 276 predichos, 226 desertan realmente

---

## 🎓 PRINCIPALES FACTORES DE DESERCIÓN

**Según los modelos, los MAYORES predictores de deserción son:**

1. **Unidades aprobadas en 2° semestre** ⭐ MÁS IMPORTANTE
   - Muy pocas aprobadas → Alto riesgo de deserción
   
2. **Calificación en 2° semestre**
   - Baja calificación → Alto riesgo
   
3. **Unidades aprobadas en 1° semestre**
   - Pocas aprobadas → Señal de alerta temprana
   
4. **Pago de matrícula al día**
   - No pagar → Mayor riesgo de deserción
   
5. **Edad al matricularse**
   - Estudiantes mayores pueden tener más dificultades

**Factores PROTECTORES (reducen deserción):**
- ✅ Pagar la matrícula a tiempo
- ✅ Tener becas
- ✅ Ser estudiante internacional
- ✅ Ocupación de la madre

---

## 📈 RECOMENDACIONES

### Si quieres **máxima precisión:**
→ Usa **Random Forest** (87.8% accuracy)

### Si quieres **detectar a TODOS los desertores:**
→ Usa **Regresión Logística** (87% recall)

### Si quieres **explicar por qué desertan:**
→ Usa **Regresión Logística** (coeficientes claros)

### Si necesitas **balance óptimo:**
→ Usa **Random Forest** + alertas basadas en variables clave

---

## 💡 INTERVENCIONES RECOMENDADAS

Basándote en los resultados, puedes:

1. **Monitorear desempeño en 1° semestre**
   - Si aprueba <4 materias → Derivar a tutoría
   
2. **Seguimiento en 2° semestre**
   - Variable más importante para predicción
   
3. **Programas de apoyo financiero**
   - Estudiantes con dificultades de pago
   
4. **Mentoría para estudiantes mayores**
   - Edad al matricularse es predictor
   
5. **Becas y apoyo institucional**
   - Reducen significativamente riesgo

---

## 📁 MODELOS GENERADOS

Cada carpeta contiene:
- `modelo_xgboost_academico.pkl` 
- `modelo_randomforest_academico.pkl`
- `modelo_logistico_academico.pkl` + `scaler_academico.pkl`

---

## 🚀 CÓMO USAR LOS MODELOS

### Para predicciones nuevas:

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load("modelo_randomforest_academico.pkl")

# Cargar datos nuevos
df_nuevo = pd.read_csv("nuevos_estudiantes.csv", sep=";")

# Hacer predicciones
predicciones = modelo.predict_proba(df_nuevo)
riesgo_desercion = predicciones[:, 1]  # Probabilidad de deserción
```

---

## 📊 MATRIZ DE CONFUSIÓN EXPLICADA

```
Random Forest Results:

                   Predicción
                No Des  Desertor
Real      No Des  551    50      ← 550 correctos, 50 errores
          Desertor 58    226     ← 226 correctos, 58 errores

- 551: Estudiantes bien clasificados (no desertor)
- 226: Desertores bien identificados
- 50: Falsa alarma (predijo desertor pero no fue)
- 58: Desertores no detectados
```

---

## 🎯 CONCLUSIÓN

**Random Forest es el modelo recomendado** porque:
- ✅ Mejor accuracy general (87.8%)
- ✅ Mejor AUC-ROC (0.9294)
- ✅ Buen balance entre detectar desertores (80%) y no tener falsas alarmas
- ✅ Variables importantes claras y interpretables
- ✅ Robusto a datos nuevos

**Use Regresión Logística si necesita** máxima interpretabilidad y detectar todos los desertores posibles.

