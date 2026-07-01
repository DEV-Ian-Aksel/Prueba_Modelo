# COMPARATIVA: XGBoost vs Random Forest vs Regresión Logística

## 📊 Comparación General de los 3 Modelos

### 1. **ALGORITMO**

**XGBoost (Extreme Gradient Boosting):**
- ✅ Técnica: **Boosting** (secuencial)
- Cada árbol intenta corregir los errores del anterior
- Los árboles se entrenan DE FORMA SECUENCIAL
- Más "astuto" pero requiere más tiempo

**Random Forest:**
- ✅ Técnica: **Bagging** (paralelo)
- Los árboles son independientes entre sí
- Se pueden entrenar EN PARALELO
- Más simple pero muy efectivo

**Regresión Logística:**
- ✅ Técnica: **Modelo lineal**
- Combina variables linealmente
- Aplica función logística
- Muy simple y ultra rápido

---

### 2. **HIPERPARÁMETROS**

| Parámetro | XGBoost | Random Forest | Regresión Logística |
|-----------|---------|---------------|-----------------|
| n_estimators | 350 árboles | 200 árboles | N/A (1 modelo) |
| max_depth | 3 (profundidad baja) | 15 (profundidad alta) | N/A |
| learning_rate | 0.01 (aprendizaje lento) | N/A | N/A |
| subsample | 0.85 | N/A | N/A |
| solver | N/A | N/A | LBFGS |
| C (regularización) | N/A | N/A | 1.0 |
| Escalado de variables | No necesario | No necesario | ✅ Sí, obligatorio |
| complexity | Alta | Baja | Muy baja |

---

### 3. **VELOCIDAD**

**XGBoost:**
- ⏱️ Entrenamiento: **Más lento**
- Razón: Los árboles se entrenan secuencialmente
- No se paraleliza bien

**Random Forest:**
- ⚡ Entrenamiento: **Más rápido**
- Razón: Los árboles se entrenan en paralelo
- Usa todos los cores del procesador

**Regresión Logística:**
- ⚡⚡⚡ Entrenamiento: **ULTRA RÁPIDO**
- Razón: Es un modelo simple, converge en iteraciones
- Entrena en milisegundos

---

### 4. **RENDIMIENTO PREDICTIVO**

**XGBoost:**
- 🏆 Performance: **Típicamente superior**
- Muy usado en competiciones de ML
- Mejor en datasets complejos
- Requiere más tuning para no sobreajustar

**Random Forest:**
- 📊 Performance: **Muy bueno**
- Más robusto y menos propenso a sobreajuste
- Menos necesidad de tuning
- Excelente "modelo por defecto"

**Regresión Logística:**
- 📈 Performance: **Bueno en datos lineales**
- Excelente baseline para comparar
- Puede fallar con datos no-lineales
- Mejor con features bien preparadas

---

### 5. **SOBREAJUSTE**

**XGBoost:**
- ⚠️ Riesgo: **Alto**
- Muy flexible, puede memorizar datos
- Requiere cuidado con parámetros

**Random Forest:**
- ✅ Riesgo: **Bajo**
- Regularización natural por naturaleza del algoritmo
- Más resistente al ruido

**Regresión Logística:**
- ✅ Riesgo: **Muy bajo**
- Es un modelo muy simple
- Casi imposible sobreajustar
- Regularización L2 incluida

---

### 6. **INTERPRETABILIDAD**

**XGBoost:**
- ❌ Interpretabilidad: **Difícil**
- "Caja negra" más oscura
- Difícil explicar por qué predice X

**Random Forest:**
- ✅ Interpretabilidad: **Muy buena**
- Importancia de features muy clara
- Fácil ver qué variables influyen en la predicción
- **Bonus:** Puedes inspeccionar árboles individuales

**Regresión Logística:**
- ✅✅✅ Interpretabilidad: **PERFECTA**
- Coeficientes = impacto directo de cada variable
- ¿Coeficiente positivo? → Aumenta riesgo
- ¿Coeficiente negativo? → Disminuye riesgo
- Magnitud = fuerza del efecto

---

### 7. **USO DE MEMORIA**

**XGBoost:**
- 💾 Memoria: **Moderada**
- Optimizado

**Random Forest:**
- 💾 Memoria: **Mayor**
- Guarda 200 árboles completos

**Regresión Logística:**
- 💾 Memoria: **Mínima**
- Solo almacena coeficientes (vector)

---

### 8. **CASOS DE USO**

**Usa XGBoost cuando:**
- ✅ Necesitas máxima precisión
- ✅ Tienes recursos computacionales (tiempo de entrenamiento)
- ✅ Los datos son complejos y no lineales
- ✅ Es una competición o proyecto crítico

**Usa Random Forest cuando:**
- ✅ Necesitas rapidez
- ✅ Necesitas interpretabilidad
- ✅ No quieres hacer tuning exhaustivo
- ✅ Es un prototipo o análisis exploratorio
- ✅ Necesitas entender QUÉ influye en la predicción

**Usa Regresión Logística cuando:**
- ✅ Necesitas MÁXIMA rapidez
- ✅ Necesitas MÁXIMA interpretabilidad
- ✅ Requieres explicabilidad para stakeholders
- ✅ Los datos tienen relaciones lineales
- ✅ Es un baseline para comparar
- ✅ Necesitas un modelo minimalista

---

## 🎯 PARA ESTE PROYECTO (Deserción Universitaria)

### Decisión:
**Los tres modelos ofrecen diferentes ventajas:**

| Modelo | Ventaja |
|--------|---------|
| **XGBoost** | 🏆 Máxima precisión en predicciones |
| **Random Forest** | 🔍 Excelente equilibrio precisión-interpretabilidad |
| **Regresión Logística** | 📊 Máxima interpretabilidad y rapidez |

### Recomendación por objetivo:

**Si SOLO quieres predecir correctamente:**
→ Usa **XGBoost**

**Si quieres equilibrio entre precisión y entender el modelo:**
→ Usa **Random Forest**

**Si necesitas explicar a directivos por qué desertan los alumnos:**
→ Usa **Regresión Logística**

**Si quieres lo mejor de todo (ensemble):**
→ Entrena los 3 y promedia predicciones

---

## 📈 ¿CÓMO COMPARAR RESULTADOS DE LOS 3 MODELOS?

Ejecuta los tres scripts y compara:

```bash
# Terminal 1: XGBoost
cd Practica_modelo
python entrenamiento.py
python prediccion.py

# Terminal 2: Random Forest
cd ../Practica_RandomForest
python entrenamiento.py
python prediccion.py

# Terminal 3: Regresión Logística
cd ../Practica_RegresionLogistica
python entrenamiento.py
python prediccion.py
```

**Luego compara métricas:**
- **Accuracy:** ¿Cuál es más preciso?
- **AUC-ROC:** ¿Cuál discrimina mejor?
- **Tiempo entrenamiento:** ¿Cuál es más rápido?
- **Interpretabilidad:** ¿Cuál explica mejor?

**Crea una tabla de comparación:**

| Métrica | XGBoost | Random Forest | Regresión Logística |
|---------|---------|---------------|-----------------|
| Accuracy | ? | ? | ? |
| AUC-ROC | ? | ? | ? |
| Tiempo (seg) | ? | ? | ? |
| Interpretabilidad | ❌ | ✅✅ | ✅✅✅ |

