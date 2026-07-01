# Random Forest - Modelo de Deserción Universitaria

## ¿Qué es Random Forest?

Random Forest es un algoritmo de aprendizaje automático que crea múltiples árboles de decisión de forma independiente y combina sus predicciones para obtener una predicción más robusta y precisa.

## Características del Modelo

- **Número de árboles:** 200
- **Profundidad máxima:** 15
- **Mínimo de muestras para dividir:** 10
- **Mínimo de muestras en hoja:** 5
- **Features por split:** √(n_features)
- **Bootstrap:** Sí (muestreo con reemplazo)

## Cómo funciona Random Forest

1. **Crea múltiples subconjuntos** de datos (muestreo con reemplazo)
2. **Entrena un árbol independiente** en cada subconjunto
3. **Para predicción:**
   - Cada árbol da su predicción
   - Se promedia el resultado de todos los árboles
   - Resultado = probabilidad final

## Ventajas vs XGBoost

| Aspecto | Random Forest | XGBoost |
|--------|---------------|---------|
| Velocidad entrenamiento | ⚡ Más rápido | 🐢 Más lento |
| Paralelización | ✅ Fácil (n_jobs=-1) | ❌ Más limitada |
| Sobreajuste | 📉 Menor (más robusto) | ⬆️ Mayor (requiere tuning) |
| Interpretabilidad | ✅ Muy buena | ❌ Más compleja |
| Performance | 📊 Generalmente bueno | 🏆 Normalmente mejor |
| Importancia features | ✅ Muy clara | ⚠️ Puede ser engañosa |

## Archivos del Proyecto

- **entrenamiento.py** - Entrena el modelo y lo guarda
- **prediccion.py** - Carga el modelo y hace predicciones
- **modelo_randomforest.pkl** - Modelo entrenado
- **predicciones_rf.csv** - Resultados de las predicciones

## Cómo ejecutar

### Paso 1: Entrenar el modelo
```bash
cd Practica_RandomForest
python entrenamiento.py
```

### Paso 2: Hacer predicciones
```bash
python prediccion.py
```

## Salida esperada

El script de entrenamiento mostrará:
- AUC-ROC promedio en validación cruzada
- Accuracy en datos de prueba
- Matriz de confusión
- Reporte de clasificación
- **Top 10 variables más importantes** para predecir deserción

El script de predicción generará:
- `predicciones_rf.csv` con probabilidades y riesgo para cada alumno
- Métricas de validación (si los datos incluyen la variable real)

## Diferencias en el output

Con Random Forest obtenemos además:
- **Importancia de características:** Qué variables son más influyentes en la predicción de deserción
- Esto es muy útil para entender qué hace que un alumno sea propenso a desertar
