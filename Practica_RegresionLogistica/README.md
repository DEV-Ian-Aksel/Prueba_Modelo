# Regresión Logística - Modelo de Deserción Universitaria

## ¿Qué es Regresión Logística?

Regresión Logística es un modelo de **clasificación lineal** que usa la **función logística** para modelar la probabilidad de pertenencia a una clase. Es simple, interpretable y muy eficiente.

## Características del Modelo

- **Tipo:** Clasificador lineal
- **Solver:** LBFGS (optimizador de segunda derivada)
- **Max iteraciones:** 1000
- **Class weight:** Balanced (maneja clases desbalanceadas)
- **C (regularización):** 1.0
- **Escalado:** SÍ (StandardScaler)

## Por qué escalado de variables?

Regresión Logística es **sensible a la escala** de las variables. Si una variable tiene valores 0-1000 y otra 0-1, la primera dominaría. Por eso usamos `StandardScaler` para normalizar todas las variables.

## Cómo funciona Regresión Logística

1. **Combina linealmente las variables** con coeficientes
2. **Aplica la función logística** para convertir en probabilidad
3. **Resultado:** Probabilidad entre 0 y 1

**Fórmula simplificada:**
```
P(deserción) = 1 / (1 + e^(-z))
donde z = coef₁*var₁ + coef₂*var₂ + ... + intercept
```

## Ventajas

✅ **Muy rápida** - entrena en milisegundos  
✅ **Altamente interpretable** - coeficientes directos  
✅ **Bajo riesgo sobreajuste** - modelo simple  
✅ **Probabilidades calibradas** - confiables  
✅ **Requisitos computacionales bajos**  

## Desventajas

❌ **No captura relaciones no-lineales**  
❌ **Puede ser muy simple** para datos complejos  
❌ **Puede bajar rendimiento** vs árboles  

## Archivos del Proyecto

- **entrenamiento.py** - Entrena el modelo y lo guarda
- **prediccion.py** - Carga el modelo y hace predicciones
- **modelo_logistico.pkl** - Modelo entrenado
- **scaler.pkl** - Escalador de variables
- **predicciones_logistica.csv** - Resultados

## Cómo ejecutar

### Paso 1: Entrenar el modelo
```bash
cd Practica_RegresionLogistica
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
- **Coeficientes del modelo** (impacto de cada variable)

Ejemplo de salida de coeficientes:
```
                   variable  coeficiente
     asistencia_promedio           0.85
     promedio_notas               0.72
     edad                        -0.15
```

Significa:
- Mayor asistencia → ↓ riesgo deserción
- Mayor promedio → ↓ riesgo deserción
- Mayor edad → ↑ riesgo deserción

## Diferencias vs Random Forest y XGBoost

| Aspecto | Regresión Logística |
|--------|-------------------|
| **Velocidad** | ⚡⚡⚡ Instantánea |
| **Interpretabilidad** | ✅✅✅ Perfecta |
| **Precisión** | 📊 Buena en datos lineales |
| **Complejidad** | 🟢 Muy simple |
| **Coeficientes** | ✅ Directamente interpretables |

## Importancia de coeficientes

Un coeficiente **positivo** = aumenta riesgo de deserción  
Un coeficiente **negativo** = disminuye riesgo de deserción  

La **magnitud** indica la fuerza del efecto.

