# 📈 RESUMEN EJECUTIVO - COMPARATIVA DE MODELOS

## 🎯 Problema
**Predecir deserción académica** usando 4,425 registros de estudiantes con 36 características.

---

## 📊 RESULTADOS PRINCIPALES

```
┌─────────────────────┬──────────┬─────────┬────────────┐
│ MODELO              │ ACCURACY │ AUC-ROC │ MEJOR EN   │
├─────────────────────┼──────────┼─────────┼────────────┤
│ XGBoost             │ 87.0%    │ 0.9100  │ Balance    │
│ Random Forest 🏆    │ 87.8%    │ 0.9294  │ GENERAL    │
│ Regresión Logística │ 84.6%    │ 0.9274  │ Recall     │
└─────────────────────┴──────────┴─────────┴────────────┘

Recall (Detectar desertores):
  • XGBoost:              79%
  • Random Forest:        80%
  • Regresión Logística:  87% ← Detecta más desertores
```

---

## 🎓 FACTORES CLAVE DE DESERCIÓN

### ⭐ TOP 5 Predictores de Deserción

1. **Unidades aprobadas en 2° semestre** (MÁS IMPORTANTE)
   - Pocas aprobadas = Alto riesgo de deserción
   
2. **Calificación en 2° semestre**
   - Baja calificación = Riesgo
   
3. **Unidades aprobadas en 1° semestre**
   - Variable de alerta temprana
   
4. **Pago de matrícula al día**
   - No pagar = Mayor riesgo
   
5. **Edad al matricularse**
   - Estudiantes mayores = Mayor riesgo

### ✅ FACTORES PROTECTORES

- Pagar la matrícula a tiempo (-72%)
- Tener becas (-24%)
- Ser estudiante internacional (-36%)

---

## 🚀 CÓMO EJECUTAR

### Opción 1: Todos a la vez
```bash
cd /workspaces/Prueba_Modelo
bash ejecutar_todos.sh
```

### Opción 2: Individual
```bash
# Random Forest (RECOMENDADO)
cd Practica_RandomForest && python entrenamiento.py
```

---

## 🏆 RECOMENDACIÓN

### **GANADOR: Random Forest**

**Por qué:**
✅ Mejor accuracy (87.8%)
✅ Mejor AUC-ROC (0.9294)
✅ Detecta 80% de desertores
✅ Variables importantes claras
✅ Bajo riesgo de falsos positivos

---

## 📚 DOCUMENTACIÓN COMPLETA

- [RESULTADOS_ACADEMICOS.md](RESULTADOS_ACADEMICOS.md) - Análisis detallado
- [COMPARATIVA_MODELOS.md](COMPARATIVA_MODELOS.md) - Comparativa técnica
- [GUÍA_RÁPIDA.md](GUÍA_RÁPIDA.md) - Cómo ejecutar paso a paso

---

## 🔧 ARCHIVOS GENERADOS

```
Practica_modelo/
└── modelo_xgboost_academico.pkl

Practica_RandomForest/
└── modelo_randomforest_academico.pkl

Practica_RegresionLogistica/
├── modelo_logistico_academico.pkl
└── scaler_academico.pkl
```

---

## 💡 CASOS DE USO

### **Intervención Temprana**
Usar **Regresión Logística** para identificar a TODOS los estudiantes en riesgo (87% recall) y hacerles seguimiento.

### **Precisión Máxima**
Usar **Random Forest** cuando recursos son limitados y quieres máxima precisión (87.8%).

### **Interpretabilidad**
Usar **Regresión Logística** para explicar a directivos por qué un estudiante está en riesgo.

---

## 📞 ¿Qué sigue?

1. ✅ Modelos entrenados
2. 📊 Resultados analizados
3. 🎯 Factores identificados
4. 🚀 Próximo: Implementar en producción

**Recomendación:** Iniciar con Random Forest y monitorear desempeño en 2° semestre.

