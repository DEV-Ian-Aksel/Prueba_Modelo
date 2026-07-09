import pandas as pd
import joblib

# ── Cargar modelo y codificadores ──
modelo = joblib.load("modelos_entrenados_xgboost/modelo_desercion_xgboost.pkl")
label_encoders = joblib.load("modelos_entrenados_xgboost/label_encoders.pkl")

# ── Cargar datos de predicción ──
df = pd.read_csv("xgboost/Dataset_Pruebas_Predictivo_Simulado.csv")
df.columns = df.columns.str.strip()  # limpiar espacios en nombres

print(f"Registros cargados: {len(df)}")

# ── Guardar para referencia y evaluación posterior ──
tiene_deserto = "deserto" in df.columns
if tiene_deserto:
    y_test = df["deserto"].copy()

# ── Aplicar el MISMO procesamiento que en entrenamiento ──

# Quitar las mismas columnas que se quitaron en entrenamiento
cols_drop = [
    'Matricula', 'Estatus', 'Grupo',
    'Estado Nacimiento', 'Municipio Nacimiento', 'Estado', 'Municipio',
    'Bachillerato de Procedencia', 'Bachillerato (Estado)', 'Bachillerato (Municipio)',
    'deserto'  # Si existe, quitarla para predicción
]
cols_a_quitar = [col for col in cols_drop if col in df.columns]
df_procesado = df.drop(columns=cols_a_quitar).copy()

print(f"Columnas eliminadas: {cols_a_quitar}")

# ── Codificar las mismas columnas de texto que se codificaron en entrenamiento ──
for col, le in label_encoders.items():
    if col in df_procesado.columns:
        df_procesado[col] = le.transform(df_procesado[col].astype(str))
        print(f"  ✓ {col} codificada")

X = df_procesado

# ── Realizar predicción ──
probs = modelo.predict_proba(X)
df["probabilidad_desercion"] = probs[:, 1]

umbral = 0.40
df["riesgo"] = (df["probabilidad_desercion"] >= umbral).astype(int)

print(f"\nPredicciones realizadas (umbral = {umbral})")
print(f"  Estudiantes en riesgo: {df['riesgo'].sum()}")
print(f"  Estudiantes sin riesgo: {(df['riesgo'] == 0).sum()}")

print("\nPrimeros registros:")
print(df.head())

# ── Guardar predicciones ──
df.to_csv("prediccion_csv_xgboost/predicciones.csv", index=False)
print("\nPredicciones guardadas en: predicciones.csv")

# ── Evaluación (si tenemos etiqueta verdadera) ──
if tiene_deserto:
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    acc = accuracy_score(y_test, df["riesgo"])
    auc = roc_auc_score(y_test, df["probabilidad_desercion"])
    
    print("\n" + "="*40)
    print("EVALUACIÓN DEL MODELO")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")

    cm = confusion_matrix(y_test, df["riesgo"])
    tn, fp, fn, tp = cm.ravel()

    print("\nMatriz de Confusión:")
    print(cm)
    print(f"\n  TN (acertó que NO desertan):  {tn}")
    print(f"  TP (acertó que SÍ desertan):  {tp}")
    print(f"  FP (falsa alarma):            {fp}")
    print(f"  FN (desertor no detectado):   {fn}")

print("\n✓ Proceso completado.")