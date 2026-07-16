import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

print("Cargando dataset académico...")
df = pd.read_csv("randomforest/Muestra_Limpia_Predictivo_SEP-DIC2025_2.csv")
df.columns = df.columns.str.strip()
print(f"Registros encontrados: {len(df)}")
print(f"Valores en Estatus: {df['Estatus'].unique()}")

# ── Filtrar solo las dos clases que nos interesan ──
df = df[df['Estatus'].isin(['Baja Definitiva', 'Regular'])].copy()
print(f"Registros tras filtro: {len(df)}")

# ── Variable objetivo ──
df['desertor'] = (df['Estatus'] == 'Baja Definitiva').astype(int)
print(f"Desertores: {df['desertor'].sum()}  |  No desertores: {(df['desertor'] == 0).sum()}")

# ── Quitar columnas que no usaremos ──
cols_drop = [
    'Matricula', 'Estatus', 'Grupo',
    'Estado Nacimiento', 'Municipio Nacimiento', 'Estado', 'Municipio',
    'Bachillerato de Procedencia', 'Bachillerato (Estado)', 'Bachillerato (Municipio)'
]
df = df.drop(columns=cols_drop)

# ── Codificar texto restante (Oferta Educativa, Genero) ──
columnas_texto = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nColumnas de texto a codificar: {columnas_texto}")

orden_grado = {
    'primero':        1,
    'segundo':        2,
    'tercero':        3,
    'cuarto':         4,
    'quinto':         5,
    'sexto':          6,
    'séptimo':        7,
    'octavo':         8,
    'noveno':         9,
    'décimo':         10,
    'undécimo':       11,
    'duodécimo':      12,
    'décimo tercero': 13
}

df['Grado'] = df['Grado'].str.strip().str.lower().map(orden_grado)

columnas_texto = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nColumnas de texto a codificar: {columnas_texto}")

# Guardar los LabelEncoders en un diccionario
label_encoders = {}
for col in columnas_texto:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

    print(f"\nCodificación de {col}:")
    for i, clase in enumerate(le.classes_):
        print(f"  {clase} -> {i}")

X = df.drop('desertor', axis=1)
y = df['desertor']
print(f"\nVariables usadas: {list(X.columns)}")

# ── Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nEntrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

# ── Modelo ──
modelo = RandomForestClassifier(
    n_estimators=350,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)


# ── Cross-validation ──
print("\nValidando con cross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")
print(f"AUC-ROC promedio: {cv_auc.mean():.4f} (±{cv_auc.std():.4f})")

# ── Entrenamiento ──
print("\nEntrenando modelo final...")
modelo.fit(X_train, y_train)
print("Entrenamiento completado.")

# ── Evaluación ──
probs = modelo.predict_proba(X_test)
predicciones = (probs[:, 1] > 0.40).astype(int)

accuracy = accuracy_score(y_test, predicciones)
auc      = roc_auc_score(y_test, probs[:, 1])
cm       = confusion_matrix(y_test, predicciones)
tn, fp, fn, tp = cm.ravel()

print("\n==============================")
print("RESULTADOS")
print("==============================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"\nMatriz de Confusión:\n{cm}")
print(f"\n  TN (acertó no desertor): {tn}")
print(f"  TP (acertó desertor):    {tp}")
print(f"  FP (falsa alarma):       {fp}")
print(f"  FN (desertor no visto):  {fn}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones, target_names=['Regular', 'Baja Definitiva']))

# ── Guardar ──
joblib.dump(modelo, "modelos_entrenados_rf/modelo_randomforest.pkl")
joblib.dump(label_encoders, "modelos_entrenados_rf/label_encoders.pkl")
print("\nModelo guardado como: modelo_randomforest.pkl")
print("LabelEncoders guardados como: label_encoders.pkl")

# ── Importancia ──
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)
print("\nImportancia de variables:")
print(importancias.to_string(index=False))