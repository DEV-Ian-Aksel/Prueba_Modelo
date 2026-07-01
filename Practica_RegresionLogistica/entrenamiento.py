import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# Cargar dataset
# ==========================================

print("Cargando dataset académico...")

df = pd.read_csv("Practica_RandomForest/Muestra_Limpia_Predictivo_SEP-DIC2025.csv")

df.columns = df.columns.str.strip()

print(f"Registros encontrados: {len(df)}")
print(f"Valores en Estatus: {df['Estatus'].unique()}")

# ==========================================
# Filtrar clases
# ==========================================

df = df[df['Estatus'].isin(['Baja Definitiva', 'Regular'])].copy()

print(f"Registros tras filtro: {len(df)}")

# ==========================================
# Variable objetivo
# ==========================================

df['desertor'] = (df['Estatus'] == 'Baja Definitiva').astype(int)

print(f"Desertores: {df['desertor'].sum()}")
print(f"No desertores: {(df['desertor'] == 0).sum()}")

# ==========================================
# Eliminar columnas no útiles
# ==========================================

cols_drop = [
    'Matricula',
    'Estatus',
    'Grupo',
    'Estado Nacimiento',
    'Municipio Nacimiento',
    'Estado',
    'Municipio',
    'Bachillerato de Procedencia',
    'Bachillerato (Estado)',
    'Bachillerato (Municipio)'
]

df = df.drop(columns=cols_drop)

# ==========================================
# Codificar variables de texto
# ==========================================

columnas_texto = df.select_dtypes(
    include=['object', 'string']
).columns.tolist()

print("\nColumnas de texto a codificar:")
print(columnas_texto)

for col in columnas_texto:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"✓ {col} codificada")

# ==========================================
# Separar X e Y
# ==========================================

X = df.drop('desertor', axis=1)
y = df['desertor']

print("\nVariables usadas:")
print(list(X.columns))

# ==========================================
# Escalado (MUY IMPORTANTE)
# ==========================================

print("\nEscalando variables...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("✓ Escalado completado")

# ==========================================
# División entrenamiento/prueba
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

print(f"\nEntrenamiento: {len(X_train)}")
print(f"Prueba: {len(X_test)}")

# ==========================================
# Modelo Regresión Logística
# ==========================================

modelo = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    class_weight="balanced",
    C=1.0,
    random_state=42
)

print("\n✓ Modelo Regresión Logística creado")

# ==========================================
# Validación cruzada
# ==========================================

print("\nValidando con cross-validation (5-fold)...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_auc = cross_val_score(
    modelo,
    X_scaled,
    y,
    cv=cv,
    scoring="roc_auc"
)

print(f"AUC-ROC promedio: {cv_auc.mean():.4f} (±{cv_auc.std():.4f})")

# ==========================================
# Entrenamiento
# ==========================================

print("\nEntrenando modelo final...")

modelo.fit(X_train, y_train)

print("Entrenamiento completado.")

# ==========================================
# Evaluación
# ==========================================

probs = modelo.predict_proba(X_test)

predicciones = (probs[:, 1] > 0.40).astype(int)

accuracy = accuracy_score(y_test, predicciones)
auc = roc_auc_score(y_test, probs[:, 1])

cm = confusion_matrix(y_test, predicciones)

tn, fp, fn, tp = cm.ravel()

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

print("\nMatriz de Confusión:")
print(cm)

print(f"\nTN (acertó no desertor): {tn}")
print(f"TP (acertó desertor):    {tp}")
print(f"FP (falsa alarma):       {fp}")
print(f"FN (desertor no visto):  {fn}")

print("\nReporte de Clasificación:")

print(
    classification_report(
        y_test,
        predicciones,
        target_names=['Regular', 'Baja Definitiva']
    )
)

# ==========================================
# Guardar modelo
# ==========================================

joblib.dump(modelo, "modelo_logistico.pkl")
joblib.dump(scaler, "scaler_logistico.pkl")

print("\nModelo guardado como:")
print("modelo_logistico.pkl")

print("\nScaler guardado como:")
print("scaler_logistico.pkl")

# ==========================================
# Importancia de variables
# ==========================================

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_[0]
})

coeficientes["Impacto"] = coeficientes["Coeficiente"].abs()

coeficientes = coeficientes.sort_values(
    by="Impacto",
    ascending=False
)

print("\nImportancia de variables:")
print(
    coeficientes[
        ["Variable", "Coeficiente"]
    ].to_string(index=False)
)