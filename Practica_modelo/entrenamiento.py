import pandas as pd
import joblib
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# Cargar dataset

print("Cargando dataset académico...")

df = pd.read_csv("datos.csv", sep=";")

print(f"Registros encontrados: {len(df)}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nPrimeras filas:")
print(df.head())

# ==========================================
# CONVERTIR VARIABLES DE TEXTO A BINARIO
# ==========================================

print("\n==============================")
print("PROCESAMIENTO DE VARIABLES")
print("==============================")

# Buscar columnas de tipo objeto (texto)
columnas_texto = df.select_dtypes(include=['object']).columns

if len(columnas_texto) > 0:
    print(f"\nEncontramos {len(columnas_texto)} variables de texto:")
    print(list(columnas_texto))
    print("Convirtiendo texto a binario...")
    
    le = LabelEncoder()
    for col in columnas_texto:
        if col != 'Target':  # No convertir la variable objetivo aún
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  ✓ {col} convertida a binario")

# Crear variable objetivo: clasificación
# 0 = Dropout, 1 = Graduate, 2 = Enrolled
print(f"\nValores en Target: {df['Target'].unique()}")

# Convertir Target a binario: Dropout (1) vs el resto (0)
df['desertor'] = (df['Target'] == 'Dropout').astype(int)

print(f"Desertores: {(df['desertor'] == 1).sum()} estudiantes")
print(f"No desertores: {(df['desertor'] == 0).sum()} estudiantes")

# Separar variables

X = df.drop(['Target', 'desertor'], axis=1)
y = df['desertor']

print(f"\nVariables independientes (X): {list(X.columns)}")
print(f"Variable dependiente (y): desertor")
print(f"Forma de X: {X.shape}")

# ==========================================
# Dividir datos
# 80% entrenamiento
# 20% prueba
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n==============================")
print("DIVISIÓN DE DATOS")
print("==============================")
print(f"Entrenamiento: {len(X_train)} registros")
print(f"Prueba:        {len(X_test)} registros")


# Crear modelo (hiperparámetros)

print("\n==============================")
print("CREANDO MODELO XGBOOST")
print("==============================")

modelo = XGBClassifier(
    objective="binary:logistic",
    n_estimators=350,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    device="cpu",
    random_state=42,
    verbose=0
)

print("✓ Modelo XGBoost creado")

print("\n==============================")
print("VALIDACIÓN CRUZADA (5-fold)")
print("==============================")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")

print(f"AUC-ROC promedio: {cv_auc.mean():.4f} (±{cv_auc.std():.4f})")

print("\n==============================")
print("ENTRENANDO MODELO...")
print("==============================")

modelo.fit(X_train, y_train)

print("✓ Entrenamiento completado")

# Evaluación

probs = modelo.predict_proba(X_test)

predicciones = (probs[:, 1] > 0.40).astype(int)

accuracy = accuracy_score(y_test, predicciones)
auc      = roc_auc_score(y_test, probs[:, 1])

print("\n" + "="*40)
print("RESULTADOS - XGBOOST (Datos Académicos)")
print("="*40)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"AUC-ROC:   {auc:.4f}")

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, predicciones)
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\n  ✓ Verdaderos negativos (TN): {tn}")
print(f"  ✓ Verdaderos positivos (TP): {tp}")
print(f"  ✗ Falsos positivos (FP): {fp}")
print(f"  ✗ Falsos negativos (FN): {fn}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones, target_names=['No Desertor', 'Desertor']))

# Guardar modelo

joblib.dump(modelo, "modelo_xgboost_academico.pkl")

print("\n✓ Modelo guardado como 'modelo_xgboost_academico.pkl'")

# Guardar modelo

joblib.dump(modelo, "modelo_desercion.pkl")

print("\nModelo guardado como:")
print("modelo_desercion.pkl")

# Importancia de variables

importancias = pd.DataFrame({
    "Variable":   X.columns,
    "Importancia": modelo.feature_importances_
})

print("\nImportancia de variables:")
print(
    importancias
    .sort_values(by="Importancia", ascending=False)
    .to_string(index=False)
)