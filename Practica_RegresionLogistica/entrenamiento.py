import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

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
        if col != 'Target':
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  ✓ {col} convertida a binario")

# Crear variable objetivo: clasificación
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
# ESCALADO DE VARIABLES (IMPORTANTE para Regresión Logística)
# ==========================================

print("\n==============================")
print("ESCALADO DE VARIABLES")
print("==============================")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("✓ Variables escaladas (StandardScaler)")

# ==========================================
# Dividir datos
# 80% entrenamiento
# 20% prueba
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
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


# Crear modelo Regresión Logística (hiperparámetros)

print("\n==============================")
print("CREANDO MODELO LOGÍSTICO")
print("==============================")

modelo = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
    class_weight="balanced",
    C=1.0,
    verbose=0
)

print("✓ Modelo Regresión Logística creado")

print("\n==============================")
print("VALIDACIÓN CRUZADA (5-fold)")
print("==============================")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

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
print("RESULTADOS - REGRESIÓN LOGÍSTICA (Datos Académicos)")
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

print("\n==============================")
print("COEFICIENTES DEL MODELO")
print("==============================")

coeficientes = pd.DataFrame({
    'variable': X.columns,
    'coeficiente': modelo.coef_[0]
}).sort_values('coeficiente', ascending=False)

print("\nTop 15 variables con mayor impacto POSITIVO:")
print(coeficientes.head(15))

print("\nTop 10 variables con mayor impacto NEGATIVO:")
print(coeficientes.tail(10))

print(f"\nIntercept (constante): {modelo.intercept_[0]:.4f}")

# Guardar modelo y scaler

joblib.dump(modelo, "modelo_logistico_academico.pkl")
joblib.dump(scaler, "scaler_academico.pkl")

print("\n✓ Modelo guardado como 'modelo_logistico_academico.pkl'")
print("✓ Scaler guardado como 'scaler_academico.pkl'")
