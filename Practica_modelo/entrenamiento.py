import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Cargar dataset

print("Cargando dataset...")

df = pd.read_csv("dataset_desercion_universitaria_1500.csv")

print(f"Registros encontrados: {len(df)}")

# Separar variables

X = df.drop("deserto", axis=1)
y = df["deserto"]

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
    stratify=y          # garantiza mismo % de desertores en train y test
)

print(f"\nEntrenamiento: {len(X_train)} registros")
print(f"Prueba:        {len(X_test)} registros")


# Crear modelo (hiperparámetros)

modelo = XGBClassifier(
    objective="binary:logistic",
    n_estimators=350,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.85,
    colsample_bytree=0.85,  # fracción de variables por árbol
    tree_method="hist",
    device="cpu",
    random_state=42
)

# ==========================================
# Validación cruzada antes de entrenar
# (5 particiones estratificadas)
# ==========================================

print("\nValidando con cross-validation (5-fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")

print(f"AUC-ROC promedio: {cv_auc.mean():.4f} (±{cv_auc.std():.4f})")

# ==========================================
# Entrenar con todos los datos de entrenamiento
# ==========================================

print("\nEntrenando modelo final...")

modelo.fit(X_train, y_train)

print("Entrenamiento completado.")

# Evaluación

probs = modelo.predict_proba(X_test)

predicciones = (probs[:, 1] > 0.40).astype(int)

accuracy = accuracy_score(y_test, predicciones)
auc      = roc_auc_score(y_test, probs[:, 1])

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, predicciones))

print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones))

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