import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

modelo = joblib.load("modelo_randomforest.pkl")

df = pd.read_csv("../Practica_modelo/alumnos_nuevos_dificil.csv")

# Si existe columna deserto, la quitamos
if "deserto" in df.columns:
    X = df.drop("deserto", axis=1)
else:
    X = df

probs = modelo.predict_proba(X)

df["probabilidad_desercion"] = probs[:, 1]

umbral = 0.40

df["riesgo"] = (
    df["probabilidad_desercion"] >= umbral
).astype(int)

print(df.head())

df.to_csv(
    "predicciones_rf.csv",
    index=False
)

if "deserto" in df.columns:
    acc = accuracy_score(df["deserto"], df["riesgo"])
    auc = roc_auc_score(df["deserto"], df["probabilidad_desercion"])
    print(f"\nAccuracy sobre datos nuevos: {acc:.4f}")
    print(f"AUC-ROC sobre datos nuevos:  {auc:.4f}")

    cm = confusion_matrix(df["deserto"], df["riesgo"])
    tn, fp, fn, tp = cm.ravel()

    print("\nMatriz de Confusión:")
    print(cm)
    print(f"\n  Acertó que NO desertan (TN): {tn}")
    print(f"  Acertó que SÍ desertan (TP): {tp}")
    print(f"  Falsa alarma — dijo desertor pero no lo era (FP): {fp}")
    print(f"  Desertor no detectado (FN): {fn}")

print("Predicciones guardadas como 'predicciones_rf.csv'")
