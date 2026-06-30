import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

modelo = joblib.load("modelo_desercion.pkl")

df = pd.read_csv("alumnos_nuevos_dificil.csv")

# Si existe columna deserto, la quitamos
if "deserto" in df.columns:
    X = df.drop("deserto", axis=1)
else:
    X = df

probs = modelo.predict_proba(X)

df["probabilidad_desercion"] = probs[:,1]

umbral = 0.40

df["riesgo"] = (
    df["probabilidad_desercion"] >= umbral
).astype(int)

print(df.head())

df.to_csv(
    "predicciones.csv",
    index=False
)

if "deserto" in df.columns:
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    acc = accuracy_score(df["deserto"], df["riesgo"])
    auc = roc_auc_score(df["deserto"], df["probabilidad_desercion"])
    print(f"\nAccuracy sobre 10k: {acc:.4f}")
    print(f"AUC-ROC sobre 10k:  {auc:.4f}")

    cm = confusion_matrix(df["deserto"], df["riesgo"])
    tn, fp, fn, tp = cm.ravel()

    print("\nMatriz de Confusión:")
    print(cm)
    print(f"\n  Acertó que NO desertan (TN): {tn}")
    print(f"  Acertó que SÍ desertan (TP): {tp}")
    print(f"  Falsa alarma — dijo desertor pero no lo era (FP): {fp}")
    print(f"  Desertor no detectado (FN): {fn}")

print("Predicciones guardadas.")