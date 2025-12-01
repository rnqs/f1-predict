import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm


progress_bar = tqdm(total=5, desc="Treinando RandomForest", unit="etapa", file=sys.stdout)
progress_bar.set_postfix_str("Carregando CSV")
df = pd.read_csv("f1_race_results_2022_2024.csv")
progress_bar.update(1)

df = df.dropna(subset=["finish_in_points"]).reset_index(drop=True)

TARGET = "finish_in_points"

FEATURES = [
    "grid_position",
    "team",
    "track",
    "driver_points_before",
    "team_points_before",
    "season",
]

X = df[FEATURES]
y = df[TARGET].astype(int)
progress_bar.update(1)
progress_bar.set_postfix_str("Separando treino/teste")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # mantém proporção de 0/1
)
progress_bar.update(1)

categorical_features = ["team", "track", "season"]
numerical_features = ["grid_position", "driver_points_before", "team_points_before"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    class_weight="balanced",  # dataset vai ter mais 0 do que 1
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf),
    ]
)

progress_bar.set_postfix_str("Treinando modelo")
progress_bar.write("Treinando RandomForestClassifier...")
pipeline.fit(X_train, y_train)
progress_bar.update(1)

progress_bar.set_postfix_str("Avaliando")

y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print("\n=== MÉTRICAS DO MODELO (finish_in_points) ===")
print(f"Accuracy: {acc:.3f}\n")
print("Matriz de confusão (linhas = verdade, colunas = predito):")
print(cm)
print("\nRelatório de classificação:")
print(report)

progress_bar.update(1)

MODEL_PATH = "race_result_rf.pkl"
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModelo salvo em: {MODEL_PATH}")

progress_bar.close()
