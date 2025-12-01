import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

progress_bar = tqdm(
    total=6,
    desc="Treinando pipeline",
    unit="etapa",
    leave=True,
    file=sys.stdout,
)

progress_bar.set_postfix_str("Carregando dataset")
df = pd.read_csv("f1_dataset_2022_2024_dry.csv")
progress_bar.update(1)

TARGET = "lap_time"

FEATURES = [
    "compound",
    "stint_lap",
    "stint_number",
    "race_lap",
    "track",
    "team",
]

X = df[FEATURES]
y = df[TARGET]
progress_bar.update(1)

progress_bar.set_postfix_str("Separando treino/teste")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
progress_bar.update(1)

categorical_features = ["compound", "track", "team"]
numerical_features = ["stint_lap", "stint_number", "race_lap"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

progress_bar.set_postfix_str("Treinando modelo")
progress_bar.write("Treinando modelo...")
pipeline.fit(X_train, y_train)
progress_bar.update(1)

progress_bar.set_postfix_str("Avaliando")
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
progress_bar.update(1)

progress_bar.write(f"MAE (erro médio absoluto): {mae:.3f} segundos")

joblib.dump(pipeline, "pitstop_model_v1.pkl")
progress_bar.update(1)

progress_bar.write("Modelo salvo como pitstop_model_v1.pkl")
progress_bar.close()
