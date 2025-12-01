import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm

progress_bar = tqdm(total=6, desc="Treinando MLP", unit="etapa", file=sys.stdout)

df = pd.read_csv("f1_dataset_2022_2024_dry.csv")
df = df[df["race_lap"] > 1].reset_index(drop=True)
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
progress_bar.update(1)

categorical_features = ["compound", "track", "team"]
numerical_features = ["stint_lap", "stint_number", "race_lap"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

progress_bar.write("Treinando MLP...")
pipeline.fit(X_train, y_train)
progress_bar.update(1)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
progress_bar.update(1)

print(f"\nMAE MLP: {mae:.3f} segundos")

joblib.dump(pipeline, "pitstop_model_mlp.pkl")
progress_bar.update(1)

print("Modelo salvo como pitstop_model_mlp.pkl")
progress_bar.close()
