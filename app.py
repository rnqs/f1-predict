import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "race_result_rf.pkl"
DATASET_PATH = "f1_race_results_2022_2024.csv"

st.set_page_config(
    page_title="Simulador F1 - Previsão de Pontos",
    layout="centered",
)

# Carregamento de modelo e dados

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df

model = load_model()
df = load_dataset()

# mesmas FEATURES usadas no treino
FEATURES = [
    "grid_position",
    "team",
    "track",
    "driver_points_before",
    "team_points_before",
    "season",
]

st.title("🏁 Simulador F1 – Previsão de Pontos")

st.markdown(
    """
Este simulador usa um modelo de **Random Forest** treinado com dados reais da F1 (2022–2024)  
para estimar se um piloto **vai ou não terminar nos pontos** em um determinado cenário.
"""
)


# Sidebar – seleção de cenário

st.sidebar.header("Configuração do cenário")

# temporadas disponíveis no dataset
seasons = sorted(df["season"].unique())
season = st.sidebar.selectbox("Temporada", seasons, index=len(seasons) - 1)

# pistas disponíveis na temporada selecionada
tracks_season = (
    df[df["season"] == season]["track"]
    .dropna()
    .unique()
)
track = st.sidebar.selectbox("Pista (track)", sorted(tracks_season))

# equipes disponíveis naquela temporada
teams_season = (
    df[(df["season"] == season) & (df["track"] == track)]["team"]
    .dropna()
    .unique()
)

team = st.sidebar.selectbox("Equipe (team)", sorted(teams_season))

# posição de largada
grid_position = st.sidebar.slider("Posição de largada (grid_position)", 1, 20, 10)

# valores default de pontos antes da corrida (médias do dataset)
df_team_season = df[(df["season"] == season) & (df["team"] == team)]

default_driver_pts = float(df_team_season["driver_points_before"].median() or 0.0)
default_team_pts = float(df_team_season["team_points_before"].median() or 0.0)

st.sidebar.markdown("### Pontos acumulados antes da corrida")

driver_points_before = st.sidebar.number_input(
    "Pontos do piloto antes da corrida (driver_points_before)",
    min_value=0.0,
    max_value=500.0,
    value=round(default_driver_pts, 1),
    step=1.0,
)

team_points_before = st.sidebar.number_input(
    "Pontos da equipe antes da corrida (team_points_before)",
    min_value=0.0,
    max_value=1000.0,
    value=round(default_team_pts, 1),
    step=1.0,
)

st.sidebar.info(
    "Dica: você pode usar os valores sugeridos (mediana histórica) ou editar para testar cenários "
    "de piloto novato, equipe forte/fraca etc."
)

# Input do modelo

input_data = pd.DataFrame(
    [
        {
            "grid_position": grid_position,
            "team": team,
            "track": track,
            "driver_points_before": driver_points_before,
            "team_points_before": team_points_before,
            "season": season,
        }
    ]
)

st.subheader("Cenário escolhido")

st.write(
    f"- **Temporada:** {season}  \n"
    f"- **Pista:** {track}  \n"
    f"- **Equipe:** {team}  \n"
    f"- **Grid:** P{grid_position}  \n"
    f"- **Pontos do piloto antes:** {driver_points_before}  \n"
    f"- **Pontos da equipe antes:** {team_points_before}"
)

# Predição

if st.button("Simular resultado"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]  # [P(classe 0), P(classe 1)]

    prob_out = float(proba[0]) * 100
    prob_in = float(proba[1]) * 100

    st.markdown("---")
    st.subheader("Resultado previsto pelo modelo")

    if pred == 1:
        st.success(
            f"🔮 O modelo prevê que **VAI TERMINAR NOS PONTOS** "
            f"(probabilidade ≈ **{prob_in:.1f}%**)."
        )
        st.write(
            f"Probabilidade de **ficar fora dos pontos**: {prob_out:.1f}%"
        )
    else:
        st.error(
            f"🔮 O modelo prevê que **NÃO DEVE TERMINAR NOS PONTOS** "
            f"(probabilidade de marcar pontos ≈ **{prob_in:.1f}%**)."
        )
        st.write(
            f"Probabilidade de **ficar fora dos pontos**: {prob_out:.1f}%"
        )

    st.markdown("### Debug do input enviado ao modelo")
    st.dataframe(input_data)

# Histórico

with st.expander("📊 Ver histórico real parecido com esse cenário"):
    # corridas reais daquela combinação (season, track, team)
    df_hist = df[
        (df["season"] == season)
        & (df["track"] == track)
        & (df["team"] == team)
    ][
        [
            "driver",
            "grid_position",
            "final_position",
            "status",
            "points_race",
            "finish_in_points",
        ]
    ].copy()

    if df_hist.empty:
        st.write("Sem histórico real dessa combinação no dataset.")
    else:
        st.write(
            "Alguns resultados reais recentes dessa equipe nessa pista "
            f"na temporada {season}:"
        )
        st.dataframe(df_hist.reset_index(drop=True))