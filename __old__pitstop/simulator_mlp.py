import joblib
import pandas as pd


MODEL_PATH = "pitstop_model_mlp.pkl"
model = joblib.load(MODEL_PATH)


def validar_estrategia(estrategia, race_laps=None):
    """Valida se a estratégia faz sentido básico."""
    stints = estrategia["stints"]
    total_laps = sum(s["laps"] for s in stints)

    if race_laps is not None and total_laps != race_laps:
        print(
            f"[AVISO] Estratégia '{estrategia['nome']}' tem {total_laps} voltas, "
            f"mas a corrida foi configurada com {race_laps} voltas."
        )

    for s in stints:
        if s["laps"] <= 0:
            raise ValueError(
                f"Stint com voltas <= 0 na estratégia '{estrategia['nome']}'"
            )
        if s["compound"] not in ["SOFT", "MEDIUM", "HARD"]:
            print(
                f"[AVISO] Composto '{s['compound']}' não é SOFT/MEDIUM/HARD "
                f"na estratégia '{estrategia['nome']}'."
            )


def gerar_df_para_estrategia(estrategia):
    """Gera o DataFrame de entrada pro modelo, sem somar tempo ainda."""
    track = estrategia["track"]
    stints = estrategia["stints"]
    team = estrategia.get("team", "Unknown")

    dados = []
    stint_number = 1
    lap_number = 1

    for stint in stints:
        compound = stint["compound"]
        laps = stint["laps"]

        for stint_lap in range(1, laps + 1):
            dados.append(
                {
                    "lap_number": lap_number,
                    "compound": compound,
                    "stint_lap": stint_lap,
                    "stint_number": stint_number,
                    "race_lap": lap_number,
                    "track": track,
                    "team": team,
                }
            )
            lap_number += 1

        stint_number += 1

    df = pd.DataFrame(dados)
    return df


def simular_estrategia(estrategia, incluir_pit=True):
    """
    Simula UMA estratégia, usando o modelo para prever tempo de volta
    e somando tempo de pit stop entre stints.
    """
    validar_estrategia(estrategia)

    df = gerar_df_para_estrategia(estrategia)

    X = df[["compound", "stint_lap", "stint_number", "race_lap", "track", "team"]]

    df["lap_time"] = model.predict(X)

    pit_loss = estrategia.get("pit_loss", 22.0)
    n_stints = len(estrategia["stints"])

    total_pit_time = (n_stints - 1) * pit_loss if incluir_pit and n_stints > 1 else 0.0

    tempo_acumulado = []
    tempo_corrida = 0.0
    current_stint = df.loc[0, "stint_number"]

    for i, row in df.iterrows():
        if incluir_pit and row["stint_number"] != current_stint:
            tempo_corrida += pit_loss
            current_stint = row["stint_number"]

        tempo_corrida += row["lap_time"]
        tempo_acumulado.append(tempo_corrida)

    df["tempo_acumulado"] = tempo_acumulado

    tempo_voltas = df["lap_time"].sum()
    tempo_total = tempo_voltas + total_pit_time

    resumo = {
        "nome": estrategia["nome"],
        "track": estrategia["track"],
        "team": estrategia.get("team", "Unknown"),
        "n_voltas": len(df),
        "n_stints": n_stints,
        "tempo_voltas": tempo_voltas,
        "tempo_pit": total_pit_time,
        "tempo_total": tempo_total,
    }

    return resumo, df


def comparar_estrategias(estrategias, incluir_pit=True):
    """Roda várias estratégias e retorna um ranking em DataFrame."""
    resumos = []
    detalhes = {}

    for strat in estrategias:
        resumo, df = simular_estrategia(strat, incluir_pit=incluir_pit)
        resumos.append(resumo)
        detalhes[strat["nome"]] = df

    ranking = pd.DataFrame(resumos).sort_values("tempo_total").reset_index(drop=True)
    return ranking, detalhes


if __name__ == "__main__":
    estrategia_1 = {
        "nome": "SOFT-MEDIUM-HARD (2 paradas)",
        "track": "São Paulo Grand Prix",
        "pit_loss": 22.0,
        "team": "Red Bull",
        "stints": [
            {"compound": "SOFT", "laps": 14},
            {"compound": "MEDIUM", "laps": 22},
            {"compound": "HARD", "laps": 20},
        ],
    }

    estrategia_2 = {
        "nome": "MEDIUM-HARD (1 parada)",
        "track": "São Paulo Grand Prix",
        "pit_loss": 22.0,
        "team": "Red Bull",
        "stints": [
            {"compound": "MEDIUM", "laps": 25},
            {"compound": "HARD", "laps": 31},
        ],
    }

    estrategias = [estrategia_1, estrategia_2]

    ranking, detalhes = comparar_estrategias(estrategias, incluir_pit=True)

    print("\nRANKING DE ESTRATÉGIAS (menor tempo_total primeiro):")
    print(ranking[["nome", "team", "tempo_voltas", "tempo_pit", "tempo_total"]])

    print("\nExemplo de voltas da melhor estratégia:")
    melhor_nome = ranking.iloc[0]["nome"]
    print(detalhes[melhor_nome])
