import os
import sys

import fastf1
import pandas as pd
from tqdm.auto import tqdm

CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

SEASONS = [2022, 2023, 2024]
OUTPUT_FILE = "f1_race_results_2022_2024.csv"

def safe_float(x):
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def main():
    all_rows = []

    season_schedules = {season: fastf1.get_event_schedule(season) for season in SEASONS}
    total_events = sum(len(s) for s in season_schedules.values())

    progress_bar = tqdm(
        total=total_events,
        desc="Processando temporadas",
        unit="corrida",
        file=sys.stdout,
        leave=True,
    )

    for season in SEASONS:
        print(f"\n=== Temporada {season} ===")
        schedule = season_schedules[season]

        driver_points_cumulative = {}
        team_points_cumulative = {}

        schedule = schedule.sort_values("RoundNumber")

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            track_name = event["EventName"]

            progress_bar.set_postfix_str(f"{season} - {track_name}")
            progress_bar.write(f"Baixando resultados: {track_name} ({season})")

            try:
                session = fastf1.get_session(season, round_number, "R")
                session.load()

                results = session.results  # DataFrame com resultado da corrida

                if results is None or results.empty:
                    progress_bar.write("Sem resultados, pulando...")
                    progress_bar.update(1)
                    continue

                for _, row in results.iterrows():
                    driver = row.get("Abbreviation") or row.get("BroadcastName") or row.get("DriverNumber")
                    team = row.get("TeamName")
                    position = row.get("Position")
                    grid = row.get("GridPosition")
                    status = row.get("Status")
                    points = safe_float(row.get("Points"))

                    if pd.isna(driver) or pd.isna(team):
                        continue

                    driver_pts_before = driver_points_cumulative.get(driver, 0.0)
                    team_pts_before = team_points_cumulative.get(team, 0.0)

                    try:
                        pos_int = int(position)
                    except Exception:
                        pos_int = None

                    finish_in_points = None
                    if pos_int is not None:
                        finish_in_points = 1 if pos_int <= 10 else 0

                    row_dict = {
                        "season": season,
                        "round": round_number,
                        "track": track_name,
                        "driver": driver,
                        "team": team,
                        "grid_position": grid,
                        "final_position": position,
                        "status": status,
                        "points_race": points,
                        "driver_points_before": driver_pts_before,
                        "team_points_before": team_pts_before,
                        "finish_in_points": finish_in_points,
                    }

                    all_rows.append(row_dict)

                    driver_points_cumulative[driver] = driver_pts_before + points
                    team_points_cumulative[team] = team_pts_before + points

            except Exception as e:
                progress_bar.write(f"Erro em {track_name} ({season}): {e}")

            progress_bar.update(1)

    progress_bar.close()

    df_final = pd.DataFrame(all_rows)

    df_final = df_final.sort_values(
        ["season", "round", "final_position"], ignore_index=True
    )

    df_final.to_csv(OUTPUT_FILE, index=False)
    print("\n==============================")
    print("DATASET FINAL GERADO")
    print(df_final.shape)
    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print("==============================")


if __name__ == "__main__":
    main()
