import os
import sys

import fastf1
import pandas as pd
from tqdm.auto import tqdm

CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

SEASONS = [2022, 2023, 2024]
OUTPUT_FILE = "f1_dataset_2022_2024_dry.csv"
LOAD_KWARGS = {"laps": True, "telemetry": False, "weather": False}

all_data = []
season_schedules = {season: fastf1.get_event_schedule(season) for season in SEASONS}
total_events = sum(len(schedule) for schedule in season_schedules.values())
progress_bar = tqdm(
    total=total_events,
    desc="Processando eventos",
    unit="evento",
    leave=True,
    file=sys.stdout,
)

for season in SEASONS:
    print(f"\n=== Temporada {season} ===")
    schedule = season_schedules[season]

    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        track_name = event["EventName"]
        progress_bar.set_postfix_str(f"{season} - {track_name}")

        progress_bar.write(f"Baixando: {track_name} ({season})")

        try:
            session = fastf1.get_session(season, round_number, "R")
            try:
                session.load(**LOAD_KWARGS)
            except TypeError:
                session.load()

            laps = session.laps

            if laps.empty:
                progress_bar.write("Sem dados, pulando...")
                continue

            laps = laps.dropna(subset=["LapTime"])

            laps = laps[(laps["PitInTime"].isna()) & (laps["PitOutTime"].isna())]

            lap_time_sec = laps["LapTime"].dt.total_seconds()

            laps = laps[(lap_time_sec > 50) & (lap_time_sec < 130)]

            lap_time_sec = laps["LapTime"].dt.total_seconds()

            if laps.empty:
                progress_bar.write("Sem voltas válidas após limpeza, pulando...")
                continue

            stint_lap = laps.groupby(["Driver", "Stint"]).cumcount() + 1

            race_lap = laps["LapNumber"]

            df = pd.DataFrame({
                "lap_time": lap_time_sec,
                "compound": laps["Compound"],
                "stint_lap": stint_lap,
                "stint_number": laps["Stint"],
                "race_lap": race_lap,
                "track": track_name,
                "team": laps["Team"],
                "driver": laps["Driver"],
            })

            df = df.dropna(subset=[
                "lap_time",
                "compound",
                "stint_number",
                "team",
                "driver",
                "race_lap",
            ])

            dry_compounds = ["SOFT", "MEDIUM", "HARD"]
            df = df[df["compound"].isin(dry_compounds)].copy()

            df["season"] = season
            df["is_wet"] = False

            all_data.append(df)

            progress_bar.write(f"Voltas válidas após limpeza: {len(df)}")

        except Exception as e:
            progress_bar.write(f"Erro em {track_name}: {e}")
        finally:
            progress_bar.update(1)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv(OUTPUT_FILE, index=False)

progress_bar.write("\n==============================")
progress_bar.write("DATASET FINAL GERADO")
progress_bar.write(str(final_df.shape))
progress_bar.write(f"Arquivo salvo em: {OUTPUT_FILE}")
progress_bar.write("==============================")
progress_bar.close()
