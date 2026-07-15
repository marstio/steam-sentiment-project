"""Enrich the indie game list with metadata from Steam's public appdetails API."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "steam_indie_games_list.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "steam_games_enriched.csv"


def names_from_list(items):
    """Convert Steam's [{"name": "..."}] format into a readable string."""
    return ", ".join(item.get("name", "") for item in (items or []) if item.get("name"))


def fetch_game_details(appid, session):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    response = session.get(url, timeout=30)
    response.raise_for_status()

    payload = response.json().get(str(appid), {})
    if not payload.get("success"):
        raise ValueError("Steam returned no app details")

    details = payload.get("data", {})
    return {
        "short_description": details.get("short_description", ""),
        "developer": ", ".join(details.get("developers", [])),
        "publisher": ", ".join(details.get("publishers", [])),
        "genres": names_from_list(details.get("genres")),
        "enrichment_status": "success",
        "enrichment_error": "",
    }


def enrich_games(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, delay=3.0, limit=None):
    games = pd.read_csv(input_path, dtype={"AppID": "string"})
    required_columns = {"AppID", "Name"}
    missing = required_columns - set(games.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    if limit is not None:
        games = games.head(limit).copy()

    session = requests.Session()
    session.headers.update({"User-Agent": "steam-sentiment-project/1.0"})
    enriched_rows = []

    for index, row in games.iterrows():
        appid = str(row["AppID"]).strip()
        print(f"[{index + 1}/{len(games)}] Fetching {appid} ({row['Name']})...")

        try:
            metadata = fetch_game_details(appid, session)
        except (requests.RequestException, ValueError) as exc:
            metadata = {
                "short_description": "",
                "developer": "",
                "publisher": "",
                "genres": "",
                "enrichment_status": "failed",
                "enrichment_error": str(exc),
            }
            print(f"  Failed: {exc}")

        enriched_rows.append({**row.to_dict(), **metadata})
        if index < len(games) - 1:
            time.sleep(delay)

    enriched = pd.DataFrame(enriched_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(enriched)} games to {output_path}")
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich Steam games with appdetails metadata.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--limit", type=int, help="Process only the first N rows for testing.")
    args = parser.parse_args()
    enrich_games(args.input, args.output, args.delay, args.limit)


if __name__ == "__main__":
    main()
