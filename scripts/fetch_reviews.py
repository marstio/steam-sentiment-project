import requests
import time
import json
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "data" / "logs"

def get_reviews(appid, max_reviews=1000, delay=1.0):
    reviews = []
    # Steam uses '*' to represent the very first page of reviews
    cursor = '*'
    
    print(f"Fetching reviews for App ID: {appid}...")
    
    while len(reviews) < max_reviews:
        # The public Store API endpoint
        url = f"https://store.steampowered.com/appreviews/{appid}"
        params = {
            "json": 1,
            "filter": "recent",
            "language": "english",
            "cursor": cursor,
            "num_per_page": min(100, max_reviews - len(reviews)),
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": "steam-sentiment-project/1.0"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching data for {appid}: {exc}")
            break
        
        data = response.json()
        
        if 'reviews' not in data or not data['reviews']:
            print("No more reviews found.")
            break
            
        new_reviews = data['reviews']
        reviews.extend(new_reviews)
        
        # Steam gives us the next cursor to use in the response
        next_cursor = data.get('cursor')
        
        # If the cursor hasn't changed, we've hit the end of the reviews
        if next_cursor == cursor:
            break
            
        if not next_cursor:
            print("Steam did not provide another cursor.")
            break

        cursor = next_cursor
        
        print(f"Fetched {len(reviews)} reviews so far...")
        
        # MANDATORY: Add a 1-second delay so Steam doesn't block our IP address
        time.sleep(delay)
        
    return reviews[:max_reviews]

def load_appids(appid_args, appid_file):
    appids = list(appid_args or [])
    if appid_file:
        path = Path(appid_file)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if "AppID" not in (reader.fieldnames or []):
                    raise ValueError("CSV must contain an AppID column")
                appids.extend((row.get("AppID") or "").strip() for row in reader)
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            appids.extend(line.split("#", 1)[0].strip() for line in lines)
    return list(dict.fromkeys(appid for appid in appids if appid))


def write_log(log_path, appid, status, review_count=0, error=None):
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "appid": appid,
        "status": status,
        "review_count": review_count,
    }
    if error:
        record["error"] = error
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch public Steam reviews.")
    parser.add_argument("--appid", action="append", help="Steam App ID; repeat for multiple games.")
    parser.add_argument("--appid-file", help="Text file or CSV containing Steam App IDs.")
    parser.add_argument("--max-reviews", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API pages.")
    parser.add_argument("--force", action="store_true", help="Refetch files that already exist.")
    args = parser.parse_args()

    appids = load_appids(args.appid, args.appid_file) or ["391540"]
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "fetch_reviews.jsonl"

    for appid in appids:
        output_path = RAW_DATA_DIR / f"{appid}_reviews.json"
        if output_path.exists() and not args.force:
            print(f"Skipping {appid}; {output_path.name} already exists.")
            write_log(log_path, appid, "skipped_existing")
            continue

        try:
            reviews = get_reviews(appid, max_reviews=args.max_reviews, delay=args.delay)
            if not reviews:
                raise RuntimeError("No reviews returned")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(reviews, f, indent=4)
            write_log(log_path, appid, "success", len(reviews))
            print(f"Saved {len(reviews)} reviews to {output_path}")
        except Exception as exc:
            write_log(log_path, appid, "failed", error=str(exc))
            print(f"Failed to fetch {appid}: {exc}")


if __name__ == "__main__":
    main()
