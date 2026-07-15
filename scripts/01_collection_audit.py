"""Audit the requested Steam App IDs against raw review files and fetch logs."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_requested_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"AppID": "string"})
        return frame["AppID"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    return list(dict.fromkeys(
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ))


def audit(input_path: Path, raw_dir: Path, log_path: Path, output_dir: Path) -> pd.DataFrame:
    requested = read_requested_ids(input_path)
    rows = []
    for appid in requested:
        raw_path = raw_dir / f"{appid}_reviews.json"
        count = 0
        valid_json = False
        error = ""
        if raw_path.exists():
            try:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                count = len(payload) if isinstance(payload, list) else 0
                valid_json = isinstance(payload, list)
            except (OSError, json.JSONDecodeError) as exc:
                error = str(exc)
        rows.append({
            "AppID": appid,
            "raw_file_exists": raw_path.exists(),
            "valid_json": valid_json,
            "review_count": count,
            "collection_status": "available" if count else ("invalid_json" if error else "missing_or_no_reviews"),
            "error": error,
        })

    audit_frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(output_dir / "collection_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_games": len(requested),
        "available_games": int((audit_frame.review_count > 0).sum()),
        "missing_or_no_review_games": int((audit_frame.review_count == 0).sum()),
        "total_reviews": int(audit_frame.review_count.sum()),
        "missing_or_no_review_appids": audit_frame.loc[audit_frame.review_count == 0, "AppID"].tolist(),
    }
    (output_dir / "collection_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if log_path.exists():
        statuses = Counter(json.loads(line).get("status") for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
        print("log status counts:", dict(statuses))
    return audit_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "appids.txt")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--log", type=Path, default=ROOT / "data" / "logs" / "fetch_reviews.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    audit(args.input, args.raw_dir, args.log, args.output_dir)
