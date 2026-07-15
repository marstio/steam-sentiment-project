"""Build the cleaned review table and comprehensive, dashboard-ready EDA exports."""

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SCOPE_EXCLUSIONS_PATH = ROOT / "data" / "game_scope_exclusions.csv"
SLANG = {"gg": "good game", "ez": "easy", "pog": "exciting", "goated": "great", "op": "overpowered", "nerf": "reduce power", "buff": "increase power", "imo": "in my opinion", "idk": "i do not know"}
ASCII_ART_LINE = re.compile(r"^[^A-Za-z0-9]*$")
HTML_TAG = re.compile(r"<[^>]+>")


def clean_review(value):
    text = html.unescape(str(value or ""))
    text = " ".join(line for line in text.splitlines() if not ASCII_ART_LINE.fullmatch(line.strip()))
    text = HTML_TAG.sub(" ", text).lower()
    for slang, replacement in SLANG.items():
        text = re.sub(rf"\b{re.escape(slang)}\b", replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def load_reviews(raw_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(raw_dir.glob("*_reviews.json")):
        appid = path.name.removesuffix("_reviews.json")
        for review in json.loads(path.read_text(encoding="utf-8")):
            author = review.get("author") or {}
            records.append({"AppID": appid, "author_name": author.get("personaname", ""), **{k: v for k, v in review.items() if k != "author"}, "playtime_forever": author.get("playtime_forever", 0), "playtime_at_review": author.get("playtime_at_review", 0), "num_reviews_author": author.get("num_reviews", 0)})
    return pd.DataFrame(records)


def build(input_dir: Path, metadata_path: Path, output_dir: Path):
    reviews = load_reviews(input_dir)
    metadata = pd.read_csv(metadata_path, dtype={"AppID": "string"})
    if SCOPE_EXCLUSIONS_PATH.exists():
        excluded = pd.read_csv(SCOPE_EXCLUSIONS_PATH, dtype={"AppID": "string"})
        excluded_ids = set(excluded.loc[excluded["scope_status"].eq("excluded"), "AppID"].astype(str))
        reviews = reviews.loc[~reviews["AppID"].astype(str).isin(excluded_ids)].copy()
        print(f"Excluded from indie scope: {sorted(excluded_ids)}")
    reviews["AppID"] = reviews["AppID"].astype("string")
    reviews = reviews.merge(metadata, on="AppID", how="left", suffixes=("", "_game"))
    reviews["review_clean"] = reviews["review"].map(clean_review)
    reviews["is_empty_review"] = reviews["review_clean"].eq("")
    reviews["sentiment"] = reviews["voted_up"].map({True: "Positive", False: "Negative"})
    reviews["sentiment_label"] = reviews["voted_up"].astype("Int64")
    reviews["review_word_count"] = reviews["review_clean"].str.split().str.len().fillna(0).astype(int)
    reviews["review_char_count"] = reviews["review_clean"].str.len().fillna(0).astype(int)
    reviews["playtime_hours"] = pd.to_numeric(reviews["playtime_at_review"], errors="coerce").fillna(0) / 60
    reviews["review_date"] = pd.to_datetime(reviews["timestamp_created"], unit="s", errors="coerce", utc=True)
    reviews["review_year"] = reviews["review_date"].dt.year.astype("Int64")
    reviews["review_month"] = reviews["review_date"].dt.to_period("M").astype("string")
    reviews["helpful_votes"] = pd.to_numeric(reviews["votes_up"], errors="coerce").fillna(0).astype(int)
    reviews["engagement_score"] = reviews["helpful_votes"] + pd.to_numeric(reviews["votes_funny"], errors="coerce").fillna(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(output_dir / "reviews_clean.csv", index=False, encoding="utf-8-sig")

    usable = reviews[~reviews["is_empty_review"]].copy()
    game = usable.groupby(["AppID", "Name"], dropna=False).agg(review_count=("AppID", "size"), positive_rate=("voted_up", "mean"), median_review_words=("review_word_count", "median"), median_playtime_hours=("playtime_hours", "median"), median_helpful_votes=("helpful_votes", "median"), first_review_date=("review_date", "min"), last_review_date=("review_date", "max")).reset_index()
    monthly = usable.assign(review_month=usable["review_date"].dt.to_period("M").astype("string")).groupby(["review_month", "voted_up"], dropna=False).size().rename("review_count").reset_index()
    length = usable.groupby("sentiment", dropna=False)["review_word_count"].describe().reset_index()
    sentiment = usable.groupby("sentiment", dropna=False).size().rename("review_count").reset_index()
    game.to_csv(output_dir / "eda_game_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "eda_monthly_sentiment.csv", index=False, encoding="utf-8-sig")
    length.to_csv(output_dir / "eda_review_length_summary.csv", index=False, encoding="utf-8-sig")
    sentiment.to_csv(output_dir / "eda_sentiment_summary.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    sns.countplot(data=usable, x="sentiment", ax=axes[0, 0])
    axes[0, 0].set_title("Recommendation label balance")
    sns.histplot(data=usable, x="review_word_count", hue="sentiment", bins=60, element="step", ax=axes[0, 1])
    axes[0, 1].set_xlim(0, usable["review_word_count"].quantile(.99))
    axes[0, 1].set_title("Review length (99th percentile)")
    sns.scatterplot(data=game, x="review_count", y="positive_rate", size="median_playtime_hours", sizes=(20, 250), alpha=.7, ax=axes[0, 2])
    axes[0, 2].set_xscale("log"); axes[0, 2].set_title("Game volume vs positive rate")
    top = game.nlargest(15, "review_count").sort_values("review_count")
    axes[1, 0].barh(top["Name"].fillna(top["AppID"]), top["review_count"])
    axes[1, 0].set_title("Top games by collected reviews")
    sns.boxplot(data=usable, x="sentiment", y="playtime_hours", showfliers=False, ax=axes[1, 1])
    axes[1, 1].set_ylim(0, usable["playtime_hours"].quantile(.99)); axes[1, 1].set_title("Playtime at review")
    monthly_pivot = monthly.pivot(index="review_month", columns="voted_up", values="review_count").fillna(0)
    monthly_pivot.plot(ax=axes[1, 2], title="Reviews by creation month", legend=True)
    axes[1, 2].tick_params(axis="x", rotation=45)
    plt.tight_layout(); fig.savefig(output_dir / "eda_overview.png", dpi=150); plt.close(fig)
    print(f"Saved {len(reviews):,} rows ({len(usable):,} non-empty) and EDA exports to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "processed" / "steam_games_enriched.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args(); build(args.raw_dir, args.metadata, args.output_dir)
