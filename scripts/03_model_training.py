"""Train reproducible game-held-out text baselines and save comparable metrics."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SCOPE_EXCLUSIONS_PATH = ROOT / "data" / "game_scope_exclusions.csv"


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    scores = model.decision_function(x_test) if hasattr(model, "decision_function") else model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
    }


def train(data_path: Path, output_path: Path):
    data = pd.read_csv(data_path)
    if SCOPE_EXCLUSIONS_PATH.exists():
        excluded = pd.read_csv(SCOPE_EXCLUSIONS_PATH, dtype={"AppID": "string"})
        excluded_ids = set(excluded.loc[excluded["scope_status"].eq("excluded"), "AppID"].astype(str))
        data = data.loc[~data["AppID"].astype(str).isin(excluded_ids)].copy()
        print(f"Excluded from indie scope: {sorted(excluded_ids)}")
    data = data.loc[data["review_clean"].fillna("").str.strip().ne("")].drop_duplicates("recommendationid")
    x, y = data["review_clean"], data["voted_up"].astype(int)
    if data["AppID"].nunique() >= 5:
        split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        train_idx, test_idx = next(split.split(x, y, groups=data["AppID"]))
        split_type = "game_held_out"
        x_train, x_test, y_train, y_test = x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
        split_type = "stratified_review_split"

    vectorizer = lambda analyzer="word": TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2) if analyzer == "word" else (3, 5), min_df=2, max_features=200_000, sublinear_tf=True)
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_word_tfidf": Pipeline([("tfidf", vectorizer()), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1))]),
        "linear_svm_word_tfidf": Pipeline([("tfidf", vectorizer()), ("classifier", LinearSVC(class_weight="balanced"))]),
        "logistic_char_tfidf": Pipeline([("tfidf", vectorizer("char")), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1))]),
    }
    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        results.append({"model": name, **evaluate(model, x_test, y_test)})
        print(results[-1])
    result = {"split_type": split_type, "train_rows": len(x_train), "test_rows": len(x_test), "train_games": int(data.iloc[train_idx]["AppID"].nunique()) if "train_idx" in locals() else int(data.iloc[:len(x_train)]["AppID"].nunique()), "test_games": int(data.iloc[test_idx]["AppID"].nunique()) if "test_idx" in locals() else int(data.iloc[len(x_train):]["AppID"].nunique()), "models": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False).to_csv(output_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "reviews_clean.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "model_results.json")
    args = parser.parse_args(); train(args.data, args.output)
