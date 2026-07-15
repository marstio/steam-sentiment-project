"""Optional Transformer experiment for Steam recommendation classification.

This is a contextual NLP comparison for the TF-IDF baselines. The target is
Steam's ``voted_up`` recommendation label, not a human-annotated emotion label.
Install ``requirements-transformer.txt`` before running this script.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/reviews_clean.csv"))
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="artifacts/distilbert-steam-sentiment")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data)
    frame = frame.loc[frame.review_clean.fillna("").str.strip().ne("")].copy()
    frame = frame.drop_duplicates("recommendationid")
    if args.max_rows and len(frame) > args.max_rows:
        # Avoid biasing a capped run toward the first games in the CSV.
        frame = frame.sample(args.max_rows, random_state=args.seed)
    frame = frame.reset_index(drop=True)

    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(
            frame, groups=frame.AppID
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    def make_dataset(part):
        ds = Dataset.from_pandas(
            pd.DataFrame(
                {
                    "text": frame.iloc[part].review_clean.tolist(),
                    "label": frame.iloc[part].voted_up.astype(int).tolist(),
                }
            )
        )
        return ds.map(tokenize, batched=True).remove_columns(["text"])

    def compute_metrics(eval_prediction):
        import numpy as np

        logits, labels = eval_prediction
        predictions = logits.argmax(axis=-1)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = probabilities[:, 1] / probabilities.sum(axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "roc_auc": roc_auc_score(labels, probabilities),
        }

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=make_dataset(train_idx),
        eval_dataset=make_dataset(test_idx),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    result = {
        "task": "binary recommendation classification",
        "label_definition": "Steam voted_up; not a human-annotated emotion label",
        "model": args.model,
        "seed": args.seed,
        "rows": len(frame),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_games": int(frame.iloc[train_idx].AppID.nunique()),
        "test_games": int(frame.iloc[test_idx].AppID.nunique()),
        "metrics": {
            key.removeprefix("eval_"): value
            for key, value in metrics.items()
            if key.startswith("eval_")
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
