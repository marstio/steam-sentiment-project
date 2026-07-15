# Steam Sentiment Project

This project analyzes recent Steam reviews from a curated indie-game catalog.
It combines collection, cleaning, exploratory analysis, and game-held-out NLP
classification.

## Current progress

- 241 games are in the input catalog.
- 238 games produced usable raw review files.
- 3 games remain unresolved because Steam returned no reviews:
  `1568590`, `1920960`, and `700330`.
- The collection audit produced 215,893 reviews in total.
- Rocket League (`252950`) is retained as raw data but excluded from the
  current indie-scope analysis because Psyonix is now part of Epic Games.
- Empty review text is excluded from NLP training.
- EDA outputs and the processed review table have been regenerated after the
  scope exclusion.
- The model notebook uses a game-held-out split and has been evaluated across
  five seeds: `42`, `123`, `999`, `2024`, and `7`.
- Logistic regression with word TF-IDF is the selected classical baseline.
  Mean repeated balanced accuracy is approximately `0.8866`.

## Important label definition

The target is Steam's `voted_up` recommendation label. This is a binary
recommendation-sentiment task, not a human-annotated emotion taxonomy. The
project has not yet classified emotions such as joy, anger, or frustration.

## Project layout

| Path | Purpose |
|---|---|
| `data/raw/` | Unmodified Steam API responses |
| `data/processed/` | Clean review table, EDA exports, and model results |
| `scripts/fetch_reviews.py` | Rate-limited review collection |
| `scripts/01_collection_audit.py` | Requested-vs-collected audit |
| `scripts/02_cleaning_and_eda.py` | Script equivalent of the EDA notebook |
| `scripts/03_model_training.py` | Script equivalent of the classical modeling notebook |
| `scripts/04_transformer_experiment.py` | Optional DistilBERT comparison |
| `notebooks/02_cleaning_and_eda.ipynb` | Cleaning and exploratory analysis |
| `notebooks/03_model_training.ipynb` | Classical NLP modeling and repeated evaluation |

There is no `01` notebook because collection is deliberately implemented as
scripts rather than an interactive notebook. The audit script is the explicit
step-01 record without refetching or deleting raw data.

## Reproducibility commands

From the project root:

```powershell
python scripts/01_collection_audit.py --input data/raw/steam_indie_games_list.csv
python scripts/02_cleaning_and_eda.py
python scripts/03_model_training.py
```

The Transformer experiment is optional. Its dependencies are listed in
`requirements-transformer.txt`; it also requires downloading the selected
pretrained model. The current script saves evaluation metrics to
`artifacts/distilbert-steam-sentiment/metrics.json`.

## Next phase

Run the Transformer comparison when the optional dependencies and model
download are available. Compare it against the saved TF-IDF baseline using
the same game-held-out evaluation logic. A dashboard is a later presentation
phase, after the NLP results and label limitations are finalized.
