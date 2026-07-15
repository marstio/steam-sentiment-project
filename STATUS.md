# Project status — 2026-07-15

## Current state

The raw collection and current indie-scope refresh are complete.

- 241 catalog games were requested.
- 238 produced usable review JSON files.
- 3 remain unresolved because Steam returned no reviews: `1568590`, `1920960`, and `700330`.
- The audit counted 215,893 raw reviews.
- Rocket League (`252950`) is excluded from the current indie-scope analysis but its raw data is preserved.
- The processed review table and EDA exports were regenerated after the scope exclusion.

## Classical NLP status

Notebook 03 was rerun and saved with execution counts and outputs. The model uses a game-held-out split and Steam's `voted_up` recommendation label.

Repeated evaluation across seeds `42`, `123`, `999`, `2024`, and `7` produced balanced accuracy from approximately `0.880` to `0.895`, with a mean of approximately `0.8866`.

Logistic regression with word TF-IDF is the selected classical baseline. This is binary recommendation-sentiment classification, not a human-annotated emotion model.

## Next phase

The optional Transformer comparison is prepared in `scripts/04_transformer_experiment.py`. It will use the same game-held-out approach and save accuracy, balanced accuracy, macro F1, and ROC-AUC to a metrics JSON file.

The optional packages are installed, but the first run encountered a CPU-only PyTorch wheel and a blocked pretrained-model download. Resolve the CUDA/model-download environment before running the full experiment.

The frontend directory is currently a placeholder. Dashboard implementation comes after the NLP comparison and final scope/label decisions.
