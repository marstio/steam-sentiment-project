# Steam Sentiment Project Handoff

## Purpose

This project analyzes sentiment and review trends in indie Steam games. The planned workflow is:

1. Collect reviews from a curated list of indie Steam App IDs.
2. Preserve raw API responses in `data/raw/`.
3. Clean and tabularize reviews for exploratory analysis.
4. Train a TF-IDF + XGBoost baseline.
5. Compare it with a lightweight Transformer model.
6. Publish finalized findings through a future dashboard.

## Current state

- The collector is in `scripts/fetch_reviews.py`.
- The Steam public `appreviews` endpoint does not require a Steam API key.
- Undertale (`391540`) was the initial test game.
- A successful test produced 1,000 English reviews in `data/raw/391540_reviews.json`.
- Two of those records had empty text; they remain in raw data but should be excluded from NLP training.
- The current collector accepts multiple App IDs and skips completed output files by default.
- Fetch activity is logged in `data/logs/fetch_reviews.jsonl`.
- The current collection contains 238 `<AppID>_reviews.json` files for the 241-game input list.
- The latest log state is 237 successful games, 3 games returning no reviews, and 2 previously completed games skipped on rerun.
- The collection audit was run against `data/raw/steam_indie_games_list.csv`: 241 requested games, 238 available games, 3 unresolved games, and 215,893 raw reviews.
- Keep these numbered JSON files in `data/raw/`; the AppID filename is how the EDA notebook identifies and joins each game.
- `data/game_scope_exclusions.csv` currently excludes Rocket League (`252950`) because it was acquired/published under Epic Games and no longer fits the current indie-catalog definition. Its raw data is preserved for traceability.

## Collection commands

Test or refetch Undertale:

```powershell
python scripts/fetch_reviews.py --appid 391540 --force
```

Fetch several games:

```powershell
python scripts/fetch_reviews.py --appid 391540 --appid 123456 --max-reviews 1000
```

Fetch the full target list directly from the current CSV. The CSV must contain an `AppID` column:

```powershell
python scripts/fetch_reviews.py --appid-file data/raw/steam_indie_games_list.csv --max-reviews 1000 --delay 1
```

If the process stops, run the same command again. Existing JSON files are skipped, while missing or failed games are retried. The three no-review responses should be documented rather than treated as successful datasets.

## Rate-limit policy

The official Steamworks documentation allows up to 100 reviews per page and recommends using `recent` or `updated` while paging until an empty response. This project requests at most 100 reviews per page and waits one second between pages.

There is no published guarantee that this exact delay prevents throttling. Treat HTTP 429, timeouts, and 5xx responses as expected operational failures: stop or slow down, inspect the log, and retry later. Do not run many parallel workers. A 725-game collection with 1,000 reviews per game is potentially thousands of requests, so collection should be staged and monitored.

Official reference: <https://partner.steamgames.com/doc/store/getreviews>

## Data decisions

- Keep raw JSON unchanged as the source of truth.
- Create a processed table containing one row per review.
- Drop rows where `review.strip()` is empty for NLP.
- Keep `voted_up` as the initial supervised sentiment label, while documenting that it is a recommendation signal rather than a perfect emotional sentiment label.
- Preserve `appid` and game title in processed data so all games can be compared.
- Record the collection date because Steam reviews and rankings change over time.
- The current `filter=recent` setting samples recent reviews. It should not be described as a complete historical review archive.
- The target is Steam's `voted_up` recommendation label, not a fully annotated emotion taxonomy. Treat the model as binary recommendation-sentiment classification until a separate emotion-labeling plan is added.

## Current model status

The latest saved comparison used a game-held-out split (190 training games and 48 test games) before the scope-exclusion refresh:

- Logistic regression + word TF-IDF: balanced accuracy about 0.88, macro F1 about 0.80, ROC-AUC about 0.95.
- Linear SVM + word TF-IDF: balanced accuracy about 0.85, macro F1 about 0.81, ROC-AUC about 0.94.
- Majority baseline: balanced accuracy 0.50.

The model is not proven robust from one split. Repeat game-held-out evaluation across several seeds and inspect per-game results before selecting a final classical model. XGBoost is optional as a comparison baseline; it is not currently the selected model and should remain disabled until the memory-safe notebook workflow is confirmed.

The model notebook has now been rerun and saved with execution counts and outputs. Repeated game-held-out evaluation across seeds `42`, `123`, `999`, `2024`, and `7` produced balanced accuracy values ranging from approximately `0.880` to `0.895`, with a mean of approximately `0.8866`. Logistic regression with word TF-IDF remains the selected classical baseline because it leads on balanced accuracy and ROC-AUC. Linear SVM has slightly higher macro F1, so the final report should state the primary metric explicitly.

The target remains Steam's `voted_up` recommendation label. This is recommendation-sentiment classification, not a human-annotated emotion task. No emotion taxonomy such as joy, anger, or frustration has been trained yet.

## Scope and unresolved collection status

"Unresolved" means the collector did not produce a usable review JSON payload for an input AppID. It does not prove that the game has no Steam reviews. The current unresolved IDs are `1568590`, `1920960`, and `700330`; retry them only during an intentional collection pass.

The indie scope policy is maintained in `data/game_scope_exclusions.csv`. Add borderline acquired/corporate titles there rather than deleting raw files. For future scope changes, rerun audit, EDA, and modeling and record the date.

## Notebook/editor and kernel notes

The `.ipynb` file on disk is the source of truth. VS Code can keep an open notebook as a dirty in-memory document; external edits then do not appear in that tab, and closing with "Don't Save" discards the tab's stale state before reopening the disk version. After an agent edits a notebook, close/reopen the tab or use the notebook/editor "Revert/Discard Changes" action, and do not save the stale tab over the updated file.

The model notebook can exhaust RAM because TF-IDF matrices and fitted pipelines are large. Use the memory-safe defaults (`MAX_FEATURES=50_000`, one model at a time, delete fitted pipelines after each comparison), restart the kernel before a fresh full run, and save result CSVs after the comparison cell. If the kernel crashes, run the fast baselines before enabling XGBoost.

The optional Transformer script is prepared in `scripts/04_transformer_experiment.py`. It uses deterministic sampling, a game-held-out split, and saves accuracy, balanced accuracy, macro F1, and ROC-AUC to `metrics.json`. The optional packages installed successfully in the working environment, but the initial run used a CPU-only PyTorch wheel and the pretrained model download was blocked. Install a CUDA-enabled PyTorch build if GPU execution is desired, then allow the model download before running the experiment.

The `frontend/` directory is currently only a placeholder. Dashboard work should wait until the classical/Transformer comparison and the label limitations are finalized.

## Next implementation steps

1. Keep the three unresolved IDs documented unless an intentional retry produces reviews.
2. Optionally run `scripts/04_transformer_experiment.py` after resolving the model-download and CUDA/CPU environment choice.
3. Compare Transformer and TF-IDF results using the same game-held-out metrics and document whether contextual modeling materially improves the baseline.
4. Decide whether to add a separate emotion-labeling or emotion-lexicon phase; do not describe `voted_up` as a complete emotion label.
5. Build the dashboard only after the analysis outputs and scope/label decisions are stable.

There is no original `01` notebook; the collection step was previously represented only by `scripts/fetch_reviews.py`. The new audit script makes that step explicit without refetching or deleting raw files.

## Handoff checklist for another agent

- Read this file first.
- Inspect `scripts/fetch_reviews.py` and `data/logs/fetch_reviews.jsonl`.
- Do not delete raw files.
- Do not refetch existing games unless asked or `--force` is explicitly used.
- Check for empty review text before modeling.
- Report request failures and incomplete App IDs rather than silently treating them as complete.
