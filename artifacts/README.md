# artifacts/

Files produced by `notebooks/ToxicGuardAI_Training.ipynb` (download `artifacts.zip` from Colab and unzip here):

| File | Used for |
|---|---|
| `metrics.json` | Model Performance tab (test-set metrics, baseline, sentiment / IE / LLM evaluation) |
| `thresholds.json` | Per-category decision thresholds tuned on the validation set |
| `tfidf.joblib` | Corpus TF-IDF vectoriser (CO1 vectorisation view) |
| `topic_model.joblib` | LDA topic model (CO3 topic view) |

The app works without these files (it falls back gracefully). The fine-tuned model itself lives on the
Hugging Face Hub (set `MODEL_ID` in secrets), or optionally in `artifacts/model/` (git-ignored).
