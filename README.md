# 🛡️ ToxicGuardAI

**Explainable multilingual toxic comment detection using BERT + XAI + Generative AI**
English · हिन्दी (Hindi) · Hinglish (code-mixed) — NLP course project

ToxicGuardAI takes a comment and:

1. **Detects** whether it is toxic with a fine-tuned multilingual BERT,
2. **Categorises** it into 6 classes: toxic, severe toxic, obscene, threat, insult, identity hate,
3. **Explains** the decision with Integrated Gradients, Occlusion and LIME, checks the explanation's faithfulness with ERASER metrics, and adds an LLM-written explanation,
4. Adds sentiment, information extraction, topic modelling, language ID, translation and a detoxified rewrite.

---

## 📋 Rubric mapping (Project Demonstration, 30 marks)

| # | Criterion | CO | Marks | Implementation | Where to show it |
|---|---|---|---|---|---|
| 1 | Text extraction, preprocessing, cleaning & vectorisation | CO1 | 7 | Reads CSV/XLSX/TXT/PDF/JSON · 8-step cleaning with an audit trail · Hinglish spelling normalisation · EN+HI stop-words · TF-IDF (sparse) · BERT WordPiece (dense) | App: **NLP Pipeline** and **Batch** tabs · Notebook §1–3 |
| 2 | Sentiment analysis & information extraction, implemented and evaluated | CO2 | 8 | Fine-tuned mBERT multi-label classifier · VADER sentiment · spaCy NER, identity gazetteer, target detection and regex IE · P/R/F1/ROC-AUC, confusion matrix, TF-IDF baseline | App: **Sentiment & IE** and **Model Performance** tabs · Notebook §4–7 |
| 3 | Topic modelling / multilingual NLP with cloud AI services | CO3 | 7 | EN/HI/Hinglish language ID · Google Cloud Translation · translation-assisted ensemble · LDA (corpus model and per-batch) | App: **Multilingual & Topics** and **Batch** tabs · Notebook §8–9 |
| 4 | GenAI/LLM application, prompt engineering, integration & demo | CO4 | 8 | Google Gemini (or Claude) with structured JSON output · zero-shot, few-shot and XAI-grounded prompts · detoxified rewrite re-scored by BERT · XAI↔LLM agreement score | App: **GenAI Insights** tab · Notebook §10 |

---

## 🗂️ Project structure

```
ToxicGuardAi/
├── app.py                          # Streamlit app (UI)
├── requirements.txt                # Python dependencies (CPU torch)
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                 # theme + server settings
│   └── secrets.toml.example        # template -> copy to secrets.toml (never commit it)
├── notebooks/
│   └── ToxicGuardAI_Training.ipynb # Colab: data → preprocessing → mBERT fine-tuning → evaluation → LDA → LLM → export
├── artifacts/                      # outputs of the notebook (metrics, thresholds, TF-IDF, LDA)
│   └── README.md
├── src/
│   ├── config.py                   # labels, model IDs, secrets
│   ├── preprocessing.py            # CO1 extraction, cleaning, tokenisation, TF-IDF
│   ├── multilingual.py             # CO3 language ID + Google translation
│   ├── classifier.py               # BERT multi-label classifier wrapper
│   ├── explainer.py                # XAI: Integrated Gradients, Occlusion, LIME, faithfulness
│   ├── analysis.py                 # CO2 sentiment + IE, CO3 LDA topics
│   ├── llm.py                      # CO4 Gemini/Claude prompts, structured output, offline fallback
│   ├── pipeline.py                 # end-to-end orchestration
│   ├── lexicon.py                  # Hinglish markers, abuse lexicon, identity gazetteer
│   ├── examples.py                 # demo comments + live benchmark
│   └── ui.py                       # CSS, cards, heat-map, plotly charts
└── tests/
    └── test_core.py                # pytest unit tests
```

---

## 🚀 Step-by-step: train → version → deploy

### Step 1 — Train the model in Google Colab (≈40 min)

1. Go to <https://colab.research.google.com> → **Upload** → `notebooks/ToxicGuardAI_Training.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Create a Hugging Face account and a **write** token at <https://huggingface.co/settings/tokens>.
4. In Colab, click the 🔑 **Secrets** icon and add:
   - `HF_TOKEN`: your write token (enable notebook access)
   - `ANTHROPIC_API_KEY`: optional, needed for the §10 LLM evaluation (<https://console.anthropic.com>)
5. In the config cell, set `HF_REPO = "<your-hf-username>/toxicguard-mbert"`.
6. **Runtime → Run all**. At the end the notebook:
   - pushes the fine-tuned model to `https://huggingface.co/<your-hf-username>/toxicguard-mbert`
   - downloads `artifacts.zip`
7. Unzip `artifacts.zip` into the project's `artifacts/` folder. It should contain `metrics.json`, `thresholds.json`, `tfidf.joblib` and `topic_model.joblib`.

> The app also runs **before** training: it falls back to the public `unitary/toxic-bert` model and uses translation for Hindi and Hinglish.

### Step 2 — Run locally

```bash
cd ToxicGuardAi
python3.12 -m venv .venv && source .venv/bin/activate      # Python 3.11 or 3.12
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml # then edit MODEL_ID / GEMINI_API_KEY
streamlit run app.py                                       # opens http://localhost:8501
pytest -q                                                  # optional: unit tests
```

The first run downloads the BERT weights (~700 MB for mBERT), which can take a few minutes.

### Step 3 — Version control with Git and GitHub

```bash
cd ToxicGuardAi
git init
git add .
git commit -m "ToxicGuardAI: initial project"
# Create an empty repo named ToxicGuardAI on https://github.com/new (no README), then:
git branch -M main
git remote add origin https://github.com/<your-github-username>/ToxicGuardAI.git
git push -u origin main
```

Later changes, for example after adding trained artifacts:

```bash
git add artifacts/ && git commit -m "Add trained artifacts and metrics" && git push
```

`.gitignore` keeps `.venv/`, model weights and `.streamlit/secrets.toml` out of the repo.
**Never commit API keys.**

### Step 4 — Deploy on Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**:
   - Repository: `<your-github-username>/ToxicGuardAI`
   - Branch: `main`
   - Main file path: `app.py`
   - **Advanced settings → Python version: 3.12**
   - **Advanced settings → Secrets**: paste
     ```toml
     MODEL_ID = "<your-hf-username>/toxicguard-mbert"
     GEMINI_API_KEY = "AIza..."
     ```
3. Click **Deploy**. The first build takes about 5–10 minutes (it installs torch and downloads the model). After that, every `git push` redeploys automatically.
4. You get a public URL such as `https://toxicguardai.streamlit.app`.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| "Could not load model" | Check that `MODEL_ID` matches your Hub repo and the repo is public (or add `HF_TOKEN`) |
| App restarts / out of memory | Turn LIME off; mBERT needs ~1.5 GB RAM, which fits the Streamlit Cloud limit |
| "Offline template mode" in the GenAI tab | Add `GEMINI_API_KEY` (free at aistudio.google.com/apikey) to the secrets |
| Translation shows "unavailable" | Temporary rate limit; optionally add `GOOGLE_TRANSLATE_API_KEY` (official Cloud Translation API) |
| `InconsistentVersionWarning` loading joblib files | Keep `scikit-learn==1.9.1` in both the notebook and `requirements.txt` |

### Step 5 — Generate the project poster (16:9, for the Smart Board)

1. Edit the team and institute details at the top of [poster/make_poster.py](poster/make_poster.py).
2. With `artifacts/metrics.json` in place and the virtual environment active:
   ```bash
   python poster/make_poster.py          # add --no-xai to skip the live explainability example
   ```
3. Open `poster/ToxicGuardAI_Poster.html` in Chrome or Edge → **Ctrl + P** → *Save as PDF*, Margins **None**, **Background graphics** on.

Every number on the poster is read from your real `metrics.json`. The explainability example is computed live by your trained model.

---

## 🎤 Suggested demo flow (about 8 minutes)

1. **Analyze a Comment:** click **🎲 English**, then **Analyze comment**. Walk through the verdict banner, the four summary cards and **Summary** (reasons, word highlights, category scores against their thresholds).
2. **Why? (Explainability):** the word heat-map, the most influential words, and "Can we trust this explanation?" (the score change when the top words are removed).
3. Click **🎲 Hinglish** and analyse. In **Language insights**, show the language detection, the Google translation, and how the original and translated scores are combined (CO3). Open **NLP pipeline details** for cleaning, TF-IDF and BERT tokens (CO1).
4. **AI explanation:** pick a prompt strategy, click **Generate explanation**, and show the explanation, the slang meaning, the polite rewrite re-checked by BERT, and the exact prompts (CO4).
5. **Batch Analysis:** upload a CSV of comments to show category counts, toxicity by language and the topics found by LDA.
6. **Model Performance:** headline metrics, BERT vs the TF-IDF baseline, per-language results, then **Run live check**.
7. **Project & Rubric:** the rubric mapping for the examiners.

---

## 🧠 Method summary

- **Data:** Jigsaw Toxic Comment Classification (160k English comments, 6 labels) and TextDetox multilingual toxicity (`en`, `hi`, `hin` = Hinglish splits).
- **Model:** `bert-base-multilingual-cased` with 6 sigmoid heads. It is trained with a **masked BCE loss**: datasets that only have a binary label supervise only the `toxic` head. `pos_weight` handles class imbalance, and per-label thresholds are tuned on the validation set.
- **XAI:**
  - Integrated Gradients (24–32-step Riemann sum from a `[PAD]` baseline, sub-words summed into words)
  - Occlusion (leave-one-word-out)
  - LIME (positional, Devanagari-safe)

  The three are combined into a max-normalised consensus. ERASER comprehensiveness and sufficiency measure how faithful the explanation is.
- **GenAI:** Google Gemini (`gemini-3.8-flash`, set `GEMINI_API_KEY`) or Anthropic Claude (`claude-opus-5-5`, set `ANTHROPIC_API_KEY`) with JSON-schema structured output. The XAI-grounded prompt gives the model the classifier's probabilities and attributions as evidence, and asks it to agree or disagree.
