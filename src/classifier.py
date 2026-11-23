"""Multi-label BERT toxicity classifier (fine-tuned mBERT, or the public fallback model)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ARTIFACTS_DIR, LABELS

MAX_LEN = 256


class ToxicityClassifier:
    def __init__(self, model_id: str, token: str | None = None):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        try:  # transformers >= 4.56 uses `dtype`; older versions use `torch_dtype`
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token, dtype=torch.float32)
        except TypeError:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token, torch_dtype=torch.float32)
        self.model.eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))

        # Map model output index -> our canonical label order.
        id2label = {int(k): v.lower().replace(" ", "_") for k, v in self.model.config.id2label.items()}
        self.label_index = []
        for lab in LABELS:
            idx = next((i for i, name in id2label.items() if name == lab), None)
            if idx is None:
                raise ValueError(f"Model {model_id} has no '{lab}' output. Found: {id2label}")
            self.label_index.append(idx)

        self.thresholds = self._load_thresholds(model_id, token)
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ thresholds
    @staticmethod
    def _load_thresholds(model_id: str, token: str | None) -> dict[str, float]:
        candidates = [Path(model_id) / "thresholds.json", ARTIFACTS_DIR / "thresholds.json"]
        for c in candidates:
            if c.exists():
                return {k: float(v) for k, v in json.loads(c.read_text()).items()}
        if not Path(model_id).exists():
            try:
                from huggingface_hub import hf_hub_download

                p = hf_hub_download(model_id, "thresholds.json", token=token)
                return {k: float(v) for k, v in json.loads(Path(p).read_text()).items()}
            except Exception:
                pass
        return {lab: 0.5 for lab in LABELS}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return an (n, 6) array of sigmoid probabilities in canonical LABELS order."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else "." for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
            logits = self.model(**enc).logits
            probs = torch.sigmoid(logits)[:, self.label_index]
            out.append(probs.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(LABELS)))

    def subword_view(self, text: str) -> list[dict]:
        """BERT WordPiece tokenisation of the input (for the NLP-pipeline view)."""
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LEN)
        toks = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        return [{"position": i, "token": t, "id": int(tid)} for i, (t, tid) in enumerate(zip(toks, enc["input_ids"]))]

    def describe(self) -> dict:
        cfg = self.model.config
        return {
            "model_id": self.model_id,
            "architecture": cfg.architectures[0] if getattr(cfg, "architectures", None) else cfg.model_type,
            "model_type": cfg.model_type,
            "parameters": f"{self.num_parameters/1e6:.1f} M",
            "vocab_size": cfg.vocab_size,
            "hidden_size": getattr(cfg, "hidden_size", getattr(cfg, "dim", "-")),
            "layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", "-")),
            "max_length": MAX_LEN,
            "thresholds": self.thresholds,
        }
