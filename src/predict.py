"""Inference + word-level explanations (leave-one-word-out occlusion) for both models."""
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

from data import LABELS, clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class BaselineModel:
    name = "TF-IDF + Logistic Regression"

    def __init__(self):
        bundle = joblib.load(MODEL_DIR / "baseline.joblib")
        self.word, self.char, self.clf = bundle["word"], bundle["char"], bundle["clf"]

    def proba(self, texts):
        X = hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()
        return self.clf.predict_proba(X)


class BertModel:
    name = "MiniLM (fine-tuned Transformer)"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        path = MODEL_DIR / "transformer"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def proba(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self.tok(texts[i : i + 32], truncation=True, max_length=64, padding=True, return_tensors="pt")
                out.append(self.model(**enc).logits.softmax(-1).numpy())
        return np.vstack(out)


def available_models():
    models = {}
    if (MODEL_DIR / "baseline.joblib").exists():
        models[BaselineModel.name] = BaselineModel
    if (MODEL_DIR / "transformer" / "config.json").exists():
        models[BertModel.name] = BertModel
    return models


def predict(model, text: str):
    text = clean_text(text)
    probs = model.proba([text])[0]
    return {l: float(p) for l, p in zip(LABELS, probs)}, text


def explain(model, text: str):
    """Score each word by how much removing it changes the predicted class probability.

    Positive score = the word pushes towards the predicted label, negative = pushes against it.
    """
    text = clean_text(text)
    words = text.split()
    if not words:
        return [], None
    variants = [text] + [" ".join(words[:i] + words[i + 1 :]) or "." for i in range(len(words))]
    probs = model.proba(variants)
    label_id = int(probs[0].argmax())
    scores = probs[0][label_id] - probs[1:, label_id]
    return list(zip(words, scores.tolist())), LABELS[label_id]
