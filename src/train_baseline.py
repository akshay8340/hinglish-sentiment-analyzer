"""Baseline: TF-IDF (word + char n-grams) + Logistic Regression."""
import json
from pathlib import Path

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data import LABELS, load_splits

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train, dev = load_splits()

    word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    # char n-grams absorb Hinglish spelling variation (kya/kyaa/kia, bekar/bekaar)
    char_vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)

    X_train = hstack([word_vec.fit_transform(train.text), char_vec.fit_transform(train.text)]).tocsr()
    X_dev = hstack([word_vec.transform(dev.text), char_vec.transform(dev.text)]).tocsr()

    clf = LogisticRegression(C=4.0, max_iter=2000)
    clf.fit(X_train, train.y)
    pred = clf.predict(X_dev)

    acc = accuracy_score(dev.y, pred)
    f1 = f1_score(dev.y, pred, average="macro")
    print(classification_report(dev.y, pred, target_names=LABELS, digits=3))
    print(f"accuracy={acc:.4f}  macro-F1={f1:.4f}")

    joblib.dump({"word": word_vec, "char": char_vec, "clf": clf}, MODEL_DIR / "baseline.joblib")
    (MODEL_DIR / "baseline_metrics.json").write_text(json.dumps({"accuracy": acc, "macro_f1": f1}, indent=2))


if __name__ == "__main__":
    main()
