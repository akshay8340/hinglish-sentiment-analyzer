"""Evaluate both saved models on the official SentiMix test set (labels live in a separate file)."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import DATA_DIR, LABEL2ID, LABELS, clean_text
from predict import BaselineModel, BertModel

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def fix_mojibake(s: str) -> str:
    """The test file stores emojis double-encoded (e.g. 'ðŸ˜…'); undo that when possible."""
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def read_test() -> pd.DataFrame:
    rows, tokens, uid = [], [], None
    for line in (DATA_DIR / "Hindi_test_unalbelled_conll_updated.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("meta\t"):
            if uid is not None and tokens:
                rows.append((uid, clean_text(" ".join(tokens))))
            uid, tokens = line.split("\t")[1], []
        elif line.strip():
            tokens.append(fix_mojibake(line.split("\t")[0]))
    if uid is not None and tokens:
        rows.append((uid, clean_text(" ".join(tokens))))
    df = pd.DataFrame(rows, columns=["uid", "text"])
    labels = pd.read_csv(DATA_DIR / "test_labels_hinglish.txt")
    labels["uid"] = labels["Uid"].astype(str)
    df = df.merge(labels[["uid", "Sentiment"]], on="uid")
    df["y"] = df["Sentiment"].map(LABEL2ID)
    return df


def main():
    test = read_test()
    print(f"test samples: {len(test)}  distribution: {test.Sentiment.value_counts().to_dict()}")
    out = {"n": len(test), "dist": test.Sentiment.value_counts().to_dict()}
    for key, cls in [("baseline", BaselineModel), ("transformer", BertModel)]:
        pred = cls().proba(test.text.tolist()).argmax(1)
        out[key] = {
            "accuracy": accuracy_score(test.y, pred),
            "macro_f1": f1_score(test.y, pred, average="macro"),
            "confusion_matrix": confusion_matrix(test.y, pred).tolist(),
            "report": classification_report(test.y, pred, target_names=LABELS, output_dict=True),
        }
        print(key, round(out[key]["accuracy"], 4), round(out[key]["macro_f1"], 4))
    (MODEL_DIR / "test_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
