"""Load the SentiMix (SemEval-2020 Task 9) Hinglish CoNLL files into DataFrames."""
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def clean_text(text: str) -> str:
    # The CoNLL files are token-joined ("@ name", "https // t . co / xyz"), so undo that first.
    text = re.sub(r"(…\s*)?https?\s*//\s*t\s*\.?\s*co\s*/?\s*\S*", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\s*\w+(\s*__\s*\w+|\s*_\s*\w+)*", " @user ", text)
    text = text.replace("…", " ")
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"(@user\s*)+", "@user ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_conll(path: Path) -> pd.DataFrame:
    rows, tokens, langs, uid, label = [], [], [], None, None

    def flush():
        if uid is not None and tokens:
            rows.append(
                {
                    "uid": uid,
                    "text": clean_text(" ".join(tokens)),
                    "label": label,
                    "langs": langs[:],
                }
            )

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip("\n")
        if line.startswith("meta\t"):
            flush()
            _, uid, label = line.split("\t")
            tokens, langs = [], []
        elif line.strip():
            parts = line.split("\t")
            tokens.append(parts[0])
            langs.append(parts[1] if len(parts) > 1 else "O")
    flush()
    df = pd.DataFrame(rows)
    df["y"] = df["label"].map(LABEL2ID)
    return df


def load_splits():
    train = read_conll(DATA_DIR / "train_14k_split_conll.txt")
    dev = read_conll(DATA_DIR / "dev_3k_split_conll.txt")
    return train, dev


if __name__ == "__main__":
    tr, dv = load_splits()
    print(tr.shape, dv.shape)
    print(tr["label"].value_counts(normalize=True).round(3))
    print(tr.head(3).to_string())
