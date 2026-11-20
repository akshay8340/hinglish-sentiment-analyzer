"""Download the SentiMix (SemEval-2020 Task 9, Hindi-English) files into data/.

Only needed for training and evaluation. The demo app does not need the dataset.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/RTT1/SentiMix/resolve/main/"
FILES = {
    "train_14k_split_conll.txt": 3_000_000,
    "dev_3k_split_conll.txt": 600_000,
    "Hindi_test_unalbelled_conll_updated.txt": 600_000,
    "test_labels_hinglish.txt": 30_000,
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, min_bytes in FILES.items():
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"skip   {name} (already downloaded)")
            continue
        print(f"fetch  {name} ...", end=" ", flush=True)
        urllib.request.urlretrieve(BASE + name + "?download=true", dest)
        size = dest.stat().st_size
        if size < min_bytes:
            dest.unlink()
            sys.exit(f"\nDownload of {name} looks incomplete ({size} bytes). Check your connection and retry.")
        print(f"{size / 1e6:.2f} MB")
    print(f"Done. Files are in {DATA_DIR}")


if __name__ == "__main__":
    main()
