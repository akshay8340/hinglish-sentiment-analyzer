"""Fine-tune a transformer encoder (default: hinglish-minilm) for 3-class Hinglish sentiment, CPU friendly."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import LABELS, load_splits

MODEL_NAME = "anuragwagh0/hinglish-minilm"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "transformer"


def batches(texts, labels, tok, bs, max_len, shuffle):
    idx = list(range(len(texts)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        enc = tok([texts[j] for j in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield enc, torch.tensor([labels[j] for j in b])


@torch.no_grad()
def evaluate(model, texts, labels, tok, max_len):
    model.eval()
    preds = []
    for enc, _ in batches(texts, labels, tok, 64, max_len, shuffle=False):
        preds.extend(model(**enc).logits.argmax(-1).tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="use only N train rows (speed test)")
    ap.add_argument("--model", default=MODEL_NAME, help="HF model id (hing-bert is ~23s/step on a 4-core CPU)")
    args = ap.parse_args()
    model_name = args.model

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(torch.get_num_threads())

    train, dev = load_splits()
    if args.limit:
        train = train.sample(args.limit, random_state=42)
    X_tr, y_tr = train.text.tolist(), train.y.tolist()
    X_dv, y_dv = dev.text.tolist(), dev.y.tolist()

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}
    )

    steps = args.epochs * ((len(X_tr) + args.bs - 1) // args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best_f1, step, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for enc, y in batches(X_tr, y_tr, tok, args.bs, args.max_len, shuffle=True):
            loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                el = time.time() - t0
                print(f"epoch {epoch+1} step {step}/{steps} loss {loss.item():.3f} "
                      f"elapsed {el/60:.1f}m eta {el/step*(steps-step)/60:.1f}m", flush=True)

        pred = evaluate(model, X_dv, y_dv, tok, args.max_len)
        acc, f1 = accuracy_score(y_dv, pred), f1_score(y_dv, pred, average="macro")
        print(f"== epoch {epoch+1}: dev accuracy {acc:.4f}  macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
            print(classification_report(y_dv, pred, target_names=LABELS, digits=3))
            (OUT_DIR.parent / "bert_metrics.json").write_text(
                json.dumps({"accuracy": acc, "macro_f1": f1, "epoch": epoch + 1, "model": model_name}, indent=2)
            )


if __name__ == "__main__":
    main()
