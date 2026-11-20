# Data

The dataset is **not** stored in this repository. Download it with:

```bash
python scripts/download_data.py
```

It fetches the Hindi-English portion of **SentiMix (SemEval-2020 Task 9)** from the public Hugging Face mirror
[`RTT1/SentiMix`](https://huggingface.co/datasets/RTT1/SentiMix) (the mirror's card lists the OpenRAIL license):

| File | Content |
|---|---|
| `train_14k_split_conll.txt` | 14,000 training tweets |
| `dev_3k_split_conll.txt` | 3,000 development tweets |
| `Hindi_test_unalbelled_conll_updated.txt` | 3,000 test tweets (no labels) |
| `test_labels_hinglish.txt` | labels of the test tweets (`Uid,Sentiment`) |

Format (CoNLL): a header line `meta<TAB>id<TAB>label` for each tweet, followed by one `word<TAB>language-tag`
line per word (`Hin`, `Eng`, `EMT`, `O`) and an empty line.
