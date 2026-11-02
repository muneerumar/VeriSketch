"""Frozen-encoder feature extraction (DistilBERT-base-uncased, official weights,
SHA-256 verified).  Saves final-layer [CLS] and attention-masked mean-pooled
representations for every example; the federated model is the classification
head on top of the frozen encoder (parameter-efficient FL)."""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd, torch
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "model")
OFFICIAL_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
torch.set_num_threads(os.cpu_count())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_encoder():
    p = os.path.join(MD, "model.safetensors")
    assert sha256(p) == OFFICIAL_SHA256, "weights do not match the official checkpoint"
    tok = DistilBertTokenizerFast.from_pretrained(MD)
    assert tok("hello world")["input_ids"] == [101, 7592, 2088, 102], "tokenizer sanity check failed"
    enc = DistilBertModel.from_pretrained(MD).eval()
    return tok, enc


@torch.no_grad()
def encode(tok, enc, inputs, split_words=False, max_len=128, bs=64):
    lens = [len(x) if split_words else len(x.split()) for x in inputs]
    order = np.argsort(lens)
    cls = np.zeros((len(inputs), 768), np.float32); mean = np.zeros_like(cls)
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [inputs[i] for i in idx]
        b = tok(batch, is_split_into_words=split_words, truncation=True, max_length=max_len,
                padding=True, return_tensors="pt")
        h = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        cls[idx] = h[:, 0].numpy(); mean[idx] = ((h * m).sum(1) / m.sum(1)).numpy()
        if (s // bs) % 50 == 0:
            print(f"  {s + len(idx)}/{len(inputs)}  {time.time() - t0:.0f}s", flush=True)
    return cls, mean


if __name__ == "__main__":
    tok, enc = load_encoder()
    out = os.path.join(ROOT, "features"); os.makedirs(out, exist_ok=True)
    kg = pd.read_csv(os.path.join(ROOT, "data", "kaggle_clean.csv"))
    c, m = encode(tok, enc, kg.tweet_text.astype(str).tolist())
    np.save(os.path.join(out, "kaggle_cls.npy"), c); np.save(os.path.join(out, "kaggle_mean.npy"), m)
    hx = pd.read_json(os.path.join(ROOT, "data", "hatexplain_clean.jsonl"), lines=True)
    c, m = encode(tok, enc, hx.tokens.tolist(), split_words=True)
    np.save(os.path.join(out, "hx_cls.npy"), c); np.save(os.path.join(out, "hx_mean.npy"), m)
    json.dump({"weights_sha256": OFFICIAL_SHA256, "verified": True}, open(os.path.join(out, "provenance.json"), "w"))
    print("done")
