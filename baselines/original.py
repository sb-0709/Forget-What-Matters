import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from tqdm import tqdm
import json
from pathlib import Path
import os


###############################################
# Paths (folder-local)
###############################################

BASE_DIR = Path(__file__).parent
FORGET_FILE = Path("../data/forget-100.jsonl")
TEST_FILE = Path("../data/test.jsonl")


###############################################
# Load ORIGINAL pretrained model (baseline)
###############################################
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "facebook/xglm-564M"   # or "bigscience/bloom-560m"

print(f"Loading model: {MODEL_NAME}")

print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    use_fast=False   # <-- CRITICAL FIX
)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)
model.eval()


###############################################
# JSONL loader
###############################################

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


###############################################
# Metrics
###############################################

def compute_ma(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc.input_ids.to(device)

    if input_ids.size(1) <= 1:
        return None

    with torch.no_grad():
        logits = model(input_ids).logits

    pred = logits.argmax(dim=-1)
    correct = (pred[:, :-1] == input_ids[:, 1:]).float().mean().item()
    return correct


def compute_pa(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc.input_ids.to(device)

    if input_ids.size(1) <= 1:
        return None

    with torch.no_grad():
        logits = model(input_ids).logits

    pred = logits.argmax(dim=-1)
    correct = (pred[:, 1:] == input_ids[:, 1:]).float().mean().item()
    return correct


def compute_ppl(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss

    return math.exp(loss.item())


###############################################
# Evaluate JSONL (language → scores)
###############################################

def evaluate_jsonl(path):
    rows = load_jsonl(path)

    # detect languages from keys across rows
    langs = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, str):
                langs.add(k)
    langs = sorted(langs)

    results = {}

    for lang in langs:
        texts = [r[lang] for r in rows if lang in r and isinstance(r[lang], str)]

        ma_vals, pa_vals, ppl_vals = [], [], []

        for text in tqdm(texts, desc=f"Evaluating {lang} ({path.name})"):
            ma = compute_ma(text)
            pa = compute_pa(text)
            ppl = compute_ppl(text)

            if ma is not None: ma_vals.append(ma)
            if pa is not None: pa_vals.append(pa)
            if ppl is not None: ppl_vals.append(ppl)

        results[lang] = {
            "MA": sum(ma_vals)/len(ma_vals) if ma_vals else None,
            "PA": sum(pa_vals)/len(pa_vals) if pa_vals else None,
            "PPL": sum(ppl_vals)/len(ppl_vals) if ppl_vals else None,
            "N": len(texts)
        }

    return results


###############################################
# MAIN
###############################################

if __name__ == "__main__":
    print(f"Loading JSONL files...")

    forget_results = evaluate_jsonl(FORGET_FILE)
    test_results   = evaluate_jsonl(TEST_FILE)

    print("\n=== BASELINE RESULTS (Original Model) ===\n")

    for lang in sorted(set(list(forget_results.keys()) + list(test_results.keys()))):
        f = forget_results.get(lang, {})
        t = test_results.get(lang, {})

        print(f"Language: {lang}")
        print(f"  Forget MA : {f.get('MA')}")
        print(f"  Test MA   : {t.get('MA')}")
        print(f"  Forget PPL: {f.get('PPL')}")
        print(f"  Test PPL  : {t.get('PPL')}")
        print("--------------------------------------------------")
