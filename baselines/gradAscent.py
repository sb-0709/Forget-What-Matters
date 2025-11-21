# baselines/gradAscent.py
import os
import gc
import math
import json
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, set_seed
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# ---- CONFIG ----
set_seed(42)
BASE_DIR = Path(__file__).parent.parent
FORGET_FILE = BASE_DIR / "data" / "forget-100.jsonl"
TEST_FILE   = BASE_DIR / "data" / "test.jsonl"

MODEL_NAME = "bigscience/bloom-560m" # or "facebook/xglm-564M" 
OUTPUT_DIR = BASE_DIR / "gradascent_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAINING_STEPS = 200          # start small
WARMUP_STEPS = 10
LR = 1e-6                         # very small for ascent
BATCH_SIZE = 1
ACCUM_STEPS = 1
GRAD_CLIP_NORM = 1.0
UNFREEZE_LAST_N_PARAMS = 50       # only update last N parameter tensors (safe)
USE_MPS_FALLBACK = True

# allow MPS fallback (if using mac)
if USE_MPS_FALLBACK:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ---- UTILITIES ----
def load_jsonl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def decide_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def refresh_mps():
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()

# ---- METRICS ----
def compute_ma_for_text(model, tokenizer, text, device, max_length=512):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    if input_ids.size(1) <= 1:
        return None
    with torch.no_grad():
        logits = model(input_ids).logits
    pred = logits.argmax(dim=-1)
    correct = (pred[:, :-1] == input_ids[:, 1:]).float().mean().item()
    return correct

def compute_ppl_for_text(model, tokenizer, text, device, max_length=512):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False, max_length=max_length)
    input_ids = enc.input_ids.to(device)
    if input_ids.size(1) == 0:
        return None
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    # guard against overflow: if loss is nan or huge, return None
    val = loss.item()
    if math.isnan(val) or val > 100.0:
        return float("inf")
    return math.exp(val)

# ---- LOAD MODEL & TOKENIZER ----
print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = decide_device()
print("Device:", device)
model.to(device)
model.train()

# ---- FREEZE/UNFREEZE PARAMETERS (safe fine-tune) ----
# freeze everything first
for p in model.parameters():
    p.requires_grad = False

# unfreeze only last N parameter tensors (by order)
all_params = list(model.named_parameters())
if UNFREEZE_LAST_N_PARAMS > 0:
    last_slice = all_params[-UNFREEZE_LAST_N_PARAMS:]
    for name, p in last_slice:
        p.requires_grad = True
    print(f"Unfroze last {len(last_slice)} parameter tensors (examples: {[n for n,_ in last_slice[:5]]} ... )")
else:
    # fallback: unfreeze lm_head if present
    for n, p in all_params:
        if "lm_head" in n or "embed" in n or "output" in n:
            p.requires_grad = True
            print("Unfroze", n)

# verify trainable parameter count
trainable = [n for n, p in all_params if p.requires_grad]
print(f"Trainable parameter tensors: {len(trainable)}")

# ---- OPTIMIZER & SCHEDULER (only for trainable params) ----
trainable_params = [p for n, p in all_params if p.requires_grad]
optimizer = AdamW(trainable_params, lr=LR)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=NUM_TRAINING_STEPS)

# ---- LOAD DATA ----
forget_rows = load_jsonl(FORGET_FILE)
test_rows   = load_jsonl(TEST_FILE)
if not forget_rows:
    raise RuntimeError(f"No forget examples in {FORGET_FILE}")

# ---- TRAIN LOOP: SAFE GRADIENT ASCENT (minimize -loss) ----
print("Starting constrained gradient ascent...")
for step in tqdm(range(NUM_TRAINING_STEPS), desc="GradAscent"):
    try:
        row = forget_rows[step % len(forget_rows)]
        # pick first text field in the example (same as before)
        lang_keys = [k for k, v in row.items() if isinstance(v, str)]
        if not lang_keys:
            continue
        text = row[lang_keys[0]]

        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = enc.input_ids.to(device)

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # this is the negative log-likelihood (to be maximized)

        # We *maximize* loss by minimizing negative loss (keeps optimizer moments coherent)
        neg_loss = -loss / ACCUM_STEPS
        neg_loss.backward()

        # gradient clipping (only on trainable params)
        clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)

        # optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # MPS housekeeping
        if device == "mps":
            refresh_mps()

        if step % 20 == 0:
            print(f"Step {step:4d}, loss={loss.item():.6f}")

    except RuntimeError as e:
        msg = str(e).lower()
        print(f"RuntimeError at step {step}: {e}")
        # If OOM on MPS, fallback to CPU (recreate optimizer on CPU)
        if ("mps" in device and "out of memory" in msg) or ("out of memory" in msg):
            print("OOM detected. Moving model to CPU and recreating optimizer.")
            model.to("cpu")
            device = "cpu"
            refresh_mps()
            # reassign trainable params referencing CPU tensors and recreate optimizer
            all_params = list(model.named_parameters())
            trainable_params = [p for n, p in all_params if p.requires_grad]
            optimizer = AdamW(trainable_params, lr=LR)
            scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=NUM_TRAINING_STEPS)
        else:
            raise

# ---- SAVE THE FINE-TUNED (AS CENTRIC) MODEL ----
SAVE_DIR = OUTPUT_DIR / "gradascent_safe_model"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Saved safe gradascent model to:", SAVE_DIR)

# ---- EVALUATE forget/test MA & PPL (same logic as your eval script) ----
def evaluate_rows(rows, model, tokenizer, device):
    # find language keys
    langs = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, str):
                langs.add(k)
    langs = sorted(langs)

    res = {}
    for lang in langs:
        texts = [r[lang] for r in rows if lang in r and isinstance(r[lang], str)]
        ma_vals, ppl_vals = [], []
        for text in tqdm(texts, desc=f"Eval {lang}"):
            ma = compute_ma_for_text(model, tokenizer, text, device)
            ppl = compute_ppl_for_text(model, tokenizer, text, device)
            if ma is not None: ma_vals.append(ma)
            if ppl is not None: ppl_vals.append(ppl)
        res[lang] = {
            "MA": sum(ma_vals)/len(ma_vals) if ma_vals else None,
            "PPL": sum(ppl_vals)/len(ppl_vals) if ppl_vals else None,
            "N": len(texts)
        }
    return res

# load saved model for evaluation (ensures eval mode)
# eval_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, use_fast=False)
# Always load original tokenizer for evaluation
eval_tokenizer = AutoTokenizer.from_pretrained(
    "bigscience/bloom-560m", 
    use_fast=False
)
eval_model = AutoModelForCausalLM.from_pretrained(SAVE_DIR)
eval_model.to(device)
eval_model.eval()

print("Evaluating forget/test sets...")
forget_results = evaluate_rows(forget_rows, eval_model, eval_tokenizer, device)
test_results = evaluate_rows(test_rows, eval_model, eval_tokenizer, device)

print("\n=== FINAL EVALUATION ===")
for lang in sorted(set(list(forget_results.keys()) + list(test_results.keys()))):
    f = forget_results.get(lang, {})
    t = test_results.get(lang, {})
    print(f"Language: {lang}")
    print(f"  Forget MA : {f.get('MA')}")
    print(f"  Test MA   : {t.get('MA')}")
    print(f"  Forget PPL: {f.get('PPL')}")
    print(f"  Test PPL  : {t.get('PPL')}")
    print("--------------------------------------------------")
