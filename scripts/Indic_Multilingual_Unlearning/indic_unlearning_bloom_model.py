import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()
!nvidia-smi --query-gpu=name,memory.total --format=csv
print("✅ Setup complete")

import json, random, copy, gc
from typing import List
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
print("✅ Imports")

@dataclass
class Config:
    model_name: str = "bigscience/bloom-560m"
    languages: List[str] = None
    data_dir: str = "/content/data"
    output_dir: str = "/content/ckpts"
    cache_dir: str = "/content/cache"

    forget_num: int = 200
    retain_multiplier: int = 5
    max_seq_len: int = 256

    # CORRECTED: Less aggressive unlearning
    learning_rate: float = 3e-6 #1e-4  # Reduced from 5e-4
    warmup_ratio: float = 0.1
    temperature: float = 2.0
    lambda_forget: float = 0.15  # Scale down forget loss (gradient ascent)

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    max_epochs: int = 6  # More epochs for gradual forgetting
    forget_every_n_epochs: int = 2  # Forget every 4th epoch (0,4,8,12...)

    num_workers: int = 2
    seed: int = 42
    ma_max_samples: int = 10
    ma_stride: int = 16

    limit_val_batches: int = 10

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["hi","bn","te","ta","mr","gu","kn","ml","pa","ur","or","as","sd","bho","mai","san","kas_ar","kas_de"]

config = Config()
for d in [config.data_dir, config.output_dir, config.cache_dir]:
    os.makedirs(d, exist_ok=True)
L.seed_everything(config.seed)
print(f"Model: {config.model_name}")
print(f"LR: {config.learning_rate} (less aggressive)")
print(f"Forget schedule: Every {config.forget_every_n_epochs} epochs")
print(f"Lambda forget: {config.lambda_forget} (scaled down)")
print(f"Max epochs: {config.max_epochs}")
print("✅ Config")

# CELL 8: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted")

DRIVE_DATASET_PATH = "/content/drive/MyDrive/flores-jsonl"  # ⚠️ EDIT THIS

import shutil
import os

# Copy files from Drive to local data directory
print(f"📁 Loading dataset from: {DRIVE_DATASET_PATH}")

files_to_copy = [
    f"forget-200.jsonl",  # e.g., forget-200.jsonl
    f"retain-200-x5.jsonl",  # e.g., retain-200-x5.jsonl
    "test.jsonl",
    "valid.jsonl"
]

for filename in files_to_copy:
    source = os.path.join(DRIVE_DATASET_PATH, filename)
    dest = os.path.join(config.data_dir, filename)

    if os.path.exists(source):
        shutil.copy2(source, dest)
        print(f"✅ Copied: {filename}")
    else:
        print(f"❌ NOT FOUND: {filename}")
        print(f"   Looking in: {source}")

print("\n✅ Dataset loaded from Drive")

class FLORESDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256, languages=["hi"]):
        self.data, self.tokenizer, self.max_len, self.languages = data, tokenizer, max_len, languages
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        lang = random.choice(self.languages) if len(self.languages)>1 else self.languages[0]
        enc = self.tokenizer(self.data[idx][lang], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels==self.tokenizer.pad_token_id] = -100
        return {"input_ids":enc["input_ids"].squeeze(0), "attention_mask":enc["attention_mask"].squeeze(0), "labels":labels.squeeze(0)}

class FLORESDataModule(L.LightningDataModule):
    def __init__(self, cfg, tok):
        super().__init__()
        self.cfg, self.tok = cfg, tok
    def load(self, f): return [json.loads(l) for l in open(f"{self.cfg.data_dir}/{f}")]
    def setup(self, stage=None):
        if stage=="fit" or stage is None:
            fd = self.load(f"forget-{self.cfg.forget_num}.jsonl")
            rd = self.load(f"retain-{self.cfg.forget_num}-x{self.cfg.retain_multiplier}.jsonl")
            vd = self.load("valid.jsonl")
            self.forget = FLORESDataset(fd, self.tok, self.cfg.max_seq_len, self.cfg.languages)
            self.retain = FLORESDataset(rd, self.tok, self.cfg.max_seq_len, self.cfg.languages)
            self.valid = []
            for lang in self.cfg.languages:
                self.valid.append(FLORESDataset(vd, self.tok, self.cfg.max_seq_len, [lang]))
                self.valid.append(FLORESDataset(fd, self.tok, self.cfg.max_seq_len, [lang]))
        if stage=="test" or stage is None:
            fd = self.load(f"forget-{self.cfg.forget_num}.jsonl")
            td = self.load("test.jsonl")
            self.test = []
            for lang in self.cfg.languages:
                self.test.append(FLORESDataset(td, self.tok, self.cfg.max_seq_len, [lang]))
                self.test.append(FLORESDataset(fd, self.tok, self.cfg.max_seq_len, [lang]))
    def train_dataloader(self):
        # CORRECTED: Forget every N epochs, otherwise retain
        is_forget_epoch = self.trainer.current_epoch % self.cfg.forget_every_n_epochs == 0
        ds = self.forget if is_forget_epoch else self.retain
        return DataLoader(ds, batch_size=self.cfg.per_device_train_batch_size, num_workers=self.cfg.num_workers, shuffle=True, pin_memory=True)
    def val_dataloader(self):
        return [DataLoader(d, batch_size=self.cfg.per_device_eval_batch_size, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True) for d in self.valid]
    def test_dataloader(self):
        return [DataLoader(d, batch_size=self.cfg.per_device_eval_batch_size, num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True) for d in self.test]

print("✅ Dataset")

class Model(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
        self.model.gradient_checkpointing_enable()
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters(): p.requires_grad=False
        self.vnames = [f"val/{l}_{t}_" for l in cfg.languages for t in ["valid","forget"]]
        self.tnames = [f"test/{l}_{t}_" for l in cfg.languages for t in ["test","forget"]]
        self.val_ma_correct, self.val_ma_total = [], []
        self.test_ma_correct, self.test_ma_total = [], []

    def forward(self, **x): return self.model(**x)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        is_forget = self.current_epoch % self.cfg.forget_every_n_epochs == 0

        if is_forget:
            # Gradient ascent with scaling
            loss = -loss * self.cfg.lambda_forget  # Scale down
            self.log("train/forget_loss", loss, prog_bar=True)
        else:
            # Retention with KD
            logit_s = out.logits
            mask = batch["labels"].eq(-100)
            with torch.no_grad():
                logit_t = self.teacher(**batch).logits
            prob_t = F.softmax(logit_t, dim=-1)
            lbl = torch.clamp(batch["labels"], min=0)
            pt = prob_t.gather(-1, lbl.unsqueeze(-1))
            pt.masked_fill_(mask.unsqueeze(-1), 0)
            kappa = (pt.sum()/(~mask).sum()).clamp(0,1)
            lkd = F.kl_div(F.log_softmax(logit_s/self.cfg.temperature,-1), F.softmax(logit_t/self.cfg.temperature,-1), reduction="batchmean")*(self.cfg.temperature**2)
            del logit_t, prob_t
            loss = kappa*lkd + (1-kappa)*loss
            self.log_dict({"train/loss":loss,"train/kd":lkd,"train/kappa":kappa}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(**batch).loss
        ppl = torch.exp(loss.clamp(max=10))
        name = self.vnames[dataloader_idx] if dataloader_idx<len(self.vnames) else f"val/unk{dataloader_idx}_"
        corr, tot = self._ma_counts(batch)
        while len(self.val_ma_correct) <= dataloader_idx:
            self.val_ma_correct.append(0)
            self.val_ma_total.append(0)
        self.val_ma_correct[dataloader_idx] += corr
        self.val_ma_total[dataloader_idx] += tot
        self.log_dict({f"{name}ppl":ppl, f"{name}loss":loss}, add_dataloader_idx=False)
        torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(**batch).loss
        ppl = torch.exp(loss.clamp(max=10))
        name = self.tnames[dataloader_idx] if dataloader_idx<len(self.tnames) else f"test/unk{dataloader_idx}_"
        corr, tot = self._ma_counts(batch)
        while len(self.test_ma_correct) <= dataloader_idx:
            self.test_ma_correct.append(0)
            self.test_ma_total.append(0)
        self.test_ma_correct[dataloader_idx] += corr
        self.test_ma_total[dataloader_idx] += tot
        self.log_dict({f"{name}ppl":ppl, f"{name}loss":loss}, add_dataloader_idx=False)
        torch.cuda.empty_cache()
        return loss

    @torch.no_grad()
    def _ma_counts(self, batch):
        try:
            bs = batch["input_ids"].size(0)
            if bs > self.cfg.ma_max_samples:
                idx = torch.randperm(bs, device=batch["input_ids"].device)[:self.cfg.ma_max_samples]
                batch = {k:v[idx] if torch.is_tensor(v) else v for k,v in batch.items()}
            corr, tot = 0, 0
            for pos in range(self.cfg.ma_stride, self.cfg.max_seq_len, self.cfg.ma_stride):
                lbl = batch["labels"][...,pos]
                if (lbl==-100).all(): break
                out = self.model(input_ids=batch["input_ids"][...,:pos], attention_mask=batch["attention_mask"][...,:pos])
                pred = out.logits[:,-1,:].argmax(-1)
                valid = lbl!=-100
                corr += ((pred==lbl)&valid).sum().item()
                tot += valid.sum().item()
                del out
                if pos%(self.cfg.ma_stride*3)==0: torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            return corr, tot
        except: return 0, 1

    def on_validation_epoch_end(self):
        for i, (c, t) in enumerate(zip(self.val_ma_correct, self.val_ma_total)):
            if t>0:
                self.log(f"{self.vnames[i] if i<len(self.vnames) else f'val/unk{i}_'}ma", c/t)
        fidx = [i for i,n in enumerate(self.vnames) if "forget" in n]
        if fidx:
            tc = sum(self.val_ma_correct[i] for i in fidx)
            tt = sum(self.val_ma_total[i] for i in fidx)
            if tt>0:
                fma = tc/tt
                self.log("val/forget_xma", fma)
                print(f"\n📊 Epoch {self.current_epoch} Forget MA: {fma:.2%}")
        self.val_ma_correct, self.val_ma_total = [], []
        m = self.trainer.logged_metrics
        fppl = [v for k,v in m.items() if "ppl" in k and "forget" in k]
        if fppl: self.log("val/forget_xppl", torch.stack(fppl).mean())
        torch.cuda.empty_cache(); gc.collect()

    def on_test_epoch_end(self):
        for i, (c,t) in enumerate(zip(self.test_ma_correct, self.test_ma_total)):
            if t>0:
                self.log(f"{self.tnames[i] if i<len(self.tnames) else f'test/unk{i}_'}ma", c/t)
        self.test_ma_correct, self.test_ma_total = [], []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        sch = get_linear_schedule_with_warmup(opt, int(self.cfg.warmup_ratio*self.trainer.estimated_stepping_batches), self.trainer.estimated_stepping_batches)
        return {"optimizer":opt, "lr_scheduler":{"scheduler":sch,"interval":"step"}}

print("✅ Model")

class CB(Callback):
    def __init__(self, out): self.out = out
    def on_test_epoch_end(self, trainer, _):
        pd.DataFrame({k.replace("test/",""):[v.item()] for k,v in trainer.logged_metrics.items()}).to_csv(f"{self.out}/test.csv", index=False)

tok = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
if tok.pad_token is None: tok.pad_token = tok.eos_token
dm = FLORESDataModule(config, tok)
dm.setup("fit")
print(f"Forget:{len(dm.forget)} Retain:{len(dm.retain)} Valid:{len(dm.valid)}")
model = Model(config)
print(f"Params: {sum(p.numel() for p in model.model.parameters())/1e6:.0f}M")
ckpt = ModelCheckpoint(
    config.output_dir,
    "e{epoch:02d}-fppl{val/forget_xppl:.2f}",
    "val/forget_xppl",
    "max",  # Higher = more forgetting
    save_top_k=2,
    save_weights_only=True
)

trainer = L.Trainer(default_root_dir=config.output_dir, accelerator="gpu", devices=1, precision="bf16-mixed", max_epochs=config.max_epochs,
                    accumulate_grad_batches=config.gradient_accumulation_steps, gradient_clip_val=1.0,
                    callbacks=[ckpt, CB(config.output_dir)], reload_dataloaders_every_n_epochs=1, num_sanity_val_steps=0, limit_val_batches=10)
print("✅ Ready")

print(f"Forget every {config.forget_every_n_epochs} epochs ")
print(f"Retain on other epochs")
print(f"Lambda forget: {config.lambda_forget}")
print("="*80 + "\n")
trainer.fit(model, dm)
print("\n"+"="*80)
print("✅ DONE")
print("="*80)

import torch

# Allow Config class to be unpickled
torch.serialization.add_safe_globals([Config])

# Now load checkpoint
print("📊 TESTING")
best = Model.load_from_checkpoint(ckpt.best_model_path, cfg=config)
dm.setup("test")
trainer.test(best, dm)
print("✅ Done")

print("="*80)
print("📊 COMPUTING MA MANUALLY FROM TEST RESULTS")
print("="*80)

# Reload best model
torch.serialization.add_safe_globals([Config])
best = Model.load_from_checkpoint(ckpt.best_model_path, cfg=config)

# Setup test data
dm.setup("test")

# Create new callback that saves AFTER MA computation
class MACallback(Callback):
    def on_test_end(self, trainer, pl_module):
        metrics = {}
        for k, v in trainer.logged_metrics.items():
            if k.startswith('test/'):
                clean_key = k.replace('test/', '')
                metrics[clean_key] = v.item() if hasattr(v, 'item') else v

        if metrics:
            df = pd.DataFrame([metrics])
            df.to_csv(f"{config.output_dir}/test_with_ma.csv", index=False)
            print(f"\n💾 Saved {len(metrics)} metrics including MA")

            # Show MA columns
            ma_cols = [c for c in df.columns if '_ma' in c]
            print(f"MA columns: {ma_cols}")

# Create new trainer with MA callback
test_trainer = L.Trainer(
    default_root_dir=config.output_dir,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    callbacks=[MACallback()],
    enable_progress_bar=True
)

print("\n🔄 Re-running test with MA computation...")
test_trainer.test(best, dm)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# EXTRACT AND ORGANIZE RESULTS
# ============================================================================

print("="*80)
print("📊 INDIC LANGUAGE UNLEARNING - COMPREHENSIVE ANALYSIS")
print("="*80)

# Get metrics from trainer
all_metrics = {}
if hasattr(trainer, 'callback_metrics'):
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            all_metrics[key] = value.item()
        else:
            all_metrics[key] = float(value)

# Clean and organize results
results_dict = {}
for key, value in all_metrics.items():
    # Remove 'test/' prefix
    clean_key = key.replace('test/', '')
    if '_test_ma' in clean_key or '_forget_ma' in clean_key:
        results_dict[clean_key] = value

# Extract unique languages
languages = sorted(list(set([k.split('_')[0] for k in results_dict.keys()
                             if '_test_ma' in k or '_forget_ma' in k])))

print(f"\n📋 Found {len(languages)} languages: {', '.join(languages)}")

# Create detailed results dataframe
detailed_results = []
for lang in languages:
    test_key = f"{lang}_test_ma"
    forget_key = f"{lang}_forget_ma"

    if test_key in results_dict and forget_key in results_dict:
        test_val = results_dict[test_key] * 100
        forget_val = results_dict[forget_key] * 100

        detailed_results.append({
            "Language": lang.upper(),
            "Language_Code": lang,
            "Test MA (%)": round(test_val, 2),
            "Forget MA (%)": round(forget_val, 2),
            "Utility Retained (%)": round(test_val, 2),
            "Forgetting Effectiveness (%)": round(100 - forget_val, 2),
            "Utility-Forgetting Ratio": round(test_val / forget_val if forget_val > 0 else 0, 2)
        })

results_df = pd.DataFrame(detailed_results)
results_df = results_df.sort_values('Forgetting Effectiveness (%)', ascending=False)

# ============================================================================
# DISPLAY RESULTS TABLE
# ============================================================================

print("\n" + "="*80)
print("📋 DETAILED RESULTS BY LANGUAGE")
print("="*80)
print(results_df.to_string(index=False))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# EXTRACT AND ORGANIZE RESULTS
# ============================================================================

print("="*80)
print("📊 INDIC LANGUAGE UNLEARNING - COMPREHENSIVE ANALYSIS")
print("="*80)

# Get metrics from trainer
all_metrics = {}
if hasattr(trainer, 'callback_metrics'):
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            all_metrics[key] = value.item()
        else:
            all_metrics[key] = float(value)

# Clean and organize results
results_dict = {}
for key, value in all_metrics.items():
    # Remove 'test/' prefix
    clean_key = key.replace('test/', '')
    if '_test_ma' in clean_key or '_forget_ma' in clean_key:
        results_dict[clean_key] = value

# Extract unique languages
languages = sorted(list(set([k.split('_')[0] for k in results_dict.keys()
                             if '_test_ma' in k or '_forget_ma' in k])))

print(f"\n📋 Found {len(languages)} languages: {', '.join(languages)}")

# Create detailed results dataframe
detailed_results = []
for lang in languages:
    test_key = f"{lang}_test_ma"
    forget_key = f"{lang}_forget_ma"

    if test_key in results_dict and forget_key in results_dict:
        test_val = results_dict[test_key] * 100
        forget_val = results_dict[forget_key] * 100

        detailed_results.append({
            "Language": lang.upper(),
            "Language_Code": lang,
            "Test MA (%)": round(test_val, 2),
            "Forget MA (%)": round(forget_val, 2),
            "Utility Retained (%)": round(test_val, 2),
            "Forgetting Effectiveness (%)": round(100 - forget_val, 2),
            "Utility-Forgetting Ratio": round(test_val / forget_val if forget_val > 0 else 0, 2)
        })

results_df = pd.DataFrame(detailed_results)
results_df = results_df.sort_values('Forgetting Effectiveness (%)', ascending=False)

# ============================================================================
# DISPLAY RESULTS TABLE
# ============================================================================

print("\n" + "="*80)
print("📋 DETAILED RESULTS BY LANGUAGE")
print("="*80)
print(results_df.to_string(index=False))

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

stats = {
    "Number of Languages": len(languages),
    "Average Test MA (%)": results_df["Test MA (%)"].mean(),
    "Std Test MA (%)": results_df["Test MA (%)"].std(),
    "Min Test MA (%)": results_df["Test MA (%)"].min(),
    "Max Test MA (%)": results_df["Test MA (%)"].max(),
    "Average Forget MA (%)": results_df["Forget MA (%)"].mean(),
    "Std Forget MA (%)": results_df["Forget MA (%)"].std(),
    "Min Forget MA (%)": results_df["Forget MA (%)"].min(),
    "Max Forget MA (%)": results_df["Forget MA (%)"].max(),
    "Average Forgetting Effectiveness (%)": results_df["Forgetting Effectiveness (%)"].mean(),
    "Std Forgetting Effectiveness (%)": results_df["Forgetting Effectiveness (%)"].std(),
}

print("\n" + "="*80)
print("📈 SUMMARY STATISTICS")
print("="*80)
for key, value in stats.items():
    print(f"{key:.<45} {value:.2f}")


# ============================================================================
# SAVE DIRECTORY
# ============================================================================

save_dir = "/content/drive/MyDrive/indic_unlearning"
os.makedirs(save_dir, exist_ok=True)

# ============================================================================
# VISUALIZATION: MAIN COMPARISON CHART
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Test MA vs Forget MA (Side by side bars)
x = np.arange(len(results_df))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, results_df["Test MA (%)"], width,
                       label='Test MA (Utility)', color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = axes[0, 0].bar(x + width/2, results_df["Forget MA (%)"], width,
                       label='Forget MA (Residual)', color='#e74c3c', alpha=0.8, edgecolor='black')

axes[0, 0].set_xlabel('Language', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Match Accuracy (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Test MA vs Forget MA by Language', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(results_df["Language"], rotation=45, ha='right', fontsize=9)
axes[0, 0].legend(loc='upper right', fontsize=10)
axes[0, 0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (test_val, forget_val) in enumerate(zip(results_df["Test MA (%)"], results_df["Forget MA (%)"])):
    axes[0, 0].text(i - width/2, test_val + 0.5, f'{test_val:.1f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    axes[0, 0].text(i + width/2, forget_val + 0.5, f'{forget_val:.1f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

# Plot 2: Forgetting Effectiveness (Sorted)
sorted_df = results_df.sort_values('Forgetting Effectiveness (%)', ascending=True)
colors = ['#3498db' if val >= 65 else '#f39c12' if val >= 50 else '#e74c3c'
          for val in sorted_df["Forgetting Effectiveness (%)"]]

bars = axes[0, 1].barh(sorted_df["Language"], sorted_df["Forgetting Effectiveness (%)"],
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

axes[0, 1].axvline(x=65, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Good (≥65%)')
axes[0, 1].axvline(x=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate (≥50%)')

axes[0, 1].set_xlabel('Forgetting Effectiveness (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Language', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Forgetting Effectiveness by Language (Sorted)', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc='lower right', fontsize=9)
axes[0, 1].grid(axis='x', alpha=0.3)

# Add value labels
for idx, (lang, val) in enumerate(zip(sorted_df["Language"], sorted_df["Forgetting Effectiveness (%)"])):
    axes[0, 1].text(val + 1, idx, f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')

# Plot 3: Scatter - Utility vs Forgetting
scatter = axes[1, 0].scatter(results_df["Test MA (%)"], results_df["Forget MA (%)"],
                            s=200, alpha=0.6, c=results_df["Forgetting Effectiveness (%)"],
                            cmap='RdYlGn', edgecolors='black', linewidth=2, vmin=0, vmax=100)

# Add language labels
for idx, row in results_df.iterrows():
    axes[1, 0].annotate(row["Language"],
                       (row["Test MA (%)"], row["Forget MA (%)"]),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=8, fontweight='bold')

# Reference lines
max_val = max(results_df["Test MA (%)"].max(), results_df["Forget MA (%)"].max())
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.3, linewidth=2,
                label='Equal Performance Line')

axes[1, 0].set_xlabel('Test MA (%) - Utility Retained', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Forget MA (%) - Residual Memory', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Utility vs Forgetting Trade-off', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='upper left', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Forgetting Effectiveness (%)', fontsize=10)

# Plot 4: Utility-Forgetting Ratio
sorted_ratio = results_df.sort_values('Utility-Forgetting Ratio', ascending=True)
colors_ratio = ['#2ecc71' if val >= 2 else '#f39c12' if val >= 1 else '#e74c3c'
                for val in sorted_ratio["Utility-Forgetting Ratio"]]

bars = axes[1, 1].barh(sorted_ratio["Language"], sorted_ratio["Utility-Forgetting Ratio"],
                       color=colors_ratio, alpha=0.8, edgecolor='black', linewidth=1.5)

axes[1, 1].axvline(x=1, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                   label='Balanced (Ratio=1)')
axes[1, 1].axvline(x=2, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label='Good (Ratio≥2)')

axes[1, 1].set_xlabel('Utility / Forget Ratio', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Language', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Utility-to-Forgetting Ratio (Higher is Better)', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='lower right', fontsize=9)
axes[1, 1].grid(axis='x', alpha=0.3)

# Add value labels
for idx, (lang, val) in enumerate(zip(sorted_ratio["Language"], sorted_ratio["Utility-Forgetting Ratio"])):
    axes[1, 1].text(val + 0.1, idx, f'{val:.2f}', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{save_dir}/main_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n📊 Saved: main_analysis.png")
plt.show()

# ============================================================================
# VISUALIZATION: LANGUAGE CATEGORIES
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Categorize languages by performance
def categorize_forgetting(val):
    if val >= 75:
        return 'Excellent (≥75%)'
    elif val >= 50:
        return 'Good (50-75%)'
    elif val >= 25:
        return 'Moderate (25-50%)'
    else:
        return 'Poor (<25%)'

results_df['Forgetting Category'] = results_df['Forgetting Effectiveness (%)'].apply(categorize_forgetting)

category_counts = results_df['Forgetting Category'].value_counts()
colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

axes[0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Distribution of Languages by Forgetting Performance',
                 fontsize=13, fontweight='bold', pad=20)

# Average by category
category_avg = results_df.groupby('Forgetting Category').agg({
    'Test MA (%)': 'mean',
    'Forget MA (%)': 'mean',
    'Forgetting Effectiveness (%)': 'mean'
}).round(2)

x_cat = np.arange(len(category_avg))
width = 0.25

axes[1].bar(x_cat - width, category_avg['Test MA (%)'], width,
           label='Test MA', color='#2ecc71', alpha=0.8, edgecolor='black')
axes[1].bar(x_cat, category_avg['Forget MA (%)'], width,
           label='Forget MA', color='#e74c3c', alpha=0.8, edgecolor='black')
axes[1].bar(x_cat + width, category_avg['Forgetting Effectiveness (%)'], width,
           label='Effectiveness', color='#3498db', alpha=0.8, edgecolor='black')

axes[1].set_xlabel('Performance Category', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Average Percentage (%)', fontweight='bold', fontsize=11)
axes[1].set_title('Average Metrics by Performance Category', fontweight='bold', fontsize=13)
axes[1].set_xticks(x_cat)
axes[1].set_xticklabels(category_avg.index, rotation=15, ha='right', fontsize=9)
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{save_dir}/categories.png", dpi=300, bbox_inches='tight')
print(f"📊 Saved: categories.png")
plt.show()

# ============================================================================
# SAVE DETAILED REPORTS
# ============================================================================

# Save CSV
results_df.to_csv(f"{save_dir}/detailed_results.csv", index=False)
print(f"📊 Saved: detailed_results.csv")

# Save statistics summary
stats_df = pd.DataFrame([stats])
stats_df.to_csv(f"{save_dir}/summary_statistics.csv", index=False)
print(f"📊 Saved: summary_statistics.csv")

# Save JSON report
report = {
    "summary_statistics": stats,
    "performance_status": status,
    "interpretation": interpretation,
    "results_by_language": results_df.to_dict('records'),
    "category_distribution": category_counts.to_dict(),
    "top_10_languages": top_10[["Language", "Forgetting Effectiveness (%)"]].to_dict('records'),
    "bottom_10_languages": bottom_10[["Language", "Forgetting Effectiveness (%)"]].to_dict('records'),
}

with open(f"{save_dir}/analysis_report.json", 'w') as f:
    json.dump(report, f, indent=2)
print(f"📊 Saved: analysis_report.json")

# ============================================================================
# HIGH-RESOURCE vs LOW-RESOURCE INDIC LANGUAGES COMPARISON (SIMPLIFIED)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats as scipy_stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
 
# ============================================================================
# CATEGORIZE RESULTS
# ============================================================================

def get_resource_category(lang_code):
    """Categorize language as HIGH or LOW resource"""
    for category, info in LANGUAGE_CATEGORIES.items():
        if lang_code in info['languages']:
            return category
    return 'UNKNOWN'

results_df['Resource_Category'] = results_df['Language_Code'].apply(get_resource_category)

# Display categorization
print("\n📊 Language Categorization (Binary):")
print("-"*80)
for category, info in LANGUAGE_CATEGORIES.items():
    langs_in_category = results_df[results_df['Resource_Category'] == category]
    print(f"\n{category} ({len(langs_in_category)} languages):")
    print(f"  Description: {info['description']}")
    print(f"  Languages: {', '.join([l.upper() for l in sorted(langs_in_category['Language_Code'].tolist())])}")
    print(f"\n  Characteristics:")
    for char in info['characteristics']:
        print(f"    • {char}")

# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("📈 STATISTICAL COMPARISON")
print("="*80)

comparison_stats = {}

for category in ['HIGH_RESOURCE', 'LOW_RESOURCE']:
    category_data = results_df[results_df['Resource_Category'] == category]

    if len(category_data) > 0:
        comparison_stats[category] = {
            'Count': len(category_data),
            'Languages': sorted(category_data['Language_Code'].tolist()),
            'Avg Test MA (%)': category_data['Test MA (%)'].mean(),
            'Std Test MA (%)': category_data['Test MA (%)'].std(),
            'Min Test MA (%)': category_data['Test MA (%)'].min(),
            'Max Test MA (%)': category_data['Test MA (%)'].max(),
            'Avg Forget MA (%)': category_data['Forget MA (%)'].mean(),
            'Std Forget MA (%)': category_data['Forget MA (%)'].std(),
            'Min Forget MA (%)': category_data['Forget MA (%)'].min(),
            'Max Forget MA (%)': category_data['Forget MA (%)'].max(),
            'Avg Forgetting Effectiveness (%)': category_data['Forgetting Effectiveness (%)'].mean(),
            'Std Forgetting Effectiveness (%)': category_data['Forgetting Effectiveness (%)'].std(),
            'Min Forgetting Effectiveness (%)': category_data['Forgetting Effectiveness (%)'].min(),
            'Max Forgetting Effectiveness (%)': category_data['Forgetting Effectiveness (%)'].max(),
            'Avg Utility-Forgetting Ratio': category_data['Utility-Forgetting Ratio'].mean(),
            'Std Utility-Forgetting Ratio': category_data['Utility-Forgetting Ratio'].std(),
        }

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_stats).T

print("\n" + "="*80)
print("SUMMARY STATISTICS BY CATEGORY")
print("="*80)
print(comparison_df[['Count', 'Avg Test MA (%)', 'Std Test MA (%)',
                     'Avg Forget MA (%)', 'Std Forget MA (%)',
                     'Avg Forgetting Effectiveness (%)', 'Std Forgetting Effectiveness (%)']].to_string())

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================

print("\n" + "="*80)
print("🔬 STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

high_resource = results_df[results_df['Resource_Category'] == 'HIGH_RESOURCE']
low_resource = results_df[results_df['Resource_Category'] == 'LOW_RESOURCE']

print(f"\nSample sizes:")
print(f"  High-Resource: {len(high_resource)} languages")
print(f"  Low-Resource:  {len(low_resource)} languages")

if len(high_resource) > 1 and len(low_resource) > 1:
    # Test MA comparison
    t_stat_test, p_value_test = scipy_stats.ttest_ind(
        high_resource['Test MA (%)'],
        low_resource['Test MA (%)']
    )

    # Forget MA comparison
    t_stat_forget, p_value_forget = scipy_stats.ttest_ind(
        high_resource['Forget MA (%)'],
        low_resource['Forget MA (%)']
    )

    # Forgetting Effectiveness comparison
    t_stat_eff, p_value_eff = scipy_stats.ttest_ind(
        high_resource['Forgetting Effectiveness (%)'],
        low_resource['Forgetting Effectiveness (%)']
    )

    # Effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / pooled_std

    d_test = cohens_d(high_resource['Test MA (%)'], low_resource['Test MA (%)'])
    d_forget = cohens_d(high_resource['Forget MA (%)'], low_resource['Forget MA (%)'])
    d_eff = cohens_d(high_resource['Forgetting Effectiveness (%)'],
                     low_resource['Forgetting Effectiveness (%)'])

    print("\n" + "-"*80)
    print("Independent t-tests (High-Resource vs Low-Resource):")
    print("-"*80)

    def interpret_p(p):
        if p < 0.001: return "*** (highly significant)"
        elif p < 0.01: return "** (very significant)"
        elif p < 0.05: return "* (significant)"
        else: return "ns (not significant)"

    def interpret_d(d):
        d_abs = abs(d)
        if d_abs < 0.2: return "negligible"
        elif d_abs < 0.5: return "small"
        elif d_abs < 0.8: return "medium"
        else: return "large"

    print(f"\n1. Test MA (Utility):")
    print(f"   Difference: {high_resource['Test MA (%)'].mean():.2f}% - {low_resource['Test MA (%)'].mean():.2f}% = {high_resource['Test MA (%)'].mean() - low_resource['Test MA (%)'].mean():.2f}%")
    print(f"   t-statistic: {t_stat_test:.3f}")
    print(f"   p-value: {p_value_test:.4f} {interpret_p(p_value_test)}")
    print(f"   Effect size (Cohen's d): {d_test:.3f} ({interpret_d(d_test)})")

    print(f"\n2. Forget MA (Residual):")
    print(f"   Difference: {high_resource['Forget MA (%)'].mean():.2f}% - {low_resource['Forget MA (%)'].mean():.2f}% = {high_resource['Forget MA (%)'].mean() - low_resource['Forget MA (%)'].mean():.2f}%")
    print(f"   t-statistic: {t_stat_forget:.3f}")
    print(f"   p-value: {p_value_forget:.4f} {interpret_p(p_value_forget)}")
    print(f"   Effect size (Cohen's d): {d_forget:.3f} ({interpret_d(d_forget)})")

    print(f"\n3. Forgetting Effectiveness:")
    print(f"   Difference: {high_resource['Forgetting Effectiveness (%)'].mean():.2f}% - {low_resource['Forgetting Effectiveness (%)'].mean():.2f}% = {high_resource['Forgetting Effectiveness (%)'].mean() - low_resource['Forgetting Effectiveness (%)'].mean():.2f}%")
    print(f"   t-statistic: {t_stat_eff:.3f}")
    print(f"   p-value: {p_value_eff:.4f} {interpret_p(p_value_eff)}")
    print(f"   Effect size (Cohen's d): {d_eff:.3f} ({interpret_d(d_eff)})")

    print("\n" + "-"*80)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("Effect sizes: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, ≥ 0.8 large")
    print("-"*80)

