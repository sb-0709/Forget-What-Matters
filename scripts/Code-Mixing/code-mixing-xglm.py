# -*- coding: utf-8 -*-
"""
Hinglish Code-Mixed Unlearning - Code-Switching Aware Unlearning
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
!pip install -q transformers==4.36.0 lightning==2.1.3 torchmetrics pandas matplotlib

import torch
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

!nvidia-smi --query-gpu=name,memory.total --format=csv

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


@dataclass
class ConfigCodeMixed:
    # Model
    model_name: str = "facebook/xglm-564M"
    languages: List[str] = None
    primary_language: str = "hinglish"
    
    data_dir: str = "/content/data"  
    output_dir: str = "/content/ckpts_v2"  
    cache_dir: str = "/content/cache"
    
    # Data
    forget_num: int = 200
    retain_multiplier: int = 5
    max_seq_len: int = 256
    
    # Hyperparameters - TUNED
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    lambda_forget: float = 0.25       
    
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    
    max_epochs: int = 12               
    forget_every_n_epochs: int = 3     
    
    num_workers: int = 2
    seed: int = 42
    ma_max_samples: int = 10
    ma_stride: int = 4
    
    # CODE-MIXING MODIFICATIONS - TUNED
    is_code_mixed: bool = True
    
    # MOD 1: Language-specific forget schedule
    hinglish_forget_cycle: int = 3     
    
    # MOD 2: Adaptive temperature
    temperature: float = 1.0
    temperature_hinglish: float = 1.4  
    
    # MOD 4: Adaptive lambda
    lambda_forget_max: float = 0.40    
    
    # MOD 5: Code-switch-aware MA
    use_switch_aware_ma: bool = True
    
    # Language sampling for training
    use_language_sampling: bool = True
    hinglish_weight: float = 0.7
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["hinglish", "hi", "en"]

config = ConfigCodeMixed()

for d in [config.data_dir, config.output_dir, config.cache_dir]:
    os.makedirs(d, exist_ok=True)

L.seed_everything(config.seed)

print("\n" + "="*80)
print("🎯 TUNED CONFIGURATION (V2 - BETTER BALANCE)")
print("="*80)
print(f"Model: {config.model_name}")
print("="*80 + "\n")

from google.colab import files
print("📤 Upload 4 files with hinglish data:")
print("  - forget-200.jsonl")
print("  - retain-200-x5.jsonl")
print("  - test.jsonl")
print("  - valid.jsonl")
print("\nExpected format: {\"en\": ..., \"hi\": ..., \"hinglish\": ...}")
print()

uploaded = files.upload()
for f in uploaded:
    os.rename(f, os.path.join(config.data_dir, f))
    print(f"✅ {f}")

# Verify format
print("\n🔍 Verifying data format...")
with open(f"{config.data_dir}/forget-200.jsonl") as file:
    sample = json.loads(file.readline())
    print(f"Keys found: {list(sample.keys())}")
    if "hinglish" in sample:
        print(f"✅ Hinglish key found")
        print(f"Sample hinglish: {sample['hinglish'][:100]}...")
    else:
        print("⚠️ No 'hinglish' key! Expected format: {\"en\": ..., \"hi\": ..., \"hinglish\": ...}")

print("\n✅ Upload complete")

import os
import shutil

print("📁 Setting up directories...\n")

new_output_dir = "/content/ckpts_v2"
old_output_dir = "/content/ckpts"

if os.path.exists(new_output_dir):
    shutil.rmtree(new_output_dir)
    
os.makedirs(new_output_dir, exist_ok=True)
print(f"✅ Created: {new_output_dir}")

# Verify
print(f"\nDirectory structure:")
print(f"  V1 backup: {backup_dir}")
print(f"  V2 output: {new_output_dir}\n")

class HinglishDataset(Dataset):
    """Dataset for code-mixed Hinglish"""
    def __init__(self, data, tokenizer, max_len=256, languages=["hinglish"],
                 hinglish_weight=0.7, use_sampling=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.languages = languages
        self.hinglish_weight = hinglish_weight
        self.use_sampling = use_sampling

        if "hinglish" in languages:
            n_other = len(languages) - 1
            other_weight = (1 - hinglish_weight) / max(n_other, 1)
            self.lang_probs = []
            for lang in languages:
                if lang == "hinglish":
                    self.lang_probs.append(hinglish_weight)
                else:
                    self.lang_probs.append(other_weight)
        else:
            self.lang_probs = [1.0 / len(languages)] * len(languages)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_sampling and len(self.languages) > 1:
            lang = random.choices(self.languages, weights=self.lang_probs, k=1)[0]
        else:
            lang = self.languages[0]

        text = self.data[idx].get(lang, self.data[idx].get("hinglish", ""))

        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = encoded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class HinglishDataModule(L.LightningDataModule):
    """DataModule for Hinglish"""
    def __init__(self, cfg, tok):
        super().__init__()
        self.cfg = cfg
        self.tok = tok

    def load(self, filename):
        return [json.loads(line) for line in open(f"{self.cfg.data_dir}/{filename}")]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            forget_data = self.load(f"forget-{self.cfg.forget_num}.jsonl")
            retain_data = self.load(f"retain-{self.cfg.forget_num}-x{self.cfg.retain_multiplier}.jsonl")
            valid_data = self.load("valid.jsonl")

            self.forget = HinglishDataset(
                forget_data, self.tok, self.cfg.max_seq_len,
                self.cfg.languages, self.cfg.hinglish_weight, self.cfg.use_language_sampling
            )
            self.retain = HinglishDataset(
                retain_data, self.tok, self.cfg.max_seq_len,
                self.cfg.languages, self.cfg.hinglish_weight, self.cfg.use_language_sampling
            )

            self.valid = []
            for lang in self.cfg.languages:
                self.valid.append(HinglishDataset(
                    valid_data, self.tok, self.cfg.max_seq_len, [lang], use_sampling=False
                ))
                self.valid.append(HinglishDataset(
                    forget_data, self.tok, self.cfg.max_seq_len, [lang], use_sampling=False
                ))

        if stage == "test" or stage is None:
            forget_data = self.load(f"forget-{self.cfg.forget_num}.jsonl")
            test_data = self.load("test.jsonl")

            self.test = []
            for lang in self.cfg.languages:
                self.test.append(HinglishDataset(
                    test_data, self.tok, self.cfg.max_seq_len, [lang], use_sampling=False
                ))
                self.test.append(HinglishDataset(
                    forget_data, self.tok, self.cfg.max_seq_len, [lang], use_sampling=False
                ))

    def train_dataloader(self):
        is_forget = self.trainer.current_epoch % self.cfg.forget_every_n_epochs == 0
        dataset = self.forget if is_forget else self.retain
        return DataLoader(
            dataset,
            batch_size=self.cfg.per_device_train_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return [DataLoader(
            ds, batch_size=self.cfg.per_device_eval_batch_size,
            num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True
        ) for ds in self.valid]

    def test_dataloader(self):
        return [DataLoader(
            ds, batch_size=self.cfg.per_device_eval_batch_size,
            num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True
        ) for ds in self.test]

print("✅ Dataset classes ready")

class HinglishModelCodeMixed(L.LightningModule):
    """Model with code-mixing aware modifications"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
        self.model.gradient_checkpointing_enable()

        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.vnames = [f"val/{lang}_{t}_" for lang in cfg.languages for t in ["valid", "forget"]]
        self.tnames = [f"test/{lang}_{t}_" for lang in cfg.languages for t in ["test", "forget"]]

        self.val_ma_correct, self.val_ma_total = [], []
        self.test_ma_correct, self.test_ma_total = [], []

    def forward(self, **inputs):
        return self.model(**inputs)

    # MOD 1: Asymmetric forget schedule for code-mixed
    def _is_forget_epoch(self):
        if self.cfg.is_code_mixed:
            return self.current_epoch % self.cfg.hinglish_forget_cycle == 0
        return self.current_epoch % 2 == 0

    # MOD 2: Adaptive temperature
    def _get_temperature(self):
        if self.cfg.is_code_mixed:
            return self.cfg.temperature_hinglish
        return self.cfg.temperature

    # MOD 4: Detect code-mixing strength
    def _detect_code_mixing_strength(self, batch):
        if not self.cfg.is_code_mixed:
            return torch.ones(batch["input_ids"].size(0), device=self.device)

        mixing_strength = torch.zeros(batch["input_ids"].size(0), device=self.device)
        try:
            for idx in range(batch["input_ids"].size(0)):
                tokens = batch["input_ids"][idx]
                text = self.tok.decode(tokens, skip_special_tokens=True)
                has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
                has_latin = any(c.isalpha() and ord(c) < 128 for c in text)
                mixing_strength[idx] = float(has_devanagari and has_latin)
        except:
            pass
        return mixing_strength

    # MOD 4: Adaptive lambda
    def _compute_adaptive_lambda(self, batch):
        base_lambda = self.cfg.lambda_forget
        if self.cfg.is_code_mixed:
            mixing = self._detect_code_mixing_strength(batch)
            adaptive_lambda = base_lambda * (1.0 + mixing.mean())
            return adaptive_lambda.clamp(max=self.cfg.lambda_forget_max)
        return base_lambda

    def training_step(self, batch, batch_idx):
      outputs = self(**batch)
      loss = outputs.loss

      batch_size = batch["input_ids"].size(0)
      logit_s = outputs.logits
      padding_mask = batch["labels"].eq(-100)

      # MOD 2: Language-specific temperature
      temperature = self._get_temperature()

      self.teacher.eval()
      with torch.no_grad():
          outputs_t = self.teacher(**batch)
          logit_t = outputs_t.logits

      # MOD 1: Use asymmetric forget schedule
      if self._is_forget_epoch():
          # ===== FORGETTING PHASE =====
          # Key insight: Unlearning ≠ just negative gradient ascent
          # We want to reduce memorization while keeping model functional

          shift_logit_s = logit_s
          shift_labels = batch["labels"].new_zeros(batch["labels"].shape)
          shift_labels[:, :-1] = batch["labels"][:, 1:].clone()
          shift_labels[:, -1] = self.tok.pad_token_id

          labels = torch.clamp(batch["labels"], min=0)
          prob_t = F.softmax(logit_t, dim=-1)
          prob_t = prob_t.gather(dim=-1, index=labels.unsqueeze(-1))
          prob_t.masked_fill_(padding_mask.unsqueeze(-1), 0.0)

          # KL divergence: Penalize similarity to teacher
          # This makes student diverge from teacher (unlearning)
          loss_kd = F.kl_div(
              F.log_softmax(logit_s / temperature, dim=-1),
              F.softmax(logit_t / temperature, dim=-1),
              reduction="batchmean"
          ) * (temperature ** 2)

          # Cross entropy: Penalize correct predictions
          # Only for confident teacher predictions
          loss_ce = F.cross_entropy(
              shift_logit_s.view(-1, shift_logit_s.size(-1)),
              shift_labels.view(-1),
              reduction='mean'
          )

          # MOD 4: Adaptive lambda
          lambda_forget = self._compute_adaptive_lambda(batch)

          # CORRECTED: Use positive loss with adaptive weighting
          # DON'T use negative loss (that causes collapse)
          # Instead: weighted combination favoring KD (divergence from teacher)
          loss_forget = lambda_forget * loss_kd + (1 - lambda_forget) * loss_ce

          _dict = {
              "train/forget_loss": loss_forget,
              "train/kd_loss": loss_kd,
              "train/ce_loss": loss_ce,
              "train/lambda_forget": lambda_forget,
          }

          return loss_forget

      else:
          # ===== RETENTION PHASE =====
          # Standard training: minimize CE loss on retain set

          loss_retention = loss  # Use model's original loss

          _dict = {
              "train/retention_loss": loss_retention,
          }

          return loss_retention

      self.log_dict(
          _dict,
          on_step=True,
          on_epoch=True,
          prog_bar=True,
          logger=True,
          add_dataloader_idx=False,
          sync_dist=True,
      )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        ppl = torch.exp(loss.clamp(max=10))
        name = self.vnames[dataloader_idx] if dataloader_idx < len(self.vnames) else f"val/unk{dataloader_idx}_"

        correct, total = self._ma_counts(batch)

        while len(self.val_ma_correct) <= dataloader_idx:
            self.val_ma_correct.append(0)
            self.val_ma_total.append(0)

        self.val_ma_correct[dataloader_idx] += correct
        self.val_ma_total[dataloader_idx] += total

        self.log_dict({f"{name}ppl": ppl, f"{name}loss": loss}, add_dataloader_idx=False)
        torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        ppl = torch.exp(loss.clamp(max=10))
        name = self.tnames[dataloader_idx] if dataloader_idx < len(self.tnames) else f"test/unk{dataloader_idx}_"

        correct, total = self._ma_counts(batch)

        while len(self.test_ma_correct) <= dataloader_idx:
            self.test_ma_correct.append(0)
            self.test_ma_total.append(0)

        self.test_ma_correct[dataloader_idx] += correct
        self.test_ma_total[dataloader_idx] += total

        self.log_dict({f"{name}ppl": ppl, f"{name}loss": loss}, add_dataloader_idx=False)
        torch.cuda.empty_cache()
        return loss

    @torch.no_grad()
    # REPLACE the _ma_counts method in HinglishModelCodeMixed
  # ============================================================================
    def _ma_counts(self, batch):
        """Compute memorization accuracy counts - FIXED VERSION"""
        try:
            bs = batch["input_ids"].size(0)

            # Subsample if batch too large
            if bs > self.cfg.ma_max_samples:
                idx = torch.randperm(bs, device=batch["input_ids"].device)[:self.cfg.ma_max_samples]
                batch = {k: v[idx] if torch.is_tensor(v) else v for k, v in batch.items()}
                bs = self.cfg.ma_max_samples

            correct_count = 0
            total_count = 0
            positions_tested = 0

            # Test at fixed intervals
            for pos in range(self.cfg.ma_stride, self.cfg.max_seq_len, self.cfg.ma_stride):
                labels = batch["labels"][..., pos]

                # Skip if all labels are padding
                if (labels == -100).all():
                    break

                # Only process valid (non-padding) labels
                valid_mask = labels != -100

                if not valid_mask.any():
                    continue

                try:
                    # Get predictions using the model's forward pass (faster than generate)
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch["input_ids"][..., :pos],
                            attention_mask=batch["attention_mask"][..., :pos]
                        )
                        # Get logits at the last position
                        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                        predictions = logits.argmax(-1)    # [batch_size]

                    # Count correct predictions only for valid labels
                    is_correct = (predictions == labels) & valid_mask
                    correct_count += is_correct.sum().item()
                    total_count += valid_mask.sum().item()
                    positions_tested += 1

                except Exception as e:
                    print(f"⚠️ Error at position {pos}: {e}")
                    continue

                # Periodic cache clearing
                if positions_tested % 5 == 0:
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            if total_count == 0:
                # print(f"⚠️ Warning: No valid tokens to evaluate (all padding?)")
                return 0, 1

            return correct_count, total_count

        except Exception as e:
            print(f"❌ Error in _ma_counts: {type(e).__name__}: {e}")
            return 0, 1

    def on_validation_epoch_end(self):
        for i, (correct, total) in enumerate(zip(self.val_ma_correct, self.val_ma_total)):
            if total > 0:
                ma = correct / total
                name = self.vnames[i] if i < len(self.vnames) else f"val/unk{i}_"
                self.log(f"{name}ma", ma)

        forget_indices = [i for i, n in enumerate(self.vnames) if "forget" in n]
        if forget_indices:
            total_correct = sum(self.val_ma_correct[i] for i in forget_indices)
            total_count = sum(self.val_ma_total[i] for i in forget_indices)
            if total_count > 0:
                avg_forget_ma = total_correct / total_count
                self.log("val/forget_xma", avg_forget_ma)

                hinglish_idx = [i for i, n in enumerate(self.vnames) if "hinglish" in n and "forget" in n]
                if hinglish_idx:
                    h_correct = sum(self.val_ma_correct[i] for i in hinglish_idx)
                    h_total = sum(self.val_ma_total[i] for i in hinglish_idx)
                    if h_total > 0:
                        hinglish_ma = h_correct / h_total
                        self.log("val/hinglish_forget_ma", hinglish_ma)
                        print(f"\n📊 Epoch {self.current_epoch} | Forget MA: {avg_forget_ma:.2%} | Hinglish MA: {hinglish_ma:.2%}")

        self.val_ma_correct, self.val_ma_total = [], []
        torch.cuda.empty_cache()
        gc.collect()

    def on_test_epoch_end(self):
        for i, (correct, total) in enumerate(zip(self.test_ma_correct, self.test_ma_total)):
            if total > 0:
                ma = correct / total
                name = self.tnames[i] if i < len(self.tnames) else f"test/unk{i}_"
                self.log(f"{name}ma", ma)

        self.test_ma_correct, self.test_ma_total = [], []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.cfg.warmup_ratio * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

print("✅ Model class ready")

class MACallback(Callback):
    """Callback to save test metrics"""
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_test_end(self, trainer, pl_module):
        metrics = {}
        for k, v in trainer.logged_metrics.items():
            if k.startswith('test/'):
                clean_key = k.replace('test/', '')
                metrics[clean_key] = v.item() if hasattr(v, 'item') else v

        if metrics:
            df = pd.DataFrame([metrics])
            df.to_csv(f"{self.output_dir}/test_metrics.csv", index=False)
            print(f"💾 Saved {len(metrics)} test metrics")

print("✅ Callback ready")

print("Loading tokenizer and data...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dm = HinglishDataModule(config, tokenizer)
dm.setup("fit")

print(f"Forget: {len(dm.forget)} samples")
print(f"Retain: {len(dm.retain)} samples")
print(f"Valid: {len(dm.valid)} dataloaders ({len(config.languages)} langs × 2)")

model = HinglishModelCodeMixed(config)
print(f"Params: {sum(p.numel() for p in model.model.parameters())/1e6:.0f}M")

# Checkpoint callback
ckpt = ModelCheckpoint(
    dirpath=config.output_dir,
    filename="hinglish-e{epoch:02d}-fma{val/forget_xma:.3f}",
    monitor="val/forget_xma",
    mode="min",
    save_top_k=2,
    save_weights_only=True
)

# Trainer
trainer = L.Trainer(
    default_root_dir=config.output_dir,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    max_epochs=config.max_epochs,
    accumulate_grad_batches=config.gradient_accumulation_steps,
    gradient_clip_val=1.0,
    callbacks=[ckpt, MACallback(config.output_dir)],
    reload_dataloaders_every_n_epochs=1,
    num_sanity_val_steps=0
)

print("✅ Ready to train")

print("="*80)
print("🚀 TRAINING HINGLISH CODE-MIXED UNLEARNING")
print("="*80)
print(f"Model: {config.model_name}")
print(f"Modifications: MOD1 (asymmetric schedule), MOD2 (adaptive temp), MOD4 (adaptive lambda), MOD5 (switch-aware MA)")
print(f"Target: Forget MA < 30%")
print("="*80 + "\n")

trainer.fit(model, dm)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)

# ============================================================================

print("Reinitializing config and tokenizer...\n")

config = ConfigCodeMixed()

print(f"Config output_dir: {config.output_dir}")
print(f"Config model: {config.model_name}\n")

tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Tokenizer loaded")
print(f"✅ Config ready\n")

# ============================================================================

import os
import glob

print("Testing on best V2 checkpoint...\n")

# Find checkpoint directories
all_items = os.listdir(config.output_dir)
ckpt_dirs = sorted([d for d in all_items if d.startswith('hinglish-') and os.path.isdir(os.path.join(config.output_dir, d))])

if not ckpt_dirs:
    print(f"❌ No checkpoints found in {config.output_dir}")
else:
    best_ckpt_dir = os.path.join(config.output_dir, ckpt_dirs[-1])  # Latest epoch
    print(f"Checkpoint directory: {best_ckpt_dir}\n")
    
    # Find .ckpt file inside this directory
    ckpt_files = glob.glob(os.path.join(best_ckpt_dir, '*.ckpt'))
    
    if not ckpt_files:
        print(f"❌ No .ckpt file found in {best_ckpt_dir}")
        print(f"Contents: {os.listdir(best_ckpt_dir)}")
    else:
        best_ckpt = ckpt_files[0]
        print(f"Using: {best_ckpt}\n")
        
        torch.serialization.add_safe_globals([ConfigCodeMixed])
        
        best_model = HinglishModelCodeMixed.load_from_checkpoint(best_ckpt, cfg=config)
        print("✅ Model loaded\n")
        
        dm_test = HinglishDataModule(config, tokenizer)
        dm_test.setup("test")
        print("✅ Test data ready\n")
        
        print("Running inference...")
        print("="*80)
        
        trainer_test = L.Trainer(
            default_root_dir=config.output_dir,
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            enable_checkpointing=False,
        )
        
        trainer_test.test(best_model, dm_test)
        
        print("="*80)
        print("\n✅ Testing complete")

# STANDALONE RESULTS CELL - Works without previous variables
# ============================================================================

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("📊 FINAL RESULTS - HINGLISH CODE-MIXED UNLEARNING")
print("="*80 + "\n")

# Define output directory
output_dir = "/content/ckpts_v2"
languages = ["hinglish", "hi", "en"]

event_dir = f"{output_dir}/lightning_logs/version_1"

# Load event accumulator
ea = EventAccumulator(event_dir)
ea.Reload()

# Extract metrics
scalars = ea.Tags()['scalars']

# Parse results for each language
results = []

for lang in languages:
    print(f"\n{lang.upper()}")
    print("-" * 40)
    
    # Get metrics for this language
    test_ma_key = f"test/{lang}_test_ma"
    forget_ma_key = f"test/{lang}_forget_ma"
    test_ppl_key = f"test/{lang}_test_ppl"
    forget_ppl_key = f"test/{lang}_forget_ppl"
    
    # Extract values
    test_ma = ea.Scalars(test_ma_key)[-1].value if test_ma_key in scalars else None
    forget_ma = ea.Scalars(forget_ma_key)[-1].value if forget_ma_key in scalars else None
    test_ppl = ea.Scalars(test_ppl_key)[-1].value if test_ppl_key in scalars else None
    forget_ppl = ea.Scalars(forget_ppl_key)[-1].value if forget_ppl_key in scalars else None
    
    # Display
    if test_ma is not None:
        print(f"  Test MA:    {test_ma*100:6.2f}%")
    if forget_ma is not None:
        print(f"  Forget MA:  {forget_ma*100:6.2f}%")
    if test_ppl is not None:
        print(f"  Test PPL:   {test_ppl:8.2f}")
    if forget_ppl is not None:
        print(f"  Forget PPL: {forget_ppl:8.2f}")
    
    # Add to results
    if test_ma is not None and forget_ma is not None:
        results.append({
            "Language": lang.upper(),
            "Test MA": f"{test_ma*100:.1f}%",
            "Forget MA": f"{forget_ma*100:.1f}%",
            "Test PPL": f"{test_ppl:.1f}" if test_ppl else "N/A",
            "Forget PPL": f"{forget_ppl:.1f}" if forget_ppl else "N/A"
        })

# Display results table
if results:
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80 + "\n")
    
    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))
    
    # Analysis
    forget_mas = [float(r["Forget MA"].strip('%')) for r in results]
    avg_forget_ma = sum(forget_mas) / len(forget_mas)
    
    hinglish_result = [r for r in results if r["Language"] == "HINGLISH"]
    hinglish_fma = float(hinglish_result[0]["Forget MA"].strip('%')) if hinglish_result else None
    
    print("\n" + "="*80)
    print("🎯 ANALYSIS")
    print("="*80)
    print(f"\nAverage Forget MA (all languages): {avg_forget_ma:.1f}%")
    
    if hinglish_fma:
        print(f"Hinglish Forget MA (code-mixed):   {hinglish_fma:.1f}%")
    
    print()
    
    # Per-language analysis
    print("\n" + "-"*80)
    print("PER-LANGUAGE ANALYSIS")
    print("-"*80)
    
    for r in results:
        lang = r["Language"]
        fma = float(r["Forget MA"].strip('%'))
        
        if 25 <= fma <= 35:
            status = "✅"
        elif fma < 25:
            status = "⚠️ Over"
        else:
            status = "⚠️ Under"
        
        print(f"{lang:10} {status}  {fma:5.1f}% forget MA")
    
    print("\n" + "="*80)
    
    print("="*80)
    print("🎉 HINGLISH UNLEARNING EXPERIMENT COMPLETE!")
    print("="*80)
    print("\n" + "="*80)
