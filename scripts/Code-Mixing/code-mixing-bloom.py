import json, random, copy, gc
from typing import List
from dataclasses import dataclass
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ============================================================================
# CONFIGURATION - ENHANCED FOR CODE-SWITCHING
# ============================================================================

@dataclass
class ConfigHinglishEnhanced:
    model_name: str = "bigscience/bloom-560m"
    languages: List[str] = None
    primary_language: str = "hinglish"
    
    data_dir: str = "/content/data"
    output_dir: str = "/content/ckpts_hinglish_enhanced"
    cache_dir: str = "/content/cache"

    forget_num: int = 200
    retain_multiplier: int = 5
    max_seq_len: int = 256

    # Base hyperparameters
    learning_rate: float = 3e-6
    warmup_ratio: float = 0.1
    lambda_forget: float = 0.15
    
    # Batch sizes
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    max_epochs: int = 6
    
    # ========================================================================
    # CODE-SWITCHING SPECIFIC PARAMETERS (NEW!)
    # ========================================================================
    
    # MOD 1: Asymmetric Forget Schedule
    is_code_mixed: bool = True
    hinglish_forget_cycle: int = 3           # Forget every 3 epochs for code-mixed
    monolingual_forget_cycle: int = 2        # Standard cycle
    
    # MOD 2: Adaptive Temperature
    temperature_monolingual: float = 2.0     # Standard temperature
    temperature_hinglish: float = 2.5        # Higher for code-mixed (softer)
    
    # MOD 4: Adaptive Lambda
    lambda_forget_max: float = 0.30          # Maximum lambda for high mixing
    use_adaptive_lambda: bool = True
    mixing_boost_factor: float = 0.5         # How much to boost lambda for mixing
    
    # ========================================================================

    num_workers: int = 2
    seed: int = 42
    ma_max_samples: int = 10
    ma_stride: int = 16
    
    limit_val_batches: int = 10

    # Code-switching specific
    use_language_sampling: bool = True
    hinglish_weight: float = 0.7
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["hinglish", "hi", "en"]

config = ConfigHinglishEnhanced()

import os
for d in [config.data_dir, config.output_dir, config.cache_dir]:
    os.makedirs(d, exist_ok=True)

L.seed_everything(config.seed)

print("="*80)
print("🎯 BLOOM HINGLISH WITH CODE-SWITCHING MODS")
print("="*80)
print(f"Model: {config.model_name}")
print(f"Languages: {config.languages}")
print(f"\n🆕 CODE-SWITCHING MODIFICATIONS:")
print(f"  MOD 1 - Asymmetric Schedule: Every {config.hinglish_forget_cycle} epochs")
print(f"  MOD 2 - Adaptive Temperature: {config.temperature_hinglish} (vs {config.temperature_monolingual})")
print(f"  MOD 4 - Adaptive Lambda: {config.lambda_forget} → {config.lambda_forget_max}")
print("="*80 + "\n")

# ============================================================================
# DATASET - SAME AS BEFORE
# ============================================================================

class HinglishDataset(Dataset):
    """Dataset for code-mixed Hinglish"""
    def __init__(self, data, tokenizer, max_len=256, languages=["hinglish"],
                 hinglish_weight=0.7, use_sampling=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.languages = languages
        self.use_sampling = use_sampling
        
        if "hinglish" in languages and len(languages) > 1:
            n_other = len(languages) - 1
            other_weight = (1 - hinglish_weight) / max(n_other, 1)
            self.lang_probs = []
            for lang in languages:
                self.lang_probs.append(hinglish_weight if lang == "hinglish" else other_weight)
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
        
        if not text or len(text.strip()) == 0:
            text = "empty"
        
        try:
            enc = self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True
            )
        except Exception as e:
            print(f"❌ Tokenization failed: {e}")
            enc = self.tokenizer("error", max_length=self.max_len, padding="max_length",
                               truncation=True, return_tensors="pt")
        
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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
        is_forget = self.trainer.current_epoch % self.cfg.hinglish_forget_cycle == 0
        ds = self.forget if is_forget else self.retain
        return DataLoader(
            ds,
            batch_size=self.cfg.per_device_train_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return [DataLoader(
            d,
            batch_size=self.cfg.per_device_eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True
        ) for d in self.valid]
    
    def test_dataloader(self):
        return [DataLoader(
            d,
            batch_size=self.cfg.per_device_eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True
        ) for d in self.test]

print("✅ Dataset classes ready")

# ============================================================================
# MODEL - ENHANCED WITH CODE-SWITCHING MODS
# ============================================================================

class HinglishModelEnhanced(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        # CRITICAL: Force right-padding
        self.tok.padding_side = 'right'
        
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)
        self.model.gradient_checkpointing_enable()
        
        # Teacher (frozen)
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Metric names
        self.vnames = [f"val/{lang}_{t}_" for lang in cfg.languages for t in ["valid", "forget"]]
        self.tnames = [f"test/{lang}_{t}_" for lang in cfg.languages for t in ["test", "forget"]]
        
        # MA accumulators
        self.val_ma_correct, self.val_ma_total = [], []
        self.test_ma_correct, self.test_ma_total = [], []
        
        # Track mixing statistics
        self.epoch_mixing_stats = []
    
    def forward(self, **x):
        return self.model(**x)
    
    # ========================================================================
    # MOD 1: ASYMMETRIC FORGET SCHEDULE
    # ========================================================================
    def _is_forget_epoch(self):
        """Determine if current epoch is a forget epoch (MOD 1)"""
        if self.cfg.is_code_mixed:
            return self.current_epoch % self.cfg.hinglish_forget_cycle == 0
        else:
            return self.current_epoch % self.cfg.monolingual_forget_cycle == 0
    
    # ========================================================================
    # MOD 2: ADAPTIVE TEMPERATURE
    # ========================================================================
    def _get_temperature(self):
        """Get temperature based on code-mixing context (MOD 2)"""
        if self.cfg.is_code_mixed:
            return self.cfg.temperature_hinglish
        else:
            return self.cfg.temperature_monolingual
    
    # ========================================================================
    # MOD 4: DETECT CODE-MIXING STRENGTH
    # ========================================================================
    @torch.no_grad()
    def _detect_code_mixing_strength(self, batch):
        """
        Detect code-mixing strength in batch (MOD 4)
        Returns tensor of mixing scores [0, 1] for each sample
        1 = strong mixing (has both scripts), 0 = monolingual
        """
        if not self.cfg.is_code_mixed or not self.cfg.use_adaptive_lambda:
            return torch.zeros(batch["input_ids"].size(0), device=self.device)
        
        mixing_scores = []
        
        try:
            for idx in range(batch["input_ids"].size(0)):
                tokens = batch["input_ids"][idx]
                # Decode only non-padding tokens
                valid_tokens = tokens[batch["attention_mask"][idx] == 1]
                text = self.tok.decode(valid_tokens, skip_special_tokens=True)
                
                # Check for Devanagari script (Hindi)
                has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
                
                # Check for Latin script (English)
                has_latin = any(c.isalpha() and ord(c) < 128 for c in text)
                
                # Score: 1.0 if both scripts present (code-mixed), 0.0 otherwise
                mixing_score = 1.0 if (has_devanagari and has_latin) else 0.0
                mixing_scores.append(mixing_score)
                
        except Exception as e:
            # Fallback: assume no mixing
            print(f"⚠️ Error detecting mixing: {e}")
            mixing_scores = [0.0] * batch["input_ids"].size(0)
        
        return torch.tensor(mixing_scores, device=self.device)
    
    # ========================================================================
    # MOD 4: ADAPTIVE LAMBDA BASED ON MIXING
    # ========================================================================
    def _compute_adaptive_lambda(self, batch):
        """
        Compute adaptive lambda based on code-mixing strength (MOD 4)
        More mixing → higher lambda → more aggressive forgetting
        """
        base_lambda = self.cfg.lambda_forget
        
        if not self.cfg.use_adaptive_lambda:
            return base_lambda
        
        # Get mixing strength for batch
        mixing_strength = self._detect_code_mixing_strength(batch)
        avg_mixing = mixing_strength.mean().item()
        
        # Adaptive formula: lambda = base * (1 + boost_factor * mixing)
        adaptive_lambda = base_lambda * (1.0 + self.cfg.mixing_boost_factor * avg_mixing)
        
        # Clamp to maximum
        adaptive_lambda = min(adaptive_lambda, self.cfg.lambda_forget_max)
        
        return adaptive_lambda
    
    # ========================================================================
    # TRAINING STEP - ENHANCED WITH ALL MODS
    # ========================================================================
    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        
        # MOD 1: Use asymmetric forget schedule
        is_forget = self._is_forget_epoch()
        
        if is_forget:
            # FORGETTING PHASE with MOD 4 (Adaptive Lambda)
            
            # Compute adaptive lambda based on code-mixing
            lambda_forget = self._compute_adaptive_lambda(batch)
            
            # Get mixing statistics
            mixing_strength = self._detect_code_mixing_strength(batch)
            avg_mixing = mixing_strength.mean().item()
            
            # Gradient ascent with adaptive scaling
            loss = -loss * lambda_forget
            
            self.log_dict({
                "train/forget_loss": loss,
                "train/lambda": lambda_forget,
                "train/mixing_strength": avg_mixing
            }, prog_bar=True)
            
        else:
            # RETENTION PHASE with MOD 2 (Adaptive Temperature)
            
            logit_s = out.logits
            mask = batch["labels"].eq(-100)
            
            with torch.no_grad():
                logit_t = self.teacher(**batch).logits
            
            # Calculate kappa
            prob_t = F.softmax(logit_t, dim=-1)
            lbl = torch.clamp(batch["labels"], min=0)
            pt = prob_t.gather(-1, lbl.unsqueeze(-1))
            pt.masked_fill_(mask.unsqueeze(-1), 0)
            kappa = (pt.sum() / (~mask).sum()).clamp(0, 1)
            
            # MOD 2: Use adaptive temperature
            temperature = self._get_temperature()
            
            # KD loss with adaptive temperature
            lkd = F.kl_div(
                F.log_softmax(logit_s / temperature, -1),
                F.softmax(logit_t / temperature, -1),
                reduction="batchmean"
            ) * (temperature ** 2)
            
            del logit_t, prob_t
            
            loss = kappa * lkd + (1 - kappa) * loss
            
            self.log_dict({
                "train/loss": loss,
                "train/kd": lkd,
                "train/kappa": kappa,
                "train/temperature": temperature
            }, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(**batch).loss
        ppl = torch.exp(loss.clamp(max=10))
        name = self.vnames[dataloader_idx] if dataloader_idx < len(self.vnames) else f"val/unk{dataloader_idx}_"
        
        corr, tot = self._ma_counts(batch)
        
        while len(self.val_ma_correct) <= dataloader_idx:
            self.val_ma_correct.append(0)
            self.val_ma_total.append(0)
        
        self.val_ma_correct[dataloader_idx] += corr
        self.val_ma_total[dataloader_idx] += tot
        
        self.log_dict({f"{name}ppl": ppl, f"{name}loss": loss}, add_dataloader_idx=False)
        torch.cuda.empty_cache()
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self(**batch).loss
        ppl = torch.exp(loss.clamp(max=10))
        name = self.tnames[dataloader_idx] if dataloader_idx < len(self.tnames) else f"test/unk{dataloader_idx}_"
        
        corr, tot = self._ma_counts(batch)
        
        while len(self.test_ma_correct) <= dataloader_idx:
            self.test_ma_correct.append(0)
            self.test_ma_total.append(0)
        
        self.test_ma_correct[dataloader_idx] += corr
        self.test_ma_total[dataloader_idx] += tot
        
        self.log_dict({f"{name}ppl": ppl, f"{name}loss": loss}, add_dataloader_idx=False)
        torch.cuda.empty_cache()
        return loss
    
    @torch.no_grad()
    def _ma_counts(self, batch):
        """MA calculation with adaptive stride"""
        try:
            bs = batch["input_ids"].size(0)
            if bs > self.cfg.ma_max_samples:
                idx = torch.randperm(bs, device=batch["input_ids"].device)[:self.cfg.ma_max_samples]
                batch = {k: v[idx] if torch.is_tensor(v) else v for k, v in batch.items()}
            
            corr, tot = 0, 0
            
            seq_lengths = (batch["labels"] != -100).sum(dim=1)
            min_seq_len = seq_lengths.min().item()
            
            if min_seq_len < 5:
                return 0, 1
            
            adaptive_stride = min(self.cfg.ma_stride, max(4, min_seq_len // 4))
            start_pos = adaptive_stride
            end_pos = min(min_seq_len, self.cfg.max_seq_len)
            test_range = list(range(start_pos, end_pos, adaptive_stride))
            
            for pos in test_range:
                lbl = batch["labels"][..., pos]
                
                if (lbl == -100).all():
                    continue
                
                try:
                    out = self.model(
                        input_ids=batch["input_ids"][..., :pos],
                        attention_mask=batch["attention_mask"][..., :pos]
                    )
                    
                    pred = out.logits[:, -1, :].argmax(-1)
                    valid = lbl != -100
                    corr += ((pred == lbl) & valid).sum().item()
                    tot += valid.sum().item()
                    
                    del out
                    
                except Exception as e:
                    continue
            
            torch.cuda.empty_cache()
            return corr, tot
            
        except Exception as e:
            print(f"❌ MA error: {e}")
            return 0, 1
    
    def on_validation_epoch_end(self):
        for i, (c, t) in enumerate(zip(self.val_ma_correct, self.val_ma_total)):
            if t > 0:
                ma_val = c / t
                self.log(f"{self.vnames[i] if i < len(self.vnames) else f'val/unk{i}_'}ma", ma_val)
        
        fidx = [i for i, n in enumerate(self.vnames) if "forget" in n]
        if fidx:
            tc = sum(self.val_ma_correct[i] for i in fidx)
            tt = sum(self.val_ma_total[i] for i in fidx)
            fma = tc / tt if tt > 0 else 0.0
            self.log("val/forget_xma", fma)
            
            hinglish_idx = [i for i, n in enumerate(self.vnames) if "hinglish" in n and "forget" in n]
            if hinglish_idx:
                h_correct = sum(self.val_ma_correct[i] for i in hinglish_idx)
                h_total = sum(self.val_ma_total[i] for i in hinglish_idx)
                if h_total > 0:
                    hinglish_ma = h_correct / h_total
                    self.log("val/hinglish_forget_ma", hinglish_ma)
                    print(f"\n📊 Epoch {self.current_epoch} | Forget MA: {fma:.2%} | Hinglish MA: {hinglish_ma:.2%}")
        
        self.val_ma_correct, self.val_ma_total = [], []
        
        m = self.trainer.logged_metrics
        fppl = [v for k, v in m.items() if "ppl" in k and "forget" in k]
        if fppl:
            self.log("val/forget_xppl", torch.stack(fppl).mean())
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def on_test_epoch_end(self):
        for i, (c, t) in enumerate(zip(self.test_ma_correct, self.test_ma_total)):
            if t > 0:
                self.log(f"{self.tnames[i] if i < len(self.tnames) else f'test/unk{i}_'}ma", c / t)
        self.test_ma_correct, self.test_ma_total = [], []
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        sch = get_linear_schedule_with_warmup(
            opt,
            int(self.cfg.warmup_ratio * self.trainer.estimated_stepping_batches),
            self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

print("✅ Model class with CODE-SWITCHING MODS ready")

# ============================================================================
# CALLBACK
# ============================================================================

class CB(Callback):
    def __init__(self, out):
        self.out = out
    
    def on_test_epoch_end(self, trainer, _):
        metrics = {k.replace("test/", ""): v.item() for k, v in trainer.logged_metrics.items()}
        pd.DataFrame([metrics]).to_csv(f"{self.out}/test.csv", index=False)

print("✅ Callback ready")

# ============================================================================
# INITIALIZE & TRAIN
# ============================================================================

print("\n" + "="*80)
print("🚀 INITIALIZING WITH CODE-SWITCHING ENHANCEMENTS")
print("="*80)

tok = AutoTokenizer.from_pretrained(
    config.model_name,
    cache_dir=config.cache_dir,
    use_fast=True,
    trust_remote_code=True
)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

tok.padding_side = 'right'

print(f"Tokenizer: {tok.__class__.__name__}")
print(f"Vocab size: {len(tok)}")
print(f"Pad token: '{tok.pad_token}' (ID: {tok.pad_token_id})")
print(f"Padding side: {tok.padding_side}")

dm = HinglishDataModule(config, tok)
dm.setup("fit")
print(f"\nDatasets: Forget:{len(dm.forget)} | Retain:{len(dm.retain)} | Valid:{len(dm.valid)}")

model = HinglishModelEnhanced(config)
print(f"Params: {sum(p.numel() for p in model.model.parameters())/1e6:.0f}M")

ckpt = ModelCheckpoint(
    config.output_dir,
    "e{epoch:02d}-fma{val/forget_xma:.3f}",
    "val/forget_xma",
    "min",
    save_top_k=2,
    save_weights_only=True
)

trainer = L.Trainer(
    default_root_dir=config.output_dir,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    max_epochs=config.max_epochs,
    accumulate_grad_batches=config.gradient_accumulation_steps,
    gradient_clip_val=1.0,
    callbacks=[ckpt, CB(config.output_dir)],
    reload_dataloaders_every_n_epochs=1,
    num_sanity_val_steps=0,
    limit_val_batches=config.limit_val_batches
)

print("\n" + "="*80)
print("🚀 TRAINING WITH CODE-SWITCHING MODIFICATIONS")
print("="*80)
print("🆕 Active MODs:")
print(f"  ✓ MOD 1: Asymmetric schedule (every {config.hinglish_forget_cycle} epochs)")
print(f"  ✓ MOD 2: Adaptive temperature ({config.temperature_hinglish})")
print(f"  ✓ MOD 4: Adaptive lambda (detects mixing, adjusts {config.lambda_forget}→{config.lambda_forget_max})")
print("="*80 + "\n")

trainer.fit(model, dm)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)

import os
import glob
import torch
import pandas as pd
import lightning as L

print("="*80)
print("🧪 TESTING PHASE - BLOOM HINGLISH WITH CODE-SWITCHING MODS")
print("="*80)

# ============================================================================
# STEP 1: FIND BEST CHECKPOINT
# ============================================================================

print(f"\n📂 Searching for checkpoints in: {config.output_dir}")

# Find all checkpoint files
ckpt_pattern = os.path.join(config.output_dir, "*.ckpt")
ckpt_files = glob.glob(ckpt_pattern)

if not ckpt_files:
    print(f"❌ No .ckpt files found in {config.output_dir}")
    print(f"\nContents of directory:")
    print(os.listdir(config.output_dir))
    
    # Try looking in subdirectories
    print(f"\nSearching in subdirectories...")
    all_items = os.listdir(config.output_dir)
    ckpt_dirs = [d for d in all_items if os.path.isdir(os.path.join(config.output_dir, d))]
    
    for ckpt_dir in ckpt_dirs:
        full_path = os.path.join(config.output_dir, ckpt_dir)
        sub_ckpts = glob.glob(os.path.join(full_path, "*.ckpt"))
        if sub_ckpts:
            ckpt_files.extend(sub_ckpts)
            print(f"  Found in {ckpt_dir}: {len(sub_ckpts)} checkpoint(s)")

if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint files found in {config.output_dir} or subdirectories")

# Sort by modification time to get the latest/best
ckpt_files.sort(key=os.path.getmtime, reverse=True)

# Or sort by filename if they contain epoch/metric info
# ckpt_files.sort(reverse=True)

best_ckpt = ckpt_files[0]
print(f"\n✅ Found {len(ckpt_files)} checkpoint(s)")
print(f"📌 Using: {os.path.basename(best_ckpt)}")
print(f"   Full path: {best_ckpt}")

# ============================================================================
# STEP 2: LOAD MODEL
# ============================================================================

print(f"\n🔄 Loading model from checkpoint...")

# Allow config class to be unpickled
torch.serialization.add_safe_globals([ConfigHinglishEnhanced])

try:
    best_model = HinglishModelEnhanced.load_from_checkpoint(
        best_ckpt,
        cfg=config
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    print("\n🔄 Trying alternative loading method...")
    
    # Alternative: Load state dict manually
    checkpoint = torch.load(best_ckpt, map_location='cpu')
    best_model = HinglishModelEnhanced(config)
    best_model.load_state_dict(checkpoint['state_dict'])
    print("✅ Model loaded via state_dict")

# ============================================================================
# STEP 3: SETUP TEST DATA
# ============================================================================

print(f"\n📦 Setting up test data...")

# Check if tok/tokenizer exists, otherwise recreate
if 'tok' not in globals():
    print("   Creating tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        use_fast=True,
        trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = 'right'

dm_test = HinglishDataModule(config, tok)
dm_test.setup("test")

print(f"✅ Test data ready")
print(f"   Test dataloaders: {len(dm_test.test)}")

# ============================================================================
# STEP 4: RUN TESTING
# ============================================================================

print(f"\n🚀 Running test evaluation...")
print("="*80)

trainer_test = L.Trainer(
    default_root_dir=config.output_dir,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    enable_checkpointing=False,
    enable_progress_bar=True,
    logger=False
)

test_results = trainer_test.test(best_model, dm_test)

print("="*80)
print("\n✅ TESTING COMPLETE")

# ============================================================================
# STEP 5: EXTRACT AND ANALYZE RESULTS
# ============================================================================

print("\n" + "="*80)
print("📊 TEST RESULTS ANALYSIS")
print("="*80)

# Extract all test metrics
test_metrics = {}

# Try different sources for metrics
if hasattr(trainer_test, 'logged_metrics') and trainer_test.logged_metrics:
    for key, value in trainer_test.logged_metrics.items():
        test_metrics[key] = value.item() if hasattr(value, 'item') else float(value)
    print(f"\n✅ Found {len(test_metrics)} metrics from logged_metrics")

elif hasattr(trainer_test, 'callback_metrics') and trainer_test.callback_metrics:
    for key, value in trainer_test.callback_metrics.items():
        test_metrics[key] = value.item() if hasattr(value, 'item') else float(value)
    print(f"\n✅ Found {len(test_metrics)} metrics from callback_metrics")

elif test_results and len(test_results) > 0:
    test_metrics = test_results[0]
    print(f"\n✅ Found {len(test_metrics)} metrics from test_results")

else:
    print("\n⚠️ No metrics found from standard sources")
    print("Checking model's logged metrics...")
    if hasattr(best_model, 'test_metrics'):
        test_metrics = best_model.test_metrics

# Filter to only test metrics
test_metrics_filtered = {
    k.replace('test/', ''): v 
    for k, v in test_metrics.items() 
    if k.startswith('test/')
}

if not test_metrics_filtered:
    test_metrics_filtered = {k: v for k, v in test_metrics.items()}

print(f"📋 Processing {len(test_metrics_filtered)} test metrics")

if test_metrics_filtered:
    print("\nSample metrics:")
    for i, (key, val) in enumerate(list(test_metrics_filtered.items())[:5]):
        print(f"  {key}: {val:.4f}")

# ============================================================================
# ORGANIZE BY LANGUAGE
# ============================================================================

results_by_lang = {}
for lang in config.languages:
    # Try different possible key formats
    test_ma = (test_metrics_filtered.get(f'{lang}_test_ma', 0) or 
               test_metrics_filtered.get(f'test/{lang}_test_ma', 0) or 0) * 100
    
    forget_ma = (test_metrics_filtered.get(f'{lang}_forget_ma', 0) or
                 test_metrics_filtered.get(f'test/{lang}_forget_ma', 0) or 0) * 100
    
    test_ppl = (test_metrics_filtered.get(f'{lang}_test_ppl', 0) or
                test_metrics_filtered.get(f'test/{lang}_test_ppl', 0) or 0)
    
    forget_ppl = (test_metrics_filtered.get(f'{lang}_forget_ppl', 0) or
                  test_metrics_filtered.get(f'test/{lang}_forget_ppl', 0) or 0)
    
    # Only add if we have valid data
    if test_ma > 0 or forget_ma > 0 or test_ppl > 0:
        results_by_lang[lang] = {
            'test_ma': test_ma,
            'forget_ma': forget_ma,
            'test_ppl': test_ppl,
            'forget_ppl': forget_ppl
        }

# ============================================================================
# CREATE RESULTS TABLE
# ============================================================================

if results_by_lang:
    results_data = []
    for lang, metrics in results_by_lang.items():
        test_ma = metrics['test_ma']
        forget_ma = metrics['forget_ma']
        test_ppl = metrics['test_ppl']
        forget_ppl = metrics['forget_ppl']
        
        results_data.append({
            'Language': lang.upper(),
            'Test MA (%)': round(test_ma, 2),
            'Forget MA (%)': round(forget_ma, 2),
            'Forgetting Effectiveness (%)': round(100 - forget_ma, 2),
            'Utility Retained (%)': round(test_ma, 2),
            'Test PPL': round(test_ppl, 2) if test_ppl > 0 else 0,
            'Forget PPL': round(forget_ppl, 2) if forget_ppl > 0 else 0,
            'PPL Ratio (F/T)': round(forget_ppl / test_ppl, 2) if test_ppl > 0 else 0
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Forgetting Effectiveness (%)', ascending=False)
    
    print("\n" + "="*80)
    print("📋 RESULTS BY LANGUAGE")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)
    
    avg_test_ma = results_df['Test MA (%)'].mean()
    avg_forget_ma = results_df['Forget MA (%)'].mean()
    avg_effectiveness = results_df['Forgetting Effectiveness (%)'].mean()
    
    # Only calculate PPL ratio if we have valid PPL values
    valid_ratios = results_df[results_df['PPL Ratio (F/T)'] > 0]['PPL Ratio (F/T)']
    avg_ppl_ratio = valid_ratios.mean() if len(valid_ratios) > 0 else 0
    
    print(f"Average Test MA (Utility):        {avg_test_ma:.2f}%")
    print(f"Average Forget MA (Residual):     {avg_forget_ma:.2f}%")
    print(f"Average Forgetting Effectiveness: {avg_effectiveness:.2f}%")
    if avg_ppl_ratio > 0:
        print(f"Average PPL Ratio (F/T):          {avg_ppl_ratio:.2f}")
    
    print(f"\nStandard Deviations:")
    print(f"  Test MA:    ±{results_df['Test MA (%)'].std():.2f}%")
    print(f"  Forget MA:  ±{results_df['Forget MA (%)'].std():.2f}%")
    
    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎯 INTERPRETATION")
    print("="*80)
    
    if avg_forget_ma <= 10:
        status = "✅ EXCELLENT"
        emoji = "🎉"
        interpretation = "Strong forgetting achieved across all variants"
    elif avg_forget_ma <= 20:
        status = "✅ GOOD"
        emoji = "👍"
        interpretation = "Effective forgetting with good utility preservation"
    elif avg_forget_ma <= 35:
        status = "⚡ MODERATE"
        emoji = "⚡"
        interpretation = "Some forgetting achieved, may need tuning"
    else:
        status = "⚠️ INSUFFICIENT"
        emoji = "⚠️"
        interpretation = "Model retains too much knowledge of forget set"
    
    print(f"Status: {emoji} {status}")
    print(f"Interpretation: {interpretation}")
    
    # PPL Analysis (only if we have valid data)
    if avg_ppl_ratio > 0:
        print(f"\n📊 PPL Analysis:")
        if avg_ppl_ratio > 1.5:
            print(f"  ✅ Excellent: Forget PPL >> Test PPL")
            print(f"     Model is much more confused on forget set (ratio: {avg_ppl_ratio:.2f})")
        elif avg_ppl_ratio > 1.0:
            print(f"  ⚡ Good: Forget PPL > Test PPL")
            print(f"     Some confusion on forget set (ratio: {avg_ppl_ratio:.2f})")
        else:
            print(f"  ⚠️ Needs improvement: Forget PPL ≤ Test PPL")
            print(f"     Insufficient confusion on forget set (ratio: {avg_ppl_ratio:.2f})")
    
    # ========================================================================
    # CODE-SWITCHING SPECIFIC ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("🔄 CODE-SWITCHING ANALYSIS")
    print("="*80)
    
    hinglish_data = results_df[results_df['Language'] == 'HINGLISH']
    hi_data = results_df[results_df['Language'] == 'HI']
    en_data = results_df[results_df['Language'] == 'EN']
    
    if not hinglish_data.empty:
        hinglish_forget = hinglish_data['Forget MA (%)'].values[0]
        hinglish_test = hinglish_data['Test MA (%)'].values[0]
        hinglish_eff = hinglish_data['Forgetting Effectiveness (%)'].values[0]
        
        print(f"\n📍 Hinglish (Code-Mixed):")
        print(f"  Test MA (Utility):        {hinglish_test:.2f}%")
        print(f"  Forget MA (Residual):     {hinglish_forget:.2f}%")
        print(f"  Forgetting Effectiveness: {hinglish_eff:.2f}%")
        
        if hinglish_forget <= 10:
            print(f"  → ✅ Excellent hinglish forgetting!")
        elif hinglish_forget <= 20:
            print(f"  → ✅ Good hinglish forgetting!")
        else:
            print(f"  → ⚠️ May need stronger forgetting for hinglish")
    
    if not hi_data.empty and not en_data.empty and not hinglish_data.empty:
        hi_forget = hi_data['Forget MA (%)'].values[0]
        en_forget = en_data['Forget MA (%)'].values[0]
        
        print(f"\n📍 Monolingual Comparison:")
        print(f"  Hindi Forget MA:    {hi_forget:.2f}%")
        print(f"  English Forget MA:  {en_forget:.2f}%")
        print(f"  Hinglish Forget MA: {hinglish_forget:.2f}%")
        
        avg_mono = (hi_forget + en_forget) / 2
        diff = hinglish_forget - avg_mono
        
        print(f"\n  Average Monolingual: {avg_mono:.2f}%")
        print(f"  Hinglish vs Mono:    {diff:+.2f}%")
        
        if abs(diff) < 5:
            print(f"  → ✅ Consistent forgetting across all variants")
            print(f"     Code-switching MODs work equally well")
        elif diff > 5:
            print(f"  → 📊 Hinglish harder to forget (code-mixing effect)")
            print(f"     May need to increase lambda_forget_max")
        else:
            print(f"  → 📊 Hinglish easier to forget")
            print(f"     MODs may be too aggressive for code-mixed data")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    # Save clean results table
    results_csv = f"{config.output_dir}/test_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✅ Saved: {results_csv}")
    
    # Save all detailed metrics
    detailed_csv = f"{config.output_dir}/test_metrics_detailed.csv"
    pd.DataFrame([test_metrics_filtered]).to_csv(detailed_csv, index=False)
    print(f"✅ Saved: {detailed_csv}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 TESTING COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Scores:")
    print(f"  Utility (Test MA):        {avg_test_ma:.2f}%")
    print(f"  Forgetting Effectiveness: {avg_effectiveness:.2f}%")
    print(f"  Status:                   {status}")
    print("="*80)
    
    # Comparison with validation results
    print(f"\n📈 Training vs Testing:")
    print(f"  Last Validation Forget MA: 21.84% (cross-lingual)")
    print(f"  Last Validation Hinglish:  7.81%")
    print(f"  Test Average Forget MA:    {avg_forget_ma:.2f}%")
    if not hinglish_data.empty:
        print(f"  Test Hinglish Forget MA:   {hinglish_forget:.2f}%")
    print("="*80)

else:
    print("\n⚠️ No valid test results found")
    print("Available metrics:")
    for key in list(test_metrics_filtered.keys())[:10]:
        print(f"  {key}")
