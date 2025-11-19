#!/usr/bin/env python3
"""
Main training script for Indic language unlearning
Simplified version that works with flores_indic_datamodule.py
Includes epoch logging hooks and some robustness improvements.
"""
import os
import os.path as osp
import glob
import random
from argparse import ArgumentParser

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from flores_indic_datamodule import FLORESIndicDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def safe_load_tokenizer(name_or_path, cache_dir=None, local_files_only=False):
    """
    Try to load fast tokenizer, but if conversion fails due to missing
    converters (tiktoken/protobuf/sentencepiece), fall back to use_fast=False.
    """
    try:
        return AutoTokenizer.from_pretrained(
            name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except Exception as e:
        print("Warning: Fast tokenizer load failed; falling back to slow (use_fast=False).")
        print("Error:", e)
        return AutoTokenizer.from_pretrained(
            name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=False,
        )


class MultilingualModel(L.LightningModule):
    """
    PyTorch Lightning wrapper for multilingual language models
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # Load model
        print(f"Loading model: {args.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )

        # Load tokenizer (safe loader)
        self.tokenizer = safe_load_tokenizer(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
            local_files_only=args.offline,
        )

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Log current dataset type (forget or retain) if alternating loader behavior is enabled
        if getattr(self.args, "alternate_loader_every_n_epoch", 0):
            current_epoch = int(self.current_epoch)
            if current_epoch % (self.args.forget_multiplier + 1) == self.args.forget_multiplier:
                dataset_type = "RETAIN"
            else:
                dataset_type = "FORGET"
            self.log("dataset_type", 1 if dataset_type == "RETAIN" else 0, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        perplexity = torch.exp(loss)
        self.validation_step_outputs.append({
            "loss": loss,
            "perplexity": perplexity,
            "dataloader_idx": dataloader_idx,
        })
        self.log(f"val_loss_{dataloader_idx}", loss, prog_bar=True, add_dataloader_idx=False, sync_dist=True)
        self.log(f"val_ppl_{dataloader_idx}", perplexity, prog_bar=False, add_dataloader_idx=False, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        dataloader_losses = {}
        for output in self.validation_step_outputs:
            idx = output["dataloader_idx"]
            dataloader_losses.setdefault(idx, []).append(output["loss"])
        for idx, losses in dataloader_losses.items():
            avg_loss = torch.stack(losses).mean()
            self.log(f"val_loss_avg_{idx}", avg_loss, sync_dist=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        perplexity = torch.exp(loss)
        self.test_step_outputs.append({
            "loss": loss,
            "perplexity": perplexity,
            "dataloader_idx": dataloader_idx,
        })
        self.log(f"test_loss_{dataloader_idx}", loss, add_dataloader_idx=False, sync_dist=True)
        self.log(f"test_ppl_{dataloader_idx}", perplexity, add_dataloader_idx=False, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        dataloader_losses = {}
        dataloader_ppls = {}
        for output in self.test_step_outputs:
            idx = output["dataloader_idx"]
            dataloader_losses.setdefault(idx, []).append(output["loss"])
            dataloader_ppls.setdefault(idx, []).append(output["perplexity"])
        print("\n" + "=" * 70)
        print("TEST RESULTS:")
        print("=" * 70)
        for idx in sorted(dataloader_losses.keys()):
            avg_loss = torch.stack(dataloader_losses[idx]).mean()
            avg_ppl = torch.stack(dataloader_ppls[idx]).mean()
            dataset_type = "FORGET" if idx % 2 == 1 else "VALID"
            lang_idx = idx // 2
            print(f"Dataloader {idx} ({dataset_type}, Lang {lang_idx}): Loss={avg_loss:.4f}, Perplexity={avg_ppl:.4f}")
            self.log(f"test_loss_avg_{idx}", avg_loss, sync_dist=True)
            self.log(f"test_ppl_avg_{idx}", avg_ppl, sync_dist=True)
        print("=" * 70 + "\n")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Estimate steps: be conservative
        steps_per_epoch = max(1, getattr(self.trainer, "estimated_stepping_batches", 100) // max(1, self.args.epochs))
        total_steps = max(1, steps_per_epoch * max(1, self.args.epochs))
        warmup_steps = int(total_steps * self.args.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    #######################################################################
    # EPOCH LOGGING HOOKS
    #######################################################################
    def on_train_epoch_start(self):
        # +1 to display human-friendly epoch counting
        epoch = int(self.current_epoch) + 1
        print("\n" + "=" * 70)
        print(f"🚀  STARTING EPOCH {epoch}")
        print("=" * 70 + "\n")

    def on_train_epoch_end(self):
        epoch = int(self.current_epoch) + 1
        print("\n" + "=" * 70)
        print(f"✅  FINISHED EPOCH {epoch}")
        print("=" * 70 + "\n")


def detect_device_and_precision(args):
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        device_name = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        device_name = "mps"
    else:
        accelerator = "cpu"
        devices = "auto"
        device_name = "cpu"

    # Precision selection
    if args.fp16 and device_name == "cuda":
        precision = "16-mixed"
    elif args.bf16 and device_name in ("cuda", "mps"):
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    return accelerator, devices, precision, device_name


def main(args):
    L.seed_everything(args.seed, workers=True)

    # Ensure WANDB env var available if passed
    os.environ["WANDB_MODE"] = getattr(args, "wandb_mode", os.environ.get("WANDB_MODE", "online"))

    logger = CSVLogger(save_dir=args.output_dir, name="logs")

    print("\n" + "=" * 70)
    print("INDIC LANGUAGE UNLEARNING TRAINING")
    print("=" * 70)
    print(f"Model: {args.model_name_or_path}")
    print(f"Task: {args.task}")
    print(f"Method: {args.method}")
    print(f"Forget languages: {args.forget_lang}")
    print(f"Retain languages: {args.retain_lang}")
    print(f"Forget samples: {args.forget_num}")
    print(f"Retain samples: {args.forget_num * args.retain_multiplier}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {args.train_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # Load datamodule
    print("Loading datamodule...")
    dm = FLORESIndicDataModule(args)

    # Load or create model
    print("Loading model...")
    if args.finetuned_model_path and osp.exists(args.finetuned_model_path):
        print(f"Loading from checkpoint: {args.finetuned_model_path}")
        model = MultilingualModel.load_from_checkpoint(
            checkpoint_path=args.finetuned_model_path,
            hparams=args,
        )
    else:
        model = MultilingualModel(args)

    # Setup callbacks (use RichProgressBar if available)
    callbacks = []
    try:
        from lightning.pytorch.callbacks.progress import RichProgressBar
        callbacks.append(RichProgressBar())
    except Exception:
        print("Warning: RichProgressBar not available. Install 'rich' to enable it: pip install -U \"rich>=10.2.2\"")

    if not args.disable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename="epoch={epoch:02d}-val_loss={val_loss_0:.4f}",
            monitor="val_loss_0",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss_0",
            patience=args.max_tolerance,
            mode="min",
            verbose=True,
        )
        callbacks.extend([checkpoint_callback, early_stop_callback])

    accelerator, devices, precision, device_name = detect_device_and_precision(args)
    print(f"Using accelerator={accelerator}, devices={devices}, precision={precision}")

    trainer = L.Trainer(
        default_root_dir=args.output_dir,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=args.logging_steps,
        val_check_interval=args.evaluation_steps,
        num_sanity_val_steps=0,
        deterministic=getattr(args, "deterministic", False),
        logger=logger,
        reload_dataloaders_every_n_epochs=args.alternate_loader_every_n_epoch,
        enable_checkpointing=not args.disable_checkpointing,
        callbacks=callbacks,
    )

    # Train
    if args.do_train:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70 + "\n")
        trainer.fit(model, dm)
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70 + "\n")

    # Evaluation / Test
    if args.do_eval or args.do_test:
        print("\n" + "=" * 70)
        print("RUNNING EVALUATION")
        print("=" * 70 + "\n")

        # Try to find best checkpoint
        if args.ckpt_path:
            ckpt_path = osp.join(args.output_dir, args.ckpt_path)
        else:
            ckpt_files = glob.glob(osp.join(args.output_dir, "epoch=*.ckpt"))
            ckpt_path = sorted(ckpt_files)[0] if ckpt_files else None

        if ckpt_path and osp.exists(ckpt_path):
            print(f"Loading best checkpoint: {ckpt_path}")
            model = MultilingualModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams=args)
        else:
            print("No checkpoint found, using current model...")

        if args.do_eval:
            trainer.validate(model, dm)
        if args.do_test:
            trainer.test(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser(description="Indic Language Unlearning Training")

    # Model arguments
    parser.add_argument("--model_type", type=str, default="xglm-564M")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/xglm-564M")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--method", type=str, default="lingtea")
    parser.add_argument("--finetuned_model_path", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--task", type=str, default="flores")
    parser.add_argument("--forget_lang", type=str, nargs="+", default=["hi", "bn"])
    parser.add_argument("--retain_lang", type=str, nargs="+", default=["hi", "bn", "te", "ta"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--forget_num", type=int, default=100)
    parser.add_argument("--forget_multiplier", type=int, default=10)
    parser.add_argument("--retain_multiplier", type=int, default=10)
    parser.add_argument("--alternate_loader_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_src_lang_only", action="store_true")
    parser.add_argument("--use_extended_indic", action="store_true")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default=".checkpoints/")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--evaluation_steps", type=float, default=1.0)
    parser.add_argument("--max_tolerance", type=int, default=5)

    # System arguments
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B logging mode (online/offline/disabled)",
    )

    # Action arguments
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    args = parser.parse_args()

    # Calculate effective batch size
    args.train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    # Create output directory structure (same logic as your original script)
    if args.method == "original":
        args.output_dir = f".checkpoints/{args.model_type}/{args.task}/{args.method}"
    else:
        args.output_dir = (
            f".checkpoints/{args.model_type}/{args.task}/{args.method}/"
            f"F{args.forget_num}_R{args.retain_multiplier}/"
            f"BS{args.train_batch_size}_LR{args.learning_rate}_W{args.warmup_ratio}_S{args.seed}"
        )

    # If training and checkpoints exist, confirm with user
    if args.do_train and not args.disable_checkpointing:
        existing_ckpts = glob.glob(osp.join(args.output_dir, "*.ckpt"))
        if existing_ckpts:
            response = input(f"\n⚠️  Checkpoints found in {args.output_dir}\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                exit(0)

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
