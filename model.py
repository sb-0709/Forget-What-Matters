"""
MultilingualModel for Indic language unlearning
Adapted from original model.py for FLORES-200 Indic languages
"""
import os.path as osp
import copy

import lightning as L
import torch
import torch.nn.functional as F
from pytorch_lightning.core.saving import save_hparams_to_yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import logging
from torchmetrics import Accuracy
from flores_indic_datamodule import FLORES_LANGUAGES, BMLAMA_LANGUAGES_17, BMLAMA_LANGUAGES_53

logging.get_logger("transformers").setLevel(logging.ERROR)


class MultilingualModel(L.LightningModule):
    def __init__(self, hparams):
        super(MultilingualModel, self).__init__()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=hparams.cache_dir if hparams.cache_dir else None,
            local_files_only=hparams.offline if hasattr(hparams, 'offline') else False,
            use_fast=False,  # Use slow tokenizer to avoid tiktoken issues
        )

        # Load model
        model_kwargs = {
            "cache_dir": hparams.cache_dir if hparams.cache_dir else None,
            "local_files_only": hparams.offline if hasattr(hparams, 'offline') else False,
        }
        # Only add flash attention if supported and requested
        if hasattr(hparams, 'use_flash_attention') and hparams.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                pass  # Skip if not supported

        self.model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            **model_kwargs
        )

        # Set languages based on task
        if hparams.task == "flores":
            languages = hparams.forget_lang if hparams.test_src_lang_only else FLORES_LANGUAGES
        elif hparams.task == "bmlama":
            languages = hparams.forget_lang if hparams.test_src_lang_only else \
                        BMLAMA_LANGUAGES_17 if hparams.use_mini_bmlama else BMLAMA_LANGUAGES_53
        else:
            raise ValueError(f"Task {hparams.task} not supported.")

        # Load teacher model for Knowledge Distillation
        self.teacher = copy.deepcopy(self.model)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set dataset names for validation and test
        self.valid_dataset_names = []
        self.test_dataset_names = []
        for lang in hparams.forget_lang:
            self.valid_dataset_names.append(f"val/{lang}_")
            self.valid_dataset_names.append(f"val/{lang}_forget_")
        for lang in languages:
            self.test_dataset_names.append(f"test/{lang}_")
            self.test_dataset_names.append(f"test/{lang}_forget_")

        # For Memorization Accuracy (MA)
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=-100
        )

        self.save_hyperparameters(hparams)
        if hasattr(hparams, 'do_train') and hparams.do_train:
            save_hparams_to_yaml(osp.join(hparams.output_dir, "hparams.yaml"), hparams)

    def forward(self, **inputs):
        return self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
        )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        _dict = {"train/loss": loss}

        # Knowledge distillation setup
        batch_size = batch["input_ids"].size(0)
        logit_s = outputs.logits
        padding_mask = batch["labels"].eq(-100)

        self.teacher.eval()
        with torch.no_grad():
            outputs_t = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            logit_t = outputs_t.logits

        loss_kd = F.kl_div(
            F.log_softmax(logit_s / self.hparams.temperature, dim=-1),
            F.softmax(logit_t / self.hparams.temperature, dim=-1),
            reduction="none",
        ) * (self.hparams.temperature ** 2)

        # Change loss function based on current epoch (retain vs forget)
        if self.current_epoch % (self.hparams.forget_multiplier + 1) == self.hparams.forget_multiplier:
            # RETAIN epoch: Use knowledge distillation + cross entropy
            if "xglm" in self.hparams.model_type:
                shift_logit_s = logit_s
                shift_labels = batch["labels"].new_zeros(batch["labels"].shape)
                shift_labels[:, :-1] = batch["labels"][:, 1:].clone()
                shift_labels[:, -1] = self.tokenizer.pad_token_id
            elif "bloom" in self.hparams.model_type:
                shift_logit_s = logit_s[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
            else:
                # Default behavior for other models
                shift_logit_s = logit_s[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()

            labels = torch.clamp(batch["labels"], min=0)
            prob_t = F.softmax(logit_t, dim=-1)
            prob_t = prob_t.gather(dim=-1, index=labels.unsqueeze(-1))
            prob_t.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
            shift_prob_t = prob_t[..., 1:, :] if "bloom" in self.hparams.model_type else prob_t

            loss_kd = (loss_kd * prob_t * ~padding_mask.unsqueeze(-1)).sum() / batch_size
            loss_ce = F.cross_entropy(
                shift_logit_s.view(-1, shift_logit_s.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss_ce = (loss_ce * (1 - shift_prob_t).view(-1)).sum() / batch["attention_mask"].sum()
            loss = loss_kd + loss_ce
            _dict = {"train/loss": loss, "train/kd_loss": loss_kd, "train/ce_loss": loss_ce}
        else:
            # FORGET epoch: Negative loss (gradient ascent)
            loss = loss * -1
            _dict = {"train/forget_loss": loss}

        self.log_dict(
            _dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.valid_dataset_names[dataloader_idx]

        if self.hparams.task == "flores":
            ppl = torch.exp(loss)
            ma = self._validation_ma(batch)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}loss": loss,
                f"{dataset_name}ma": ma,
            }
        elif self.hparams.task == "bmlama":
            ppl = torch.exp(loss)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}loss": loss,
            }
        else:
            raise ValueError(f"Task {self.hparams.task} not supported.")

        self.log_dict(
            _dict,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.test_dataset_names[dataloader_idx]

        if self.hparams.task == "flores":
            ppl = torch.exp(loss)
            ma = self._validation_ma(batch)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}ma": ma,
            }
        elif self.hparams.task == "bmlama":
            ppl = torch.exp(loss)
            _dict = {
                f"{dataset_name}ppl": ppl,
            }
        else:
            raise ValueError(f"Task {self.hparams.task} not supported.")

        self.log_dict(
            _dict,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def _validation_ma(self, batch):
        """Calculate Memorization Accuracy (MA)"""
        labels, preds = [], []

        # Change the sliding direction based on the padding side
        if self.tokenizer.padding_side == "left":
            start, end, step = self.hparams.max_seq_len - 1, 0, -1
        else:
            start, end, step = 1, self.hparams.max_seq_len, 1

        for i in range(start, end, step):
            label = batch["labels"][..., i]
            prompt = batch["input_ids"][..., :i]
            att_mask = batch["attention_mask"][..., :i]

            # Break if only padding tokens are left for all sequences
            if all(label == -100):
                break

            try:
                pred = self.model.generate(
                    input_ids=prompt,
                    attention_mask=att_mask,
                    max_length=i + 1
                )[..., -1]
            except IndexError:  # if batch == 1
                pred = self.model.generate(
                    input_ids=torch.squeeze(prompt),
                    attention_mask=torch.squeeze(att_mask),
                    max_length=i + 1
                ).squeeze()[-1]

            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.stack(preds, dim=-1)
        labels = torch.stack(labels, dim=-1)
        return self.accuracy(preds, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate
        )

        # Learning rate scheduler
        if self.hparams.lr_scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise ValueError(f"Invalid lr_scheduler_type: {self.hparams.lr_scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def on_validation_epoch_end(self):
        """Aggregate validation metrics"""
        # Perplexity metrics
        ppl = {k: v for k, v in self.trainer.logged_metrics.items()
               if "ppl" in k and "forget" not in k and "x" not in k and "sent" not in k}
        if ppl:
            xppl = torch.stack([ppl[k] for k in ppl.keys()]).mean().item()
        else:
            xppl = 0.0

        forget_ppl = {k: v for k, v in self.trainer.logged_metrics.items()
                     if "ppl" in k and "forget" in k and "x" not in k and "sent" not in k}
        if forget_ppl:
            forget_xppl = torch.stack([forget_ppl[k] for k in forget_ppl.keys()]).mean().item()
        else:
            forget_xppl = 0.0

        self.log_dict({"val/xppl": xppl, "val/forget_xppl": forget_xppl}, on_epoch=True, sync_dist=True)

        if self.hparams.task == "flores":
            # Memorization Accuracy metrics
            forget_ma = {k: v for k, v in self.trainer.logged_metrics.items()
                        if "ma" in k and "forget" in k and "x" not in k}
            if forget_ma:
                forget_xma = torch.stack([forget_ma[k] for k in forget_ma.keys()]).mean().item()
                self.log_dict({"val/forget_xma": forget_xma}, on_epoch=True, sync_dist=True)