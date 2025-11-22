"""
Utility functions and callbacks for Indic language unlearning
Adapted from original utils.py
"""
import pandas as pd

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class CustomCallback:
    """
    Custom callback factory for checkpointing and early stopping
    """
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.max_tolerance = args.max_tolerance
        self.method = args.method
        self.save_top_k = 3

        # Set configurations based on task
        if args.task == "flores":
            if args.method == "finetune":
                self.monitor = "val/xppl"
                self.mode = "min"
                self.filename = "xppl={val/xppl:.2f}"
            else:
                self.monitor = "val/forget_xma"
                self.mode = "min"
                self.filename = "fxma={val/forget_xma:.4f}-xppl={val/xppl:.2f}-fxppl={val/forget_xppl:.2f}"

        elif args.task == "bmlama":
            if args.method == "finetune":
                self.monitor = "val/sent_ppl"
                self.mode = "min"
                self.filename = "sent_ppl={val/sent_ppl:.2f}"
            else:
                self.monitor = "val/forget_xpa"
                self.mode = "min"
                self.filename = "fxpa={val/forget_xpa:.4f}-xppl={val/sent_xppl:.2f}-fxppl={val/forget_sent_xppl:.2f}"
        else:
            raise ValueError(f"Task {args.task} not supported.")

    def load_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self.output_dir,
            filename=self.filename,
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=self.save_top_k,
            save_last=False,
            save_weights_only=True,
            verbose=True,
            auto_insert_metric_name=False,
        )

    def load_early_stopping_callback(self):
        return EarlyStopping(
            monitor=self.monitor,
            mode=self.mode,
            patience=self.max_tolerance,
            verbose=True,
        )


class CustomMetricTracker(Callback):
    """
    Callback to track and save metrics to CSV files
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_validation_end(self, trainer, pl_module):
        # Perplexity metrics
        ppl_df = pd.DataFrame({
            k: [v.item()] for k, v in trainer.logged_metrics.items()
            if "forget" not in k and "ppl" in k and "sent" not in k
        })
        ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
        ppl_df.to_csv(f"{self.output_dir}/val_ppl.csv", index=False, mode="a")

        # Forget perplexity metrics
        forget_ppl_df = pd.DataFrame({
            k: [v.item()] for k, v in trainer.logged_metrics.items()
            if "forget" in k and "ppl" in k and "sent" not in k
        })
        forget_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
        forget_ppl_df.to_csv(f"{self.output_dir}/val_forget_ppl.csv", index=False, mode="a")

        if pl_module.hparams.task == "flores":
            # Memorization Accuracy metrics
            forget_ma_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" in k and "ma" in k
            })
            forget_ma_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            forget_ma_df.to_csv(f"{self.output_dir}/val_forget_ma.csv", index=False, mode="a")

        elif pl_module.hparams.task == "bmlama":
            # Precision Accuracy metrics
            forget_pa_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" in k and "pa" in k
            })
            forget_pa_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            forget_pa_df.to_csv(f"{self.output_dir}/val_forget_pa.csv", index=False, mode="a")

            # Sentence perplexity
            forget_sent_ppl_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" in k and "sent_ppl" in k
            })
            forget_sent_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            forget_sent_ppl_df.to_csv(f"{self.output_dir}/val_forget_sent_ppl.csv", index=False, mode="a")

            sent_ppl_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" not in k and "sent_ppl" in k
            })
            sent_ppl_df.rename(columns=lambda x: x.replace("val/", ""), inplace=True)
            sent_ppl_df.to_csv(f"{self.output_dir}/val_sent_ppl.csv", index=False, mode="a")

    def on_test_end(self, trainer, pl_module):
        # Perplexity metrics
        ppl_df = pd.DataFrame({
            k: [v.item()] for k, v in trainer.logged_metrics.items()
            if "forget" not in k and "ppl" in k and "sent" not in k
        })
        ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
        ppl_df.to_csv(f"{self.output_dir}/test_ppl.csv", index=False)

        # Forget perplexity metrics
        forget_ppl_df = pd.DataFrame({
            k: [v.item()] for k, v in trainer.logged_metrics.items()
            if "forget" in k and "ppl" in k and "sent" not in k
        })
        forget_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
        forget_ppl_df.to_csv(f"{self.output_dir}/test_forget_ppl.csv", index=False)

        if pl_module.hparams.task == "flores":
            # Memorization Accuracy metrics
            forget_ma_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" in k and "ma" in k
            })
            forget_ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_ma_df.to_csv(f"{self.output_dir}/test_forget_ma.csv", index=False)

            ma_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" not in k and "ma" in k
            })
            ma_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            ma_df.to_csv(f"{self.output_dir}/test_ma.csv", index=False)

        elif pl_module.hparams.task == "bmlama":
            # Precision Accuracy metrics
            forget_pa_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" in k and "pa" in k
            })
            forget_pa_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_pa_df.to_csv(f"{self.output_dir}/test_forget_pa.csv", index=False)

            # Sentence perplexity
            forget_sent_ppl_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" in k and "sent_ppl" in k
            })
            forget_sent_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            forget_sent_ppl_df.to_csv(f"{self.output_dir}/test_forget_sent_ppl.csv", index=False)

            pa_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" not in k and "pa" in k
            })
            pa_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            pa_df.to_csv(f"{self.output_dir}/test_pa.csv", index=False)

            sent_ppl_df = pd.DataFrame({
                k: [v.item()] for k, v in trainer.logged_metrics.items()
                if "forget" not in k and "sent_ppl" in k
            })
            sent_ppl_df.rename(columns=lambda x: x.replace("test/", ""), inplace=True)
            sent_ppl_df.to_csv(f"{self.output_dir}/test_sent_ppl.csv", index=False)


class CustomRichProgressBar(RichProgressBar):
    """
    Custom progress bar that hides version number
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items