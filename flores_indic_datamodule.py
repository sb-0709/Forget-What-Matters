"""
Datamodules for Indic language unlearning
Adapted for FLORES-200 Indic languages
"""
import os.path as osp
import random

import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


# Indic languages for FLORES task
FLORES_LANGUAGES = ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "ur"]

# Extended Indic languages (if needed)
FLORES_LANGUAGES_EXTENDED = [
    "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "ur",
    "or", "as", "ne", "si", "sa"
]

# BMLAMA languages (keeping for compatibility)
BMLAMA_LANGUAGES_17 = ["en", "fr", "es", "ar", "zh", "vi", "ca"]
BMLAMA_LANGUAGES_53 = ["en", "fr", "es", "pt", "ar", "vi", "ca", "hi", "bn"]


class FLORESDataModule(L.LightningDataModule):
    """
    DataModule for FLORES-200 Indic languages
    """
    SUPPORTED_LANGUAGES = FLORES_LANGUAGES

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
            local_files_only=args.offline if hasattr(args, 'offline') else False,
            use_fast=False,  # ADD THIS LINE
        )
        self.flores_valid = []
        self.flores_test = []

    def setup(self, stage=None):
        if stage == "fit":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            retain_data = load_json_dataset(self.args, f"retain-{self.args.forget_num}-x{self.args.retain_multiplier}.jsonl")
            self.flores_forget = FLORESDataset(
                forget_data, self.tokenizer, self.args.max_seq_len, lang=self.args.forget_lang
            )
            self.flores_retain = FLORESDataset(
                retain_data, self.tokenizer, self.args.max_seq_len, lang=self.args.retain_lang
            )

            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = FLORESDataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_valid.append(valid_dataset)
                self.flores_valid.append(forget_dataset)

        if stage == "validate":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = FLORESDataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_valid.append(valid_dataset)
                self.flores_valid.append(forget_dataset)

        if stage == "test":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            test_data = load_json_dataset(self.args, "test.jsonl")
            # Test different languages
            langs = self.args.forget_lang if self.args.test_src_lang_only else self.SUPPORTED_LANGUAGES
            for lang in langs:
                test_dataset = FLORESDataset(test_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_test.append(test_dataset)
                self.flores_test.append(forget_dataset)

    def train_dataloader(self):
        if self.args.alternate_loader_every_n_epoch:
            if self.trainer.current_epoch % (self.args.forget_multiplier + 1) == self.args.forget_multiplier:
                dataset = self.flores_retain
            else:
                dataset = self.flores_forget
        else:
            dataset = self.flores_retain

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        dataloaders = []
        for dataset in self.flores_valid:
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                pin_memory=True
            )
            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self):
        dataloaders = []
        for dataset in self.flores_test:
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
                pin_memory=True
            )
            dataloaders.append(dataloader)
        return dataloaders


class FLORESDataset(Dataset):
    """
    Dataset for FLORES-200 sentences
    """
    def __init__(self, data, tokenizer, max_seq_len=256, lang=["hi"]):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lang = lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select language (random if multiple)
        if isinstance(self.lang, list):
            if len(self.lang) > 1:
                lang = random.choice(self.lang)
            else:
                lang = self.lang[0]
        else:
            lang = self.lang

        # Get text for selected language
        item = self.data[idx][lang]

        # Tokenize
        inputs = self.tokenizer(
            item,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Create labels (mask padding tokens)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


# Placeholder for BMLAMA (not used for Indic, but keeping for compatibility)
class BMLAMADataModule(L.LightningDataModule):
    """Placeholder for BMLAMA - not used in Indic setup"""
    SUPPORTED_LANGUAGES_17 = BMLAMA_LANGUAGES_17
    SUPPORTED_LANGUAGES_53 = BMLAMA_LANGUAGES_53

    def __init__(self, args):
        super().__init__()
        raise NotImplementedError("BMLAMA not implemented for Indic setup. Use FLORES instead.")


def load_json_dataset(args, file_path):
    """Load JSONL dataset from data directory"""
    return load_dataset(
        "json",
        data_files=osp.join(args.data_dir, file_path),
        cache_dir=args.cache_dir,
    )["train"]


if __name__ == "__main__":
    """Test the datamodule"""
    from argparse import Namespace

    args = Namespace(
        data_dir="data",
        model_name_or_path="facebook/xglm-564M",
        cache_dir=".cache",
        offline=False,
        forget_num=100,
        retain_multiplier=10,
        forget_lang=["hi", "bn"],
        retain_lang=["hi", "bn", "te", "ta"],
        max_seq_len=256,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_workers=0,
        alternate_loader_every_n_epoch=1,
        forget_multiplier=10,
        test_src_lang_only=False,
    )

    print("Testing FLORESDataModule...")
    print(f"Supported languages: {FLORES_LANGUAGES}")

    # Test loading
    try:
        data = load_json_dataset(args, "valid.jsonl")
        print(f"✓ Loaded {len(data)} validation samples")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )

        dataset = FLORESDataset(data, tokenizer, max_seq_len=256, lang=["hi", "bn"])
        print(f"✓ Created dataset with {len(dataset)} samples")

        sample = dataset[0]
        print(f"✓ Sample keys: {sample.keys()}")
        print(f"✓ Input shape: {sample['input_ids'].shape}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure to run create_indic_flores_from_local.py first!")