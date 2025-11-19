import os.path as osp
import random

import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class FLORESIndicDataModule(L.LightningDataModule):
    """
    DataModule for FLORES-200 Indic languages
    """
    # Major Indic languages with good representation
    SUPPORTED_LANGUAGES = [
        "hi",  # Hindi
        "bn",  # Bengali
        "te",  # Telugu
        "ta",  # Tamil
        "mr",  # Marathi
        "gu",  # Gujarati
        "kn",  # Kannada
        "ml",  # Malayalam
        "pa",  # Punjabi
        "ur",  # Urdu
    ]

    # Extended set with more Indic languages
    EXTENDED_LANGUAGES = [
        "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "ur",
        "or",  # Odia
        "as",  # Assamese
        "ne",  # Nepali
        "si",  # Sinhala
        "sd",  # Sindhi
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
            local_files_only=args.offline,
        )

        # Use extended language set if specified
        if hasattr(args, 'use_extended_indic') and args.use_extended_indic:
            self.available_languages = self.EXTENDED_LANGUAGES
        else:
            self.available_languages = self.SUPPORTED_LANGUAGES

        self.flores_valid = []
        self.flores_test = []

    def setup(self, stage=None):
        if stage == "fit":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            retain_data = load_json_dataset(self.args,
                                            f"retain-{self.args.forget_num}-x{self.args.retain_multiplier}.jsonl")
            self.flores_forget = FLORESIndicDataset(forget_data, self.tokenizer, self.args.max_seq_len,
                                                    lang=self.args.forget_lang)
            self.flores_retain = FLORESIndicDataset(retain_data, self.tokenizer, self.args.max_seq_len,
                                                    lang=self.args.retain_lang)

            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = FLORESIndicDataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESIndicDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_valid.append(valid_dataset)
                self.flores_valid.append(forget_dataset)

        if stage == "validate":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = FLORESIndicDataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESIndicDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_valid.append(valid_dataset)
                self.flores_valid.append(forget_dataset)

        if stage == "test":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            test_data = load_json_dataset(self.args, "test.jsonl")
            # Test different languages
            langs = self.args.forget_lang if self.args.test_src_lang_only else self.available_languages
            for lang in langs:
                test_dataset = FLORESIndicDataset(test_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESIndicDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
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

        return DataLoader(dataset,
                          batch_size=self.args.per_device_train_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        dataloaders = []
        for dataset in self.flores_valid:
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.per_device_eval_batch_size,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self):
        dataloaders = []
        for dataset in self.flores_test:
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.per_device_eval_batch_size,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
            dataloaders.append(dataloader)
        return dataloaders


class FLORESIndicDataset(Dataset):
    """
    Dataset for FLORES-200 Indic languages
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
            "language": lang,  # Added for tracking
        }


def load_json_dataset(args, file_path):
    """Load JSONL dataset"""
    return load_dataset(
        "json",
        data_files=osp.join(args.data_dir, file_path),
        cache_dir=args.cache_dir,
    )["train"]


if __name__ == "__main__":
    """
    Test the datamodule with sample data
    """
    from argparse import Namespace

    # Test configuration
    args = Namespace(
        data_dir="../data/flores_indic",
        model_name_or_path="bigscience/bloom-560m",
        cache_dir="../../../.cache",
        offline=True,
        forget_num=100,
        retain_multiplier=10,
        forget_lang=["hi", "bn"],
        retain_lang=["hi", "bn", "te", "ta"],
        max_seq_len=256,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_workers=0,
        alternate_loader_every_n_epoch=False,
        test_src_lang_only=False,
        use_extended_indic=False,
    )

    print("Testing FLORESIndicDataModule...")
    print(f"Forget languages: {args.forget_lang}")
    print(f"Retain languages: {args.retain_lang}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )

    # Test loading a sample file
    try:
        data = load_dataset("json", data_files="../data/flores_indic/valid.jsonl")["train"]
        print(f"\n✓ Loaded {len(data)} validation samples")

        # Check token lengths for each language
        INDIC_LANGS = ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "ur"]

        print("\nToken length statistics per language:")
        print("-" * 60)

        for lang in INDIC_LANGS:
            lengths = []
            for item in data:
                if lang in item:
                    text = item[lang]
                    tokens = tokenizer.encode(text)
                    lengths.append(len(tokens))

            if lengths:
                print(f"{lang:3s} | Max: {max(lengths):3d} | Min: {min(lengths):2d} | "
                      f"Mean: {sum(lengths) / len(lengths):5.1f} | Samples: {len(lengths)}")

        print("-" * 60)

        # Test dataset creation
        dataset = FLORESIndicDataset(data, tokenizer, max_seq_len=256, lang=["hi", "bn"])
        print(f"\n✓ Created dataset with {len(dataset)} samples")

        # Test getting a sample
        sample = dataset[0]
        print(f"\n✓ Sample batch keys: {sample.keys()}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Language: {sample['language']}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Make sure to run create_indic_flores_forget_retain.py first!")