#!/usr/bin/env python3
"""Fine-tune transformer with domain tokens for better domain-specific detection.

This script takes a pretrained prompt injection model (e.g., ProtectAI/deberta-v3)
and fine-tunes it with domain tokens prepended, so the model learns to associate
domain tokens with domain-specific attack patterns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


DEFAULT_DOMAIN_TOKENS: Dict[str, str] = {
    "healthcare": "[DOMAIN_HEALTHCARE]",
    "finance": "[DOMAIN_FINANCE]",
    "legal": "[DOMAIN_LEGAL]",
    "retail": "[DOMAIN_RETAIL]",
}


class DomainConditionedDataset(Dataset):
    """Dataset that prepends domain tokens to prompts."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        domain_token: str,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.domain_token = domain_token
        self.max_length = max_length
        self.examples = []

        # Load JSONL dataset
        with jsonl_path.open("r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    query = data.get("query", "")
                    label = data.get("label", "benign")
                    # Convert label to binary: attacked=1, benign=0
                    is_attack = 1 if label == "attacked" else 0
                    self.examples.append({"query": query, "label": is_attack})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # Prepend domain token
        text = f"{self.domain_token} {example['query']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def fine_tune_with_domain_tokens(
    base_model: str,
    train_jsonl: Path,
    domain: str,
    domain_token: str,
    output_dir: Path,
    val_jsonl: Path | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> None:
    """Fine-tune a prompt injection model with domain tokens.

    Args:
        base_model: HuggingFace model name (e.g., "ProtectAI/deberta-v3-base-prompt-injection-v2")
        train_jsonl: Path to training JSONL (query, label fields)
        domain: Domain name (e.g., "healthcare")
        domain_token: Domain token to prepend (e.g., "[DOMAIN_HEALTHCARE]")
        output_dir: Where to save fine-tuned model
        val_jsonl: Optional validation JSONL
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    print("=" * 80)
    print(f"FINE-TUNING {base_model} FOR DOMAIN: {domain}")
    print("=" * 80)

    # Load tokenizer and model
    print("\nLoading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)

    # Add domain token to vocabulary
    print(f"\nAdding domain token: {domain_token}")
    num_added = tokenizer.add_tokens([domain_token])
    if num_added > 0:
        print(f"  Added {num_added} new token(s)")
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Model embeddings resized to {len(tokenizer)}")
    else:
        print(f"  Token already exists in vocabulary")

    # Create datasets
    print(f"\nLoading training data from {train_jsonl}...")
    train_dataset = DomainConditionedDataset(
        train_jsonl, tokenizer, domain_token
    )
    print(f"  Loaded {len(train_dataset)} training examples")

    val_dataset = None
    if val_jsonl and val_jsonl.exists():
        print(f"\nLoading validation data from {val_jsonl}...")
        val_dataset = DomainConditionedDataset(
            val_jsonl, tokenizer, domain_token
        )
        print(f"  Loaded {len(val_dataset)} validation examples")

    # Training arguments
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if val_dataset else "no",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        use_cpu=not torch.cuda.is_available(),  # Explicit CPU flag
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    print(f"\n\nSaving final model to {final_path}...")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {final_path}")
    print(f"\nTo use the fine-tuned model:")
    print(f"  export RAGWALL_TRANSFORMER_MODEL={final_path}")
    print(f"  export RAGWALL_TRANSFORMER_DOMAIN={domain}")
    print("=" * 80)


def create_train_val_split(
    input_jsonl: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Split a JSONL dataset into train/val sets."""
    import random

    random.seed(seed)

    # Load all examples
    examples = []
    with input_jsonl.open("r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Shuffle
    random.shuffle(examples)

    # Split
    val_size = int(len(examples) * val_ratio)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    # Write
    with train_output.open("w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with val_output.open("w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Split {len(examples)} examples:")
    print(f"  Training: {len(train_examples)} ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"  Validation: {len(val_examples)} ({len(val_examples)/len(examples)*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune transformer with domain tokens"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to JSONL dataset (will auto-split train/val if no --val-data)",
    )
    parser.add_argument(
        "--domain",
        default="healthcare",
        help="Domain name (default: healthcare)",
    )
    parser.add_argument(
        "--domain-token",
        default=None,
        help="Domain token (default: [DOMAIN_{DOMAIN}])",
    )
    parser.add_argument(
        "--base-model",
        default="ProtectAI/deberta-v3-base-prompt-injection-v2",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/domain_finetuned"),
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Validation JSONL (if not provided, will split from dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio if auto-splitting (default: 0.2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )

    args = parser.parse_args()

    # Determine domain token
    domain_token = args.domain_token or f"[DOMAIN_{args.domain.upper()}]"

    # Handle train/val split
    train_jsonl = args.dataset
    val_jsonl = args.val_data

    if val_jsonl is None:
        # Auto-split
        print("No validation data provided, creating train/val split...")
        train_jsonl = args.output_dir / "train.jsonl"
        val_jsonl = args.output_dir / "val.jsonl"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        create_train_val_split(
            args.dataset, train_jsonl, val_jsonl, args.val_ratio
        )
        print()

    # Fine-tune
    fine_tune_with_domain_tokens(
        base_model=args.base_model,
        train_jsonl=train_jsonl,
        domain=args.domain,
        domain_token=domain_token,
        output_dir=args.output_dir,
        val_jsonl=val_jsonl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
