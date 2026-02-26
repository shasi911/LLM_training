#!/usr/bin/env python3
"""
Train BPE tokenizer on OpenWebText dataset.

Usage:
    python train_bpe_owt.py

Outputs:
    - owt_vocab.json: Vocabulary mapping token IDs to byte sequences
    - owt_merges.json: List of merge operations
"""

import json
import time
import os
from bpe_tokenizer import train_bpe

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, memory tracking disabled")


def main():
    INPUT_FILE = "data/owt_train.txt"
    VOCAB_SIZE = 10000
    SPECIALS = ["<|endoftext|>"]

    # Output files
    VOCAB_OUTPUT = "owt_vocab.json"
    MERGES_OUTPUT = "owt_merges.json"

    print("=" * 60)
    print("Training BPE Tokenizer on OpenWebText")
    print("=" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Special tokens: {SPECIALS}")
    print()

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("Please ensure the OpenWebText training data is available.")
        return 1

    # Check file size
    file_size_gb = os.path.getsize(INPUT_FILE) / (1024 ** 3)
    print(f"Training data size: {file_size_gb:.2f} GB")
    print()

    # Get initial memory usage
    if HAS_PSUTIL:
        process = psutil.Process()
        initial_memory_gb = process.memory_info().rss / (1024 ** 3)
        print(f"Initial memory usage: {initial_memory_gb:.2f} GB")

    print("Starting training...")
    print("-" * 60)

    start_time = time.time()
    vocab, merges = train_bpe(INPUT_FILE, VOCAB_SIZE, SPECIALS)
    end_time = time.time()

    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60
    training_time_hours = training_time_minutes / 60

    # Get peak memory usage
    if HAS_PSUTIL:
        peak_memory_gb = process.memory_info().rss / (1024 ** 3)

    print("-" * 60)
    print("Training complete!")
    print()

    # Save results
    print("Serializing vocabulary and merges...")
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    serializable_merges = [[p1.decode('latin-1'), p2.decode('latin-1')] for p1, p2 in merges]

    with open(VOCAB_OUTPUT, "w", encoding='utf-8') as f:
        json.dump(serializable_vocab, f)

    with open(MERGES_OUTPUT, "w", encoding='utf-8') as f:
        json.dump(serializable_merges, f)

    # Get longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_str = longest_token.decode('utf-8', errors='replace')

    # Print summary
    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Training time: {training_time_minutes:.2f} minutes ({training_time_hours:.2f} hours)")

    if HAS_PSUTIL:
        print(f"Peak memory usage: {peak_memory_gb:.2f} GB")
        print(f"Memory increase: {peak_memory_gb - initial_memory_gb:.2f} GB")

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token: '{longest_token_str}'")
    print(f"Longest token length: {len(longest_token)} bytes ({len(longest_token_str)} characters)")
    print()
    print(f"Output files:")
    print(f"  - {VOCAB_OUTPUT}")
    print(f"  - {MERGES_OUTPUT}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
