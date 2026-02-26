
import time
import os
import regex as re
from collections import defaultdict
from bpe_tokenizer import (
    train_bpe,
    get_stats,
    merge_vocab,
    GPT2_SPLIT_PATTERN,
    _BYTE_TO_BYTES,
    find_chunk_boundaries,
    _process_chunk
)
import multiprocessing as mp

if __name__ == '__main__':
    INPUT_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10000
    SPECIALS = ["<|endoftext|>"]

    print("=== Manual Profiling of BPE Training ===\n")

    # Overall timing
    overall_start = time.time()

    # Initialize vocab
    vocab = {i: _BYTE_TO_BYTES[i] for i in range(256)}

    # Check file size
    file_size = os.path.getsize(INPUT_FILE)
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")
    use_multiprocessing = file_size > 10 * 1024 * 1024

    # Time: Pre-tokenization
    pretok_start = time.time()

    regex_pat = re.compile(GPT2_SPLIT_PATTERN)
    num_processes = mp.cpu_count()

    # Find chunk boundaries
    boundary_start = time.time()
    with open(INPUT_FILE, 'rb') as f:
        split_token = SPECIALS[0].encode('utf-8')
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    boundary_end = time.time()

    print(f"Finding chunk boundaries: {boundary_end - boundary_start:.2f} seconds")

    # Create worker args
    worker_args = [
        (INPUT_FILE, start, end, SPECIALS)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Process chunks in parallel
    parallel_start = time.time()
    with mp.Pool(num_processes) as pool:
        chunk_results = pool.map(_process_chunk, worker_args)
    parallel_end = time.time()

    print(f"Parallel pre-tokenization: {parallel_end - parallel_start:.2f} seconds")

    # Merge results
    merge_start = time.time()
    word_counts = defaultdict(int)
    for chunk_counts in chunk_results:
        for word_tuple, count in chunk_counts.items():
            word_counts[word_tuple] += count
    word_counts = dict(word_counts)
    merge_end = time.time()

    print(f"Merging chunk results: {merge_end - merge_start:.2f} seconds")

    pretok_end = time.time()
    print(f"Total pre-tokenization: {pretok_end - pretok_start:.2f} seconds")
    print(f"Unique pre-tokens: {len(word_counts)}\n")

    # Time: BPE merge iterations
    merges = []
    num_merges = VOCAB_SIZE - 256 - len(SPECIALS)

    print(f"Starting {num_merges} BPE merges...")
    bpe_start = time.time()

    # Time first 10, middle 10, and last 10 merges separately
    get_stats_time = 0
    merge_vocab_time = 0

    for i in range(num_merges):
        # Get stats
        stats_start = time.time()
        stats = get_stats(word_counts)
        stats_end = time.time()
        get_stats_time += (stats_end - stats_start)

        if not stats:
            break

        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        # Merge vocab
        merge_start = time.time()
        new_id = 256 + i
        vocab[new_id] = best_pair[0] + best_pair[1]
        merge_vocab(word_counts, best_pair)
        merge_end = time.time()
        merge_vocab_time += (merge_end - merge_start)

        # Print progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - bpe_start
            print(f"  Completed {i + 1}/{num_merges} merges in {elapsed:.2f} seconds")

    bpe_end = time.time()

    print(f"\nBPE merge iterations: {bpe_end - bpe_start:.2f} seconds")
    print(f"  - get_stats: {get_stats_time:.2f} seconds ({get_stats_time / (bpe_end - bpe_start) * 100:.1f}%)")
    print(f"  - merge_vocab: {merge_vocab_time:.2f} seconds ({merge_vocab_time / (bpe_end - bpe_start) * 100:.1f}%)")

    # Add special tokens
    next_id = 256 + len(merges)
    for st in SPECIALS:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1

    overall_end = time.time()

    print(f"\n=== Summary ===")
    print(f"Total time: {(overall_end - overall_start) / 60:.2f} minutes")
    print(f"Pre-tokenization: {pretok_end - pretok_start:.2f} seconds ({(pretok_end - pretok_start) / (overall_end - overall_start) * 100:.1f}%)")
    print(f"BPE merges: {bpe_end - bpe_start:.2f} seconds ({(bpe_end - bpe_start) / (overall_end - overall_start) * 100:.1f}%)")
