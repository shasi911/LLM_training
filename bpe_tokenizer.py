import regex as re

import json

import os

from typing import List, Dict, Tuple, Iterable, Iterator, Optional

import time

from collections import defaultdict

import heapq

import multiprocessing as mp

from typing import BinaryIO

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Pre-compute bytes for faster lookup
_BYTE_TO_BYTES = [bytes([i]) for i in range(256)]


class _RevPair:
    """
    Heap entries are (-count, _RevPair(pair)).  Reversing the pair comparison
    means heapq (a min-heap) will pop the LARGEST pair when counts tie,
    matching the tie-breaking of max(items, key=lambda x: (x[1], x[0])).
    """
    __slots__ = ("pair",)

    def __init__(self, pair: Tuple):
        self.pair = pair

    def __lt__(self, other): return self.pair > other.pair
    def __le__(self, other): return self.pair >= other.pair
    def __gt__(self, other): return self.pair < other.pair
    def __ge__(self, other): return self.pair <= other.pair
    def __eq__(self, other): return self.pair == other.pair


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



class Tokenizer:

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):

       

        self.vocab = {int(k): v for k, v in vocab.items()}

        self.merges = merges

        self.special_tokens = special_tokens if special_tokens else []

       



        self.vocab_inv = {v: k for k, v in self.vocab.items()}

        self.merges_ranks = {pair: i for i, pair in enumerate(merges)}

       

        # Cache for pre-tokenization regex to avoid recompiling

        self.pat = re.compile(GPT2_SPLIT_PATTERN)

       

        # Cache for special tokens regex

        if self.special_tokens:

            # Escape special tokens to ensure they are treated as literal strings

            self.special_pat = re.compile("|".join(re.escape(tok) for tok in self.special_tokens))



    @classmethod

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):

        """

        Construct and return a Tokenizer from serialized vocabulary and merges.

        [cite: 301]

        """

        # Load vocab: expect a JSON mapping int_str -> bytes_list_ints or raw latin1 string

        with open(vocab_filepath, 'r', encoding='utf-8') as f:

            raw_vocab = json.load(f)

       



        vocab = {}

        for k, v in raw_vocab.items():

            vocab[int(k)] = v.encode('latin-1') if isinstance(v, str) else bytes(v)



        # Load merges: expect a JSON list of (bytes_str, bytes_str)

        with open(merges_filepath, 'r', encoding='utf-8') as f:

            raw_merges = json.load(f)

       

        merges = []

        for p1, p2 in raw_merges:

            b1 = p1.encode('latin-1')

            b2 = p2.encode('latin-1')

            merges.append((b1, b2))



        return cls(vocab, merges, special_tokens)



    def encode(self, text: str) -> List[int]:

        """

        Encode an input text into a sequence of token IDs.

        [cite: 303]

        """

        if not text:

            return []



        if self.special_tokens:

            parts = []

            last_end = 0

            for match in self.special_pat.finditer(text):

                start, end = match.span()

                if start > last_end:

                    parts.append((text[last_end:start], False)) # False = not a special token

                parts.append((match.group(), True)) # True = is a special token

                last_end = end

            if last_end < len(text):

                parts.append((text[last_end:], False))

        else:

            parts = [(text, False)]



        encoded_ids = []



        # 2. Process chunks

        for part, is_special in parts:

            if is_special:

                if part.encode("utf-8") in self.vocab_inv:

                    encoded_ids.append(self.vocab_inv[part.encode("utf-8")])

                else:

                    # Fallback or error; ideally special tokens are in vocab

                    # For safety, treat as normal text if not found (though this shouldn't happen)

                    encoded_ids.extend(self._encode_chunk(part))

            else:

                encoded_ids.extend(self._encode_chunk(part))

       

        return encoded_ids



    def _encode_chunk(self, text_chunk: str) -> List[int]:

        """

        Helper to encode a string segment that contains no special tokens.

        Applies GPT-2 pre-tokenization and then BPE merges.

        [cite: 275, 277]

        """

        ids = []

        for match in self.pat.finditer(text_chunk):

            token_bytes = match.group().encode('utf-8')



            word = [self.vocab_inv[_BYTE_TO_BYTES[b]] for b in token_bytes]



            while len(word) >= 2:

                # Find the best pair to merge according to learned ranks

                stats = {}

                for i in range(len(word) - 1):

                    pair = (self.vocab[word[i]], self.vocab[word[i+1]])

                    if pair in self.merges_ranks:

                        stats[pair] = self.merges_ranks[pair]



                if not stats:

                    break



                best_pair = min(stats, key=stats.get)

               

                # Apply the merge

                new_word = []

                i = 0

                while i < len(word):

                    if i < len(word) - 1 and (self.vocab[word[i]], self.vocab[word[i+1]]) == best_pair:

                       

                        # Find the ID for the merged bytes

                        combined_bytes = self.vocab[word[i]] + self.vocab[word[i+1]]

                        new_id = self.vocab_inv[combined_bytes]

                        new_word.append(new_id)

                        i += 2

                    else:

                        new_word.append(word[i])

                        i += 1

                word = new_word



            ids.extend(word)

        return ids



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:



        for text in iterable:

            yield from self.encode(text)



    def decode(self, ids: List[int]) -> str:



        byte_stream = b""

        for idx in ids:

            if idx in self.vocab:

                byte_stream += self.vocab[idx]

            else:

                pass

       

        return byte_stream.decode('utf-8', errors='replace')





# --- Training Logic ---



def get_stats(vocab_counts: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:

    """

    Counts frequency of adjacent pairs in the weighted vocabulary.

    [cite: 197]

    """

    pairs = defaultdict(int)

    for word_tuple, count in vocab_counts.items():
        for i in range(len(word_tuple) - 1):

            pairs[(word_tuple[i], word_tuple[i+1])] += count

    return pairs



def merge_vocab(vocab_counts: Dict[Tuple[bytes, ...], int], pair: Tuple[bytes, bytes]) -> None:

    """

    Applies the merge of `pair` to all words in the dictionary, updating in place.

    [cite: 199]

    """

    first, second = pair

    merged_bytes = first + second

    # Collect changes to apply after iteration
    to_delete = []
    to_add = defaultdict(int)

    for word_tuple, count in vocab_counts.items():
        # Quick check if this word could possibly contain the pair
        if first not in word_tuple:
            continue

        # Build new word with merges applied
        new_word = []
        i = 0
        word_len = len(word_tuple)
        changed = False

        while i < word_len:

            if i < word_len - 1 and word_tuple[i] == first and word_tuple[i+1] == second:

                new_word.append(merged_bytes)

                i += 2
                changed = True

            else:

                new_word.append(word_tuple[i])

                i += 1

        if changed:
            to_delete.append(word_tuple)
            to_add[tuple(new_word)] += count

    # Apply changes
    for word in to_delete:
        del vocab_counts[word]
    for word, count in to_add.items():
        vocab_counts[word] = vocab_counts.get(word, 0) + count


def _merge_and_update_stats(
    word_counts: Dict[Tuple[bytes, ...], int],
    pair_counts: Dict[Tuple[bytes, bytes], int],
    pair: Tuple[bytes, bytes],
    heap: list = None,
) -> None:
    """
    Apply merge of `pair` into word_counts and update pair_counts incrementally.

    Instead of recomputing all pair counts from scratch each BPE iteration
    (which is O(total_tokens) per step), only touches the words that actually
    contain the merged pair -- O(words_with_pair * avg_word_length).

    If `heap` is provided (a max-heap via negated counts), pushes updated
    pair counts onto it so the caller can do O(log n) best-pair lookups.
    """
    first, second = pair
    merged_bytes = first + second

    to_delete = []
    to_add: Dict[Tuple[bytes, ...], int] = defaultdict(int)
    changed_pairs = set()

    for word_tuple, count in word_counts.items():
        if first not in word_tuple:
            continue

        # Check whether the specific pair (first, second) appears adjacently
        word_len = len(word_tuple)
        found = False
        for i in range(word_len - 1):
            if word_tuple[i] == first and word_tuple[i + 1] == second:
                found = True
                break
        if not found:
            continue

        # Subtract old pair contributions for this word
        for j in range(word_len - 1):
            p = (word_tuple[j], word_tuple[j + 1])
            pair_counts[p] -= count
            changed_pairs.add(p)

        # Build the merged word
        new_word: List[bytes] = []
        i = 0
        while i < word_len:
            if i < word_len - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                new_word.append(merged_bytes)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_word_tuple = tuple(new_word)

        # Add new pair contributions for the merged word
        for j in range(len(new_word_tuple) - 1):
            p = (new_word_tuple[j], new_word_tuple[j + 1])
            pair_counts[p] += count
            changed_pairs.add(p)

        to_delete.append(word_tuple)
        to_add[new_word_tuple] += count

    for word in to_delete:
        del word_counts[word]
    for word, count in to_add.items():
        word_counts[word] = word_counts.get(word, 0) + count

    # The merged pair no longer exists in the corpus
    pair_counts.pop(pair, None)
    changed_pairs.discard(pair)

    # Push updated counts onto the heap (lazy -- stale entries are skipped on pop)
    if heap is not None:
        for p in changed_pairs:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, _RevPair(p)))



def _process_chunk(args):
    """
    Process one byte-range of the file and return word counts.

    Loads the entire chunk into memory as a single string so that the GPT-2
    regex sees multi-line whitespace sequences (e.g. "\n\n") as a single token,
    matching the behaviour of a reference implementation that reads the whole
    corpus.  With parallel processing each worker holds only 1/N_CPU of the
    file, so peak RAM stays at ~file_size regardless of CPU count.
    """
    input_path, start, end, special_tokens = args

    regex_pat = re.compile(GPT2_SPLIT_PATTERN)
    special_pat = (
        re.compile("|".join(re.escape(tok) for tok in special_tokens))
        if special_tokens else None
    )

    with open(input_path, 'rb') as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode('utf-8', errors='ignore')
    del raw  # free the bytes buffer as soon as we have the string

    # Split on special tokens (discarding them), then run the regex on each part
    parts = special_pat.split(text) if special_pat else [text]

    word_counts: Dict[Tuple[bytes, ...], int] = defaultdict(int)
    for part in parts:
        for match in regex_pat.finditer(part):
            token_bytes = match.group().encode('utf-8')
            word_tuple = tuple(_BYTE_TO_BYTES[b] for b in token_bytes)
            word_counts[word_tuple] += 1

    return dict(word_counts)


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE tokenizer.

    Phase 1 -- word counting: splits the file into one chunk per CPU and
    processes all chunks IN PARALLEL, using as much RAM as available instead of
    running one chunk at a time.  Results are merged into a single word_counts
    dict, then the chunk dicts are freed.

    Phase 2 -- BPE merges: uses a MAX-HEAP (lazy deletion) so each
    best-pair lookup is O(log |pairs|) instead of O(|pairs|).  For 32 k
    merges over millions of unique pairs this is a ~50 000x speedup vs the
    plain max() scan.
    """
    if special_tokens is None:
        special_tokens = []

    vocab = {i: _BYTE_TO_BYTES[i] for i in range(256)}

    # --- Phase 1: parallel chunk processing ---
    num_workers = mp.cpu_count()
    split_token = special_tokens[0].encode('utf-8') if special_tokens else b'\n'

    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # All chunks run simultaneously -- RAM goes up proportionally to num_workers,
    # but wall-clock time for this phase drops by ~num_workers.
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_process_chunk, chunk_args)

    word_counts: Dict[Tuple[bytes, ...], int] = defaultdict(int)
    for chunk_counts in results:
        for word_tuple, count in chunk_counts.items():
            word_counts[word_tuple] += count
    del results  # free all chunk dicts now that they're merged

    # --- Phase 2: BPE merge loop with heap ---
    pair_counts = get_stats(word_counts)

    # Build initial max-heap: entries are (-count, _RevPair(pair)).
    # heapq is a min-heap, so negating count makes it act as a max-heap.
    # _RevPair reverses pair comparison so that ties are broken by the
    # lexicographically LARGEST pair, matching max(items, key=(count, pair)).
    heap: list = [(-count, _RevPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)

    for i in range(num_merges):
        if not pair_counts:
            break

        # Pop until we find a heap entry whose count matches pair_counts
        # (lazy deletion: stale entries from previous updates are simply skipped)
        best_pair = None
        while heap:
            neg_count, rev = heapq.heappop(heap)
            candidate = rev.pair
            if pair_counts.get(candidate, 0) == -neg_count and -neg_count > 0:
                best_pair = candidate
                break

        if best_pair is None:
            break

        merges.append(best_pair)
        vocab[256 + i] = best_pair[0] + best_pair[1]

        # Merge in word_counts, update pair_counts, and push changed pairs
        # onto the heap so future lookups stay correct.
        _merge_and_update_stats(word_counts, pair_counts, best_pair, heap=heap)

    next_id = 256 + len(merges)
    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1

    return vocab, merges

if __name__ == "__main__":
    import psutil

    INPUT_FILE = "data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10000
    SPECIALS = ["<|endoftext|>"]

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

        start_time = time.time()
        vocab, merges = train_bpe(INPUT_FILE, VOCAB_SIZE, SPECIALS)
        end_time = time.time()

        # Get peak memory usage
        peak_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

        # Save results
        # We encode bytes to 'latin-1' strings so JSON can handle them
        serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
        serializable_merges = [[p1.decode('latin-1'), p2.decode('latin-1')] for p1, p2 in merges]

        with open("vocab.json", "w") as f:
            json.dump(serializable_vocab, f)
        with open("merges.json", "w") as f:
            json.dump(serializable_merges, f)

        longest_token = max(vocab.values(), key=len).decode('utf-8', errors='ignore')
        training_time_minutes = (end_time - start_time) / 60
        training_time_hours = training_time_minutes / 60

        print(f"\n--- Training Results ---")
        print(f"Time: {training_time_minutes:.2f} minutes ({training_time_hours:.2f} hours)")
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")
        print(f"Vocab Size: {len(vocab)}")
        print(f"Longest Token: '{longest_token}'")
        print(f"Longest Token Length: {len(longest_token)} characters")
        print(f"Files saved: vocab.json, merges.json")