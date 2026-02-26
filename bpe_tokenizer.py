import regex as re

import json

import os

from typing import List, Dict, Tuple, Iterable, Iterator, Optional

import time

from collections import defaultdict

import multiprocessing as mp

from typing import BinaryIO

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Pre-compute bytes for faster lookup
_BYTE_TO_BYTES = [bytes([i]) for i in range(256)]


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



def _process_chunk(args):
    """
    Worker function to process a file chunk and return word counts.
    """
    input_path, start, end, special_tokens = args

    regex_pat = re.compile(GPT2_SPLIT_PATTERN)

    # Read the chunk from file
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    # Decode the chunk
    text = chunk_bytes.decode('utf-8', errors='ignore')

    # Split by special tokens if needed
    if special_tokens:
        special_pat = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        chunks = []
        last_end = 0
        for match in special_pat.finditer(text):
            s, e = match.span()
            if s > last_end:
                chunks.append(text[last_end:s])
            last_end = e
        if last_end < len(text):
            chunks.append(text[last_end:])
    else:
        chunks = [text]

    # Count words in this chunk
    word_counts = defaultdict(int)
    for chunk in chunks:
        for match in regex_pat.finditer(chunk):
            token_bytes = match.group().encode('utf-8')
            word_tuple = tuple(_BYTE_TO_BYTES[b] for b in token_bytes)
            word_counts[word_tuple] += 1

    return dict(word_counts)


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    """

    Trains a byte-level BPE tokenizer.

    [cite: 233]

    """

    if special_tokens is None:

        special_tokens = []



    vocab = {i: _BYTE_TO_BYTES[i] for i in range(256)}


    # Check file size to decide whether to use multiprocessing
    file_size = os.path.getsize(input_path)
    use_multiprocessing = file_size > 10 * 1024 * 1024  # Use MP for files > 10MB

    regex_pat = re.compile(GPT2_SPLIT_PATTERN)

    if use_multiprocessing:
        # Use multiprocessing for large files
        num_processes = mp.cpu_count()

        # Find chunk boundaries
        with open(input_path, 'rb') as f:
            if special_tokens:
                # Use the first special token as the split token
                split_token = special_tokens[0].encode('utf-8')
                boundaries = find_chunk_boundaries(f, num_processes, split_token)
            else:
                # If no special tokens, just split uniformly
                f.seek(0, os.SEEK_END)
                file_size_bytes = f.tell()
                chunk_size = file_size_bytes // num_processes
                boundaries = [i * chunk_size for i in range(num_processes + 1)]
                boundaries[-1] = file_size_bytes

        # Create arguments for each worker
        worker_args = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        # Process chunks in parallel
        with mp.Pool(num_processes) as pool:
            chunk_results = pool.map(_process_chunk, worker_args)

        # Merge results from all chunks
        word_counts = defaultdict(int)
        for chunk_counts in chunk_results:
            for word_tuple, count in chunk_counts.items():
                word_counts[word_tuple] += count
        word_counts = dict(word_counts)
    else:
        # Use serial processing for small files (faster due to no MP overhead)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if special_tokens:
            # Split by special tokens
            special_pat = re.compile("|".join(re.escape(tok) for tok in special_tokens))
            chunks = []
            last_end = 0
            for match in special_pat.finditer(text):
                start, end = match.span()
                if start > last_end:
                    chunks.append(text[last_end:start])
                last_end = end
            if last_end < len(text):
                chunks.append(text[last_end:])
        else:
            chunks = [text]

        word_counts = defaultdict(int)
        for chunk in chunks:
            for match in regex_pat.finditer(chunk):
                token_bytes = match.group().encode('utf-8')
                word_tuple = tuple(_BYTE_TO_BYTES[b] for b in token_bytes)
                word_counts[word_tuple] += 1
        word_counts = dict(word_counts)



    merges = []



    num_merges = vocab_size - 256 - len(special_tokens)



    for i in range(num_merges):

        stats = get_stats(word_counts)

        if not stats:

            break

        # Use max with a tuple key for consistent tie-breaking
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]



        merges.append(best_pair)



        # New token ID

        new_id = 256 + i

        vocab[new_id] = best_pair[0] + best_pair[1]
        merge_vocab(word_counts, best_pair)
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