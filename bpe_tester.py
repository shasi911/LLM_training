# Assuming your train_bpe function is in a file named bpe.py
from bpe_tokenizer import train_bpe
import time
import json

start_time = time.time()
vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])
end_time = time.time()

# Find longest token
longest_token = max(vocab.values(), key=len).decode('utf-8', errors='ignore')
print(f"Training took {(end_time - start_time)/60:.2f} minutes.")
print(f"Longest token: {longest_token}")