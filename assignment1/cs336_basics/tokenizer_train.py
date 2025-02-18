import regex as re
from collections import Counter
from tqdm import tqdm
import threading
import queue

GPT_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
compiled_pattern = re.compile(GPT_SPLIT_PATTERN)

def initialize_vocab(special_tokens=[]):
    vocab = {}
    for index, token in enumerate(special_tokens):
        vocab[index] = token.encode()
    for index in range(256):
        vocab[index+len(special_tokens)] = bytes([index])
    
    return vocab

def split_pretoken_to_pairs(pretoken, count):
    for i in range(len(pretoken)-1):
        for _ in range(count):
            yield (pretoken[i], pretoken[i+1])

def get_most_frequent_pair(pair, count):
    return count, pair

def merge_pretokens(pretoken, pair_to_merge):
    out = []
    counter = 0

    while counter<len(pretoken):
        if counter+1<len(pretoken) and pair_to_merge[0] == pretoken[counter] and pair_to_merge[1] == pretoken[counter+1]:
            out.append(pair_to_merge[0]+pair_to_merge[1])
            counter +=2
        else:
            out.append(pretoken[counter])
            counter +=1
    return tuple(out), pretoken

def process_chunk(line):
    return [tuple(pretoken) for pretoken in compiled_pattern.findall(line)]

def read_file_multithreaded(file_path, special_tokens=[], num_threads=4, queue_size=100):
    line_queue = queue.Queue(maxsize=queue_size)
    output_queue = queue.Queue()
    stop_signal = object()
    counter = Counter()

    def producer():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_queue.put(line)
            for _ in range(num_threads):
                line_queue.put(stop_signal)
    
    def consumer(output_queue, special_tokens=[]):
        while True:
            line = line_queue.get()
            if line is stop_signal:
                output_queue.put(stop_signal)
                break
            for token in special_tokens:
                line = line.replace(token, "")
            counter.update(process_chunk(line))
            line_queue.task_done()
    
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    worker_threads = []
    
    for _ in range(num_threads):
        thread = threading.Thread(target=consumer, args=(output_queue,special_tokens), daemon=True)
        worker_threads.append(thread)
        thread.start()
    
    producer_thread.join()

    for thread in worker_threads:
        thread.join()
    
    while True:
        if output_queue.get() is stop_signal:
            return counter

def train(input_path, vocab_size, special_tokens=[]):
    vocab = initialize_vocab(special_tokens)
    merges = []
    new_index = len(vocab)
    num_merges = vocab_size - len(vocab)
    pretokens_counts = read_file_multithreaded(input_path=input_path, num_threads=32, queue_size=100, special_tokens=special_tokens)
    pretokens_pair_counts = Counter(pair for pretoken, count in pretokens_counts.items() for pair in split_pretoken_to_pairs(pretoken, count))

    for _ in tqdm(num_merges):
        if not len(pretokens_pair_counts):
            break
        most_frequent_pair = max(pretokens_pair_counts.items(), key=lambda item: get_most_frequent_pair(item[0], item[1]))
        merged_pretokens = list(merge_pretokens(pretoken, most_frequent_pair[0]) for pretoken in filter(lambda pretoken: set(most_frequent_pair[0]).issubset(pretoken), pretokens_counts.keys()))
        vocab[new_index] = b"".join([c.encode() for c in most_frequent_pair[0]])
        merges.append((b"".join([c.encode() for c in most_frequent_pair[0][0]]), b"".join([c.encode() for c in most_frequent_pair[0][1]])))
        new_index += 1

        for new_key, old_key in merged_pretokens:
            pretokens_counts[new_key] = pretokens_counts.pop(old_key)

            old_pairs = [(old_key[i], old_key[i+1]) for i in range(len(old_key)-1)]
            new_pairs = [(new_key[i], new_key[i+1]) for i in range(len(new_key)-1)]

            for pair in old_pairs:
                pretokens_pair_counts[pair] -= pretokens_counts[new_key]
            for pair in new_pairs:
                pretokens_pair_counts[pair] += pretokens_counts[new_key]
    return vocab, merges