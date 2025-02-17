---
layout: post
title:  "Assignment 1 - Byte Pair Encoder Training - IO considerations"
date:   2025-02-16 20:00:00 +0100
categories: assignment1
author: jacek
---
So first part of first assignment asks the student to implement and train BPE Tokenizer. I will not describe how BPE algorithm works since there are much better sources - especially Karpathy's [Let's build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) video. Example of BPE in the first lecture is based on his approach. However main outline of training such tokenizer is as follows:

- Initialize starting vocabulary of length 256 - since one byte can have 256 values in range 0-255. Initial vocabulary will index each value.
```python
def initialize_vocab():
    # bytes(x) function converts an integer into raw byte string which can be later decoded to unicode
    vocab = {x:bytes[(x)] for x in range(256)}
    return vocab
```
- Preprocess raw text data, for example use regex pattern to separate individual strings
- Convert unicode characters codepoints in string into sequences of bytes - indices in vocabulary
```python
# encode() method converts string into byte string
indices = list(string.encode())
```

Written part of the assignment asks question about finding bug in function which converts individual bytes into chars. During iteration over raw bytes an error can occur since unicode characters are often composed of series of bytes and only whore series can be converted into string.

- Then for given number of merges which roughly equals to the target vocabulary size:
    - Iterate over whole dataset and find most frequent pair of bytes.
    - Merge those two bytes togheter and assign them a new index in the vocabulary
    ```python
    # most_frequent_pair is a tuple of two bytes
    vocab[new_index] = vocab[most_frequent_pair[0]]+vocab[most_frequent_pair[1]]
    ```
    - `vocab[new_index]` points to the newly merged pair (which is a byte string) so when tokenizer will encounter two bytes of frequent pairs from training during tokenization, instead of returning two indices, tokenizer will return one index. With this method the length of tokenized string can be reduced (compressed) to fewer indices, and this has crucial impact on Transformer architecture and attention mechanisms - since this architecture has fixed context size which can he handled.
    - Iterate again over indices and replace each byte from most frequent pair with new index.

From the above we can see that each merge makes two passes over entire dataset which maybe is tolerable for small datasets but the assignment requires students to train on 2GB and 12GB text files. With target vocabulary size of 10000 using approach described above we would train for three months on the TinyStories-GPT4-train.txt file. For reference, here are constraints given by instructors:

- TinyStoriesDataset - 30 minutes training time and <= 30GB memory
- OpenWebText - 12 hours training time and <= 100GB memory
- The unittest given to check correctness of the code on some small excerpt has timeout of 1.5 seconds after which the test fails. So the solution has 1.5 second window of training time.

Assignment notes provides some hints about optimizations to speedup training time. 

### Fitting dataset in memory
But we cannot improve the alghoritm itself wihtout the training data. And first big factor to consider is the size of dataset and fact that it will not fit in the memory at once. When I was researching the subject of BPE I always found examples with whole dataset in the memory. The Karpaty's example is no exception. So first main thing is to find some way to read those large files and not fill the entire RAM.

One thing that came to mind is to use some sort of paralleliation and spread reading across multiple processes. Rughly speaking - divide the dataset into chunks of given size, compute offests (start and end positions) of each chunk, distribute the chunks offsets across multiple threads so that each thread has it's own start position and perform the pre-tokenization on loaded chunk. 
But this approach can be problematic because of fact mentioned above - the unicode characters are often composed from sequences of bytes and we do not have guarantee that chunking will not cut some word in the middle. This will affect computation of most frequent pairs and merges.

Another try was to have some overlap when calculating chunk offsets so the pairs will remain unchanged and not cut. Probably my implementation was buggy because when I was comparing word counts computed by parallel method with those computed when dataset was read all at once (the ground truth) I got different results. Also the unittest provided with the assignment was failing.

So after several versions of methods to efficiently read big files I ended up with somewhat hybrid method. It uses multithreading but not for file reading, but for counting the words statistics. Basically there is producer process and several consumer processes. The producer reads file in lines. This prevents loading whole file in memory at once. Producer then put each line on queue from which it is taken by consumers for further processing. 
```python
def producer():
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_queue.put(line)
        for _ in range(num_threads):
            line_queue.put(stop_signal)

def consumer(output_queue):
    while True:
        line = line_queue.get()
        if line is stop_signal:
            output_queue.put(stop_signal)
            break
        counter.update(process_chunk(line))
        line_queue.task_done()
```
In my training function there is one producer process and 4 consumer processes so it is some kind of parallelism. Obviously one could provide better and more efficient solution, but since this method improved IO time enough to meet the 1.5 seconds margin in the unittest, I decided to leave it as is and move forward. 

### Keypoints from this segment:
- We need some methods to read large files which will not fit in the memory 
- IO process have very string impact on training time
- Parallelization can improve reading time but it has many edge cases to be aware of

