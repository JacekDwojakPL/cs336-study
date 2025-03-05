---
layout: post
title: "Assignment 1 - Byte Pair Encoder Training - IO considerations"
date: 2025-02-16 20:00:00 +0100
categories: assignment1
author: jacek
---

First assignment asks the student to implement and train BPE Tokenizer. Tokenization is process of transforming characters or words into indices, which then are processed further by Language Model. This is needed due to the fact that LLMs ultimately works on numbers and can't process raw text directly. There some way of converting raw strings to numbers is nedded and this process is called tokenization. There are three main approaches to the tokenization:

- Character level tokenization - each character in vocabulary/alphabet is given it's own index
- Word-level tokenization - with this approach each unique word in the corpus gets it's own index
- Sub-word tokenization - it is kind of hybrid approach in which neither single characters nor whole words are mapped to an index

Each approach has it's pros and cons. Main thing to consider is the is length of sequences produced by tokenization. This is crucial factor due to the fact that the Transformer-based LLMs operate on fixed-size context window. Another factor is the vocabulary size which has direct influence on the size of Embedding matrix. Here are examples for illustration.

#### Character level tokenization

This method maps single characters in sequence to their own index. At the first look it seems resonable approach but there some details to consider. First of all - which alphabet/language to use? How to choose numbers for indices? There are defined standards like ASCII which are code mappings of latin characters and digits to 127 numbers (one-byte integers) used to represent individual characters in computer memory. 95 of 127 numbers are reserved for printable characters (a-zA-Z0-9 and puncuation and special characters). This means that when choosing ASCII as our base encoding we will end up with vocabulary size of 95. Which means that embedding matrix in language model will have 95 rows (one row for one index). In python we can use `ord()` function to get code assiciated with given character. For example letter 'a' is mapped to number 97:

```python
ord('a') #prints 97
```

Then process of tokenization is done by interation over each character in given string with `ord()` function call:

```python
def encode(string):
    return [ord(c) for c in string]

encode("Hello World!") # returns array of integers [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33]
```

Vocabulary size of 95 is very compact but this approach drastically limits abilites of our language model. We have no way to represent any character from outside ASCII - this includes any non-latin characters (Chinese, Cyrylic etc) as well as emojis. To mitigate those issues there is another standard - Unicode - which solves the problem of non-latin characters representation. Like ASCII the Unicode standard maps character to the code-point but it uses from 1 to 4 bytes instead of just 1 byte. That way we have a lot of possibilites (with four bytes (32 bits) we can represent 2^32 == 4,294,967,295 things) and Unicode standard uses 128,237 codes in total. Like with ASCII we can use `ord()` function in python to get code point corresponding for given character:

```python
ord('婓') # prints 23123
```

And like with ASCII we could simply iterate over string with `ord()` function call:

```python
# Good afternoon in Japanese
encode("こんにちは") # returns [12371, 12435, 12395, 12385, 12399]
```

But if we choose to use Unicode code points directly in our tokenization, the vocabulary size quickly grows from 95 to 128,237. This results in an enormous embedding matrix. To illustrate this problem: if we choose a fairly modest embedding dimension of 768, meaning that every element from our vocabulary will be converted to a vector of 768 elements, our embedding matrix will have 128,237 x 768 elements (parameters). The total size of this matrix will be 393.94 MB (each element in matrix is an 4-byte number). To summarize character-level tokenization:

- The tokenized sequence has the same length as the source string sequence.
- When using ASCII encoding, the vocabulary size is small (95 code points), but we lose the ability to represent non-Latin characters.
- By using Unicode code points directly, we gain the ability to represent non-Latin characters, but this results in an enormous vocabulary size.

#### Word level tokenization

Instead of tokenize every character separately we can use tokenization based on whole words.

#### Sub-word Level tokenization

This is a hybrid approach that tokenizes not single characters or whole words, but fragments of words.

### Byte-pair encoding

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
