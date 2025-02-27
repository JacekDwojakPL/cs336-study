import regex as re
import json

class BasicTokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.inverse_vocab = {word:idx for idx, word in self.vocab.items()}
        self.merges = {pair:idx for idx, pair in enumerate(merges)}
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.special_tokens = None
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            idx = -1 + len(self.vocab)
            for token in special_tokens:
                self.vocab[idx] = token.encode()
                self.inverse_vocab[token.encode()] = idx
                idx += 1
        self.compiled_pattern = re.compile(pattern)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}
        merges = []
        
        with open(vocab_filepath, "r") as vocab_file:
            f = json.load(vocab_file)
            vocab = {int(i):eval(token) for i, token in f.items()}
        if special_tokens:
          for token in special_tokens:
              encoded_token = token.encode()
              if encoded_token not in set(vocab.values()):
                  vocab[len(vocab)] = encoded_token
          
        with open(merges_filepath, "r") as merges_file:
            for line in merges_file:
                line = line.rstrip()
                first, second = line.split(" ")
                merges.append((eval(first), eval(second)))
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        pretokens = []
        if self.special_tokens:
            pattern = f"({'|'.join(map(re.escape, self.special_tokens))})"
            special_chunks = [chunk for chunk in re.split(pattern, text) if chunk != ""]

            for c in special_chunks:
                pretokens.extend([c] if c in self.special_tokens else self.compiled_pattern.findall(c))
        else:
            pretokens = self.compiled_pattern.findall(text)
        indices = []
        for pretoken in pretokens:
            if self.special_tokens is not None and pretoken in self.special_tokens:
                indices.append(self.inverse_vocab[pretoken.encode()])
                continue
            pretoken_tuple = list(bytes([i]) for i in pretoken.encode()) # list of splitted bytes from encoded string
         
            if len(pretoken_tuple) > 1:
                for merge_pair in self.merges.keys():
                    counter = 0
                    new_pretoken = []
                    while counter<len(pretoken_tuple):
                        if counter+1<len(pretoken_tuple) and pretoken_tuple[counter] == merge_pair[0] and pretoken_tuple[counter+1] == merge_pair[1]:
                            new_pretoken.append(merge_pair[0]+merge_pair[1])
                            counter+=2
                        else:
                            new_pretoken.append(pretoken_tuple[counter])
                            counter+=1
                    pretoken_tuple = new_pretoken
                for idx in pretoken_tuple:
                    indices.append(self.inverse_vocab[idx])
            else:
                encoded_pretoken = pretoken.encode()
                if encoded_pretoken in self.inverse_vocab.keys():
                    indices.append(self.inverse_vocab[encoded_pretoken])
                else:
                    indices += list(encoded_pretoken)
        return indices
    
    def encode_iterable(self, iterable):
        while True:
            try:
                line = next(iterable)
                indx = self.encode(line)
                for i in indx:
                    yield i
            except StopIteration:
                break
        
    def decode(self, indices):
        return b"".join(self.vocab[idx] for idx in indices).decode(errors="replace")