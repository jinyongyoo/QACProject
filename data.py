import os
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

class Dictionary(object):
    def __init__(self, paths):
        self.char2idx = {}
        self.idx2char = []
        self.max_seq_len = 0
        self.build_dictionary(paths)

    def build_dictionary(self, paths):
        self.add_char("<pad>")
        for path in paths:
            assert os.path.exists(path)
            with open(path, "r", encoding="utf8") as f:
                for line in f:
                    chars = list(line)
                    chars.append("<eos>")
                    self.max_seq_len = max(len(chars), self.max_seq_len)
                    for char in chars:
                        self.add_char(char.lower())

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1

    def get_idx(self, char):
        return self.char2idx[char]

    def get_char(self, idx):
        return idx2char[idx]

    def __len__(self):
        return len(self.idx2char)

class QueryDataset(IterableDataset):
    def __init__(self, path, dictionary):
        self.path = path
        self.dictionary = dictionary
        self.len = 0
        with open(self.path, "r") as f:
            for line in f:
                self.len+=1

    def __len__(self):
        return self.len

    def prepare_data(self, text):
        chars = list(text)
        chars.append("<eos>")
        text_length = len(chars)
        pad_tokens_to_add = self.dictionary.max_seq_len - text_length + 1
        chars += ["<pad>"] * pad_tokens_to_add
        ids = [self.dictionary.get_idx(c.lower()) for c in chars]

        input_tensor = nn.functional.one_hot(torch.LongTensor(ids[:-1]), num_classes=len(self.dictionary)).float()
        target_tensor = torch.LongTensor(ids[1:])

        return input_tensor, target_tensor, text_length

    def __iter__(self):
        file_itr = open(self.path)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return map(self.prepare_data, file_itr)
        else:
            jump = self.len // worker_info.num_workers * worker_info.id
            for i in range(jump):
                next(file_itr)
            
            return map(self.prepare_data, file_itr)
