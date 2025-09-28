# preprocessing.py
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import math
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# ---------------------------------------------------------------------
# Constants (kept compatible with your original code)
# NOTE: You used 0 for SOS and padding; we keep it as-is to match behavior.
# ---------------------------------------------------------------------
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# ---------------------------------------------------------------------
# Vocabulary container
# ---------------------------------------------------------------------
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lowercase, trim, remove non-letters (kept same as your code)
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

# ---------------------------------------------------------------------
# Data loading (expects a local "eng-fra.txt")
# ---------------------------------------------------------------------
def readLangs(lang1, lang2, reverse=False, path='eng-fra.txt'):
    print("Reading lines...")

    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False, path='eng-fra.txt'):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, path=path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# ---------------------------------------------------------------------
# Tensor helpers (keep device-agnostic here; move to device in main)
# ---------------------------------------------------------------------
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    idxs = indexesFromSentence(lang, sentence)
    idxs.append(EOS_token)
    return torch.tensor(idxs, dtype=torch.long).view(1, -1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor

# ---------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------
def get_dataloader(batch_size, path='eng-fra.txt'):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True, path=path)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    # NOTE: keep on CPU here; we'll .to(device) inside the training loop
    train_data = TensorDataset(torch.LongTensor(input_ids),
                               torch.LongTensor(target_ids))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, pairs, train_dataloader
