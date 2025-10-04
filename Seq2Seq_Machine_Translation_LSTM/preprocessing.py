import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def tokenize(x, encode_start_end=False):
    if encode_start_end:
        x = ["startofsentence " + sentence + " endofsentence" for sentence in x]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    tokenized_x = tokenizer.texts_to_sequences(x)
    return tokenized_x, tokenizer

def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')
