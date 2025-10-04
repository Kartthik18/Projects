import numpy as np
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from model import build_seq2seq
from preprocessing import tokenize, pad

# Load data
with open("data/small_vocab_en.txt", "r") as f:
    eng_sentences = f.read().split("\n")
with open("data/small_vocab_fr.txt", "r") as f:
    fre_sentences = f.read().split("\n")

# Tokenize + pad
eng_tokenized, eng_tokenizer = tokenize(eng_sentences)
fre_tokenized, fre_tokenizer = tokenize(fre_sentences, encode_start_end=True)
eng_encoded, fre_encoded = pad(eng_tokenized), pad(fre_tokenized)

english_vocab_size = len(eng_tokenizer.word_index) + 1
french_vocab_size = len(fre_tokenizer.word_index) + 1

# Prepare data
decoder_input = fre_encoded[:, :-1]
decoder_target = fre_encoded[:, 1:]
decoder_input = np.expand_dims(decoder_input, -1)
decoder_target = np.expand_dims(decoder_target, -1)

# Build model
model, enc_in, dec_in, state_h, state_c, dec_lstm, dec_dense = build_seq2seq(
    english_vocab_size, french_vocab_size
)

model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(0.002), metrics=["accuracy"])

# Train
model.fit([eng_encoded, decoder_input], decoder_target, batch_size=1024, epochs=16, validation_split=0.2)

# Save model
model.save("seq2seq_lstm.h5")
