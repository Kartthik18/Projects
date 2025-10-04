import numpy as np
from keras.layers import Input
from keras.models import Model
from model import build_seq2seq

def build_inference_models(model, enc_in, dec_in, state_h, state_c, dec_lstm, dec_dense, lstm_units=256):
    encoder_model = Model(enc_in, [state_h, state_c])

    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_outputs, state_h_dec, state_c_dec = dec_lstm(dec_in, initial_state=[decoder_state_input_h, decoder_state_input_c])
    decoder_outputs = dec_dense(decoder_outputs)

    decoder_model = Model([dec_in, decoder_state_input_h, decoder_state_input_c],
                          [decoder_outputs, state_h_dec, state_c_dec])
    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, fre_tokenizer, target_id_to_word, max_len):
    states_value = encoder_model.predict(input_seq)
    prev_word = np.zeros((1, 1, 1))
    prev_word[0, 0, 0] = fre_tokenizer.word_index["startofsentence"]

    decoded_sentence = []
    for _ in range(max_len):
        output_tokens, h, c = decoder_model.predict([prev_word] + states_value)
        predicted_id = np.argmax(output_tokens[0, -1, :])
        predicted_word = target_id_to_word.get(predicted_id, "")
        if predicted_word == "endofsentence":
            break
        decoded_sentence.append(predicted_word)
        prev_word[0, 0, 0] = predicted_id
        states_value = [h, c]
    return " ".join(decoded_sentence)
