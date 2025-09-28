# model.py
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

# Reuse MAX_LENGTH from training config if desired; default to 10 for PE.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model=128, nhead=8, num_layers=5, dropout=0.1, max_len=10):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.src_embedding = nn.Embedding(input_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=max_len)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)

        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # src, tgt: (B, T)
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        # In batch_first mode, PyTorch expects key_padding_mask of shape (B, T)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)

        output = self.fc_out(output)  # (B, T, V)
        return output

# Mask helpers
def generate_square_subsequent_mask(sz, device=None):
    m = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    if device is not None:
        m = m.to(device)
    return m

def create_padding_mask(seq, pad_id=0):
    # (B, T) -> bool mask where True marks PAD tokens
    return (seq == pad_id)
