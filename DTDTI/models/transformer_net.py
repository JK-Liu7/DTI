import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

if torch.cuda.is_available():
    device = torch.device('cuda')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask=None)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, n_layers, in_dim, embed_size, heads, forward_expansion, dropout):
        super(Encoder, self).__init__()

        self.in_dim = in_dim
        self.embed_size = embed_size
        self.embedding = nn.Linear(in_dim, embed_size)
        self.position_embedding = PositionalEncoding(d_model=100, dropout=0.0, max_len=1000)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        x = self.embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, x, x, mask=None)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, src_mask=None, trg_mask=None):
        attention = self.attention(trg, trg, trg, trg_mask)
        query = self.dropout(self.norm(attention + trg))
        out = self.transformer_block(src, src, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, n_layers=10, in_dim=1, embed_size=128, heads=4, forward_expansion=4, dropout=0.2):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Linear(in_dim, embed_size)
        self.position_embedding = PositionalEncoding(d_model=100, dropout=0.2, max_len=1000)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout)
             for _ in range(n_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, src_mask=None, trg_mask=None):
        # N, seq_length = trg.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        trg = self.dropout(self.word_embedding(trg))

        for layer in self.layers:
            trg = layer(trg, src, src)

        trg = torch.squeeze(trg, dim=0)

        return trg


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size=100,
            trg_vocab_size=100,
            src_pad_idx=0,
            trg_pad_idx=0,
            embed_size=128,
            num_layers=10,
            forward_expansion=4,
            heads=4,
            dropout=0.2,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
