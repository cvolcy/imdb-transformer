"""Transformer Block

  Typical usage example:

  tb = TransformerBlock(embed_dim, num_heads, fc_dim, rate)
  x = tb(inputs)
"""
from .multi_head_self_attention import MultiHeadSelfAttention
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Layer, LayerNormalization

class TransformerBlock(Layer):
    """TransformerBlock

    Attributes:
        embed_dim: embedding dimension, should be divisible by
                   the number of heads.
        num_heads: number of heads.
        fc_dim: Fully connected layer dimension.
    """
    def __init__(self, embed_dim, num_heads, fc_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = Sequential(
            [Dense(fc_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)
