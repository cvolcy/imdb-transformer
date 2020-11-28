"""TokenAndPositionEmbedding
"""

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

class TokenAndPositionEmbedding(Layer):
    """TokenAndPositionEmbedding

    Attributes:
        maxlen: Max number of words in each inputs.
        vocab_size: Max number of words in vocabulary.
        embed_dim: embedding dimension, should be divisible by
                   the number of heads.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        return x + positions
