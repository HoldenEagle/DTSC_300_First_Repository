from typing import Any

import keras
import tensorflow as tf


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int):
        """Token and position embedding allows us to convert a tokenized
        vocabulary into an embedding vector. And, it allows us to take
        into account the exact position of the token as well

        Args:
            vocab_size (int): the size of the vocabulary (number of tokens)
            maxlen (int): the maximum length of the input (in tokens)
            embed_dim (int): the embedding dimensions
        """
        super().__init__()
        self.token_emb = keras.layers.Embedding(vocab_size, embed_dim)

        # Position embedding can be done more thoughtfully, but here it
        # is done by learning an arbitrary vector
        self.pos_emb = keras.layers.Embedding(maxlen, embed_dim)

    def call(self, x: Any) -> Any:
        """The forward pass through the network. It combines position
        embedding and the vocab -> vector embedding. You can think of
        vocab -> vector as being fasttext (though as yet untrained).

        Args:
            x (Any): the previous layer

        Returns:
            Any: the next layer
        """
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
