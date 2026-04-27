from typing import Any

import keras


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        """Create a new TransformerEncoderBlock layer. This is split
        into the definition (__init__) and the mechanism for calling
        it (call).

        Args:
            embed_dim (int): number of embedding
            num_heads (int): number of parallel heads
            ff_dim (int): the dimensions of the feedforward layers
            dropout (float, optional): fraction of neurons to turn off
                during each training run. Defaults to 0.1.
        """
        super().__init__()
        # MultiHeadAttention is the attention matrix of the transformer
        # block
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        # This sequential is the following "multilayer perceptron" or
        # fully connected component of the transformer block.
        # Studies suggest that this is where "facts" are added to
        # embeddings.
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )

        # Layer normalization is key because information is _added_
        # to embedding vectors. To ensure that the numbers don't explode,
        # we normalize back to a mean of 0 and stdev of 1
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()

        # Dropout allows better learning. During different training
        # rounds, we temporarily turn off individual neurons. That
        # makes the network more resilient because it has to encode the
        # correct answer even when missing some neurons.
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x: Any, training: bool = False, mask: Any | None = None) -> Any:
        """The call method is defined by keras. This is used to define
        the forward pass.

        Args:
            x (Any): previous layer
            training (bool, optional): if True, it is during training.
                Defaults to False.
            mask (Any | None, optional): an optional attention mask.
                Defaults to None.

        Returns:
            Any: the processing from the transformer
        """
        attn_out = self.attn(x, x, attention_mask=mask)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)
