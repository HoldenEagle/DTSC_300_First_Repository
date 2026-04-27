import keras
import numpy as np

from dtsc330_26.tf_layers import (
    token_position_embedding,
    tokenization,
    transformer_decoder_block,
    transformer_encoder_block,
)


class Seq2SeqTransformer:
    def __init__(
        self,
        max_len: int = 32,
        embed_dim: int = 64,
        num_heads: int = 2,
        ff_dim: int = 128,
    ):
        """Initialize a seq2seq misspelling fixing transformer.

        Args:
            max_len (int, optional): the maximum input string length.
                Defaults to 32.
            embed_dim (int, optional): the number of embedding dimensions.
                Defaults to 64. This is probably too high, but it is
                reasonable.
            num_heads (int, optional): the number of parallel attention
                blocks to run. Defaults to 2.
            ff_dim (int, optional): the number of dimensions in the
                feedforward layers in each transformer block. Defaults
                to 128.
        """
        vocab_size = len(tokenization.vocab())
        self.max_len = max_len
        self.tokenizer = tokenization.Tokenization()
        self.model = self._create_model(
            vocab_size, max_len, embed_dim, num_heads, ff_dim
        )
        self.model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def fit(
        self, wrong_correct_pairs: list[tuple[str, str]], training_epochs: int = 300
    ):
        """Fit a model from a list of pairs of wrong and correct words."""
        wrong_data, corrected_comparison_data, corrected_label_data = (
            self._word_pairs_to_matrix(wrong_correct_pairs)
        )
        # One weird thing about the data is that it's padded. We have
        # to make sure that padding is ignored for computing errors.
        # If the model adds anything after end of string, it is ignored
        # To do so, we create a masking array.

        # sample weights so <pad> tokens do not count in the loss
        pad_id = self.tokenizer.pad
        pad_mask = (corrected_label_data != pad_id).astype(np.float32)

        self.model.fit(
            x=(wrong_data, corrected_comparison_data),
            y=corrected_label_data,
            sample_weight=pad_mask,
            epochs=training_epochs,
            verbose=1,
        )

    def correct(self, txt: str) -> str:
        """Feed a misspelled word through the model and decode the output."""
        input_array = self.tokenizer.encode_input(txt)

        # Start the decoder with a beginning of string. We will add onto
        # it exactly as LLMs append tokens onto strings.
        decoded = [self.tokenizer.bos]

        # The decoder can produce + 2 length (<bos> and <eos>). We
        # already have bos, so we can loop through + 1
        for _ in range(self.max_len + 1):
            # The decoder still has to be of the correct length
            decoded_array = np.array(
                decoded + [self.tokenizer.pad] * (self.max_len + 2 - len(decoded)),
                dtype=np.int32,
            )
            preds = self.model.predict(
                [input_array[np.newaxis, :], decoded_array[np.newaxis, :]], verbose=0
            )

            # Find the last predicted value and check for end of string
            next_id = int(np.argmax(preds[0, len(decoded) - 1]))
            if next_id == self.tokenizer.eos:
                break

            decoded.append(next_id)

        return self.tokenizer.decode(decoded)

    def _word_pairs_to_matrix(
        self, wrong_correct_pairs: list[tuple[str, str]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert word pairs to three numpy matrices of input training,
        decoder input (for comparison), and decoder output (for label)."""
        wrong_data = []
        corrected_comparison_data = []
        corrected_label_data = []

        for src, tgt in wrong_correct_pairs:
            s = self.tokenizer.encode_input(src)
            di, do = self.tokenizer.encode_label(tgt)
            wrong_data.append(s)
            corrected_comparison_data.append(di)
            corrected_label_data.append(do)

        wrong_data = np.array(wrong_data, dtype=np.int32)
        corrected_comparison_data = np.array(corrected_comparison_data, dtype=np.int32)
        corrected_label_data = np.array(corrected_label_data, dtype=np.int32)

        return wrong_data, corrected_comparison_data, corrected_label_data

    def _create_model(
        self,
        vocab_size: int,
        max_len: int = 32,
        embed_dim: int = 64,
        num_heads: int = 2,
        ff_dim: int = 128,
    ):
        """Create the model itself."""
        enc_inputs = keras.Input(shape=(None,), dtype="int32", name="encoder_input")
        dec_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_input")

        enc_x = token_position_embedding.TokenAndPositionEmbedding(
            vocab_size, max_len + 1, embed_dim
        )(enc_inputs)
        enc_x = transformer_encoder_block.TransformerEncoderBlock(
            embed_dim, num_heads, ff_dim
        )(enc_x)

        dec_x = token_position_embedding.TokenAndPositionEmbedding(
            vocab_size, max_len + 2, embed_dim
        )(dec_inputs)
        dec_x = transformer_decoder_block.TransformerDecoderBlock(
            embed_dim, num_heads, ff_dim
        )(dec_x, enc_x)

        outputs = keras.layers.Dense(vocab_size, activation="softmax")(dec_x)

        return keras.Model([enc_inputs, dec_inputs], outputs)
