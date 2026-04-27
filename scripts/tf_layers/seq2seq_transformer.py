from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tf_layers import (
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

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad)
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(
        self, wrong_correct_pairs: list[tuple[str, str]], training_epochs: int = 500
    ):
        """Fit a model from a list of pairs of wrong and correct words."""
        wrong_data, corrected_comparison_data, corrected_label_data = (
            self._word_pairs_to_matrix(wrong_correct_pairs)
        )

        # Convert to tensors
        wrong_data = torch.tensor(wrong_data, dtype=torch.long)
        corrected_comparison_data = torch.tensor(
            corrected_comparison_data, dtype=torch.long
        )
        corrected_label_data = torch.tensor(
            corrected_label_data, dtype=torch.long
        )

        self.model.train()

        for epoch in range(training_epochs):
            self.optimizer.zero_grad()

            outputs = self.model(wrong_data, corrected_comparison_data)
            # reshape for CrossEntropy: (N, C)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = corrected_label_data.view(-1)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def correct(self, txt: str) -> str:
        """Feed a misspelled word through the model and decode the output."""
        self.model.eval()

        input_array = torch.tensor(
            self.tokenizer.encode_input(txt), dtype=torch.long
        ).unsqueeze(0)

        decoded = [self.tokenizer.bos]

        for _ in range(self.max_len + 1):
            decoded_array = torch.tensor(
                decoded + [self.tokenizer.pad] * (self.max_len + 2 - len(decoded)),
                dtype=torch.long,
            ).unsqueeze(0)

            with torch.no_grad():
                preds = self.model(input_array, decoded_array)

            next_id = int(torch.argmax(preds[0, len(decoded) - 1]))
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

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc_emb = token_position_embedding.TokenAndPositionEmbedding(
                    vocab_size, max_len + 1, embed_dim
                )
                self.encoder = transformer_encoder_block.TransformerEncoderBlock(
                    embed_dim, num_heads, ff_dim
                )

                self.dec_emb = token_position_embedding.TokenAndPositionEmbedding(
                    vocab_size, max_len + 2, embed_dim
                )
                self.decoder = transformer_decoder_block.TransformerDecoderBlock(
                    embed_dim, num_heads, ff_dim
                )

                self.fc = nn.Linear(embed_dim, vocab_size)

            def forward(self, enc_inputs, dec_inputs):
                enc_x = self.enc_emb(enc_inputs)
                enc_x = self.encoder(enc_x)

                dec_x = self.dec_emb(dec_inputs)
                dec_x = self.decoder(dec_x, enc_x)

                return self.fc(dec_x)

        return Model()