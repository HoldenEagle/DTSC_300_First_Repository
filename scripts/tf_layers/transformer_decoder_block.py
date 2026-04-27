from typing import Any

import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        """Create a new TransformerDecoderBlock layer. This is split
        into the definition (__init__) and the mechanism for calling
        it (call). Refer to Encoder block for simple understanding.
        Decoder block differs because it contains self attention to
        learn connections between entries AS WELL AS cross attention,
        learning the connection (in this case) between the misspelled
        word and the correctly spelled word.

        Args:
            embed_dim (int): number of embedding
            num_heads (int): number of parallel heads
            ff_dim (int): the dimensions of the feedforward layers
            dropout (float, optional): fraction of neurons to turn off
                during each training run. Defaults to 0.1.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self, x: Any, enc_out: Any, training: bool = False, enc_mask: Any | None = None
    ) -> Any:
        """The call method is defined by keras. This is used to define
        the forward pass.

        Args:
            x (Any): previous layer
            enc_out (Any): the output of the encoder part of the seq2seq
                model.
            training (bool, optional): if True, it is during training.
                Defaults to False.
            enc_mask (Any | None, optional): an optional attention mask.
                Defaults to None.

        Returns:
            Any: the processing from the transformer
        """
        # --- Causal mask (replaces use_causal_mask=True) ---
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        # Self-attention
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        self_attn_out = self.drop1(self_attn_out)
        x = self.norm1(x + self_attn_out)

        # Cross-attention
        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, attn_mask=enc_mask)
        cross_attn_out = self.drop2(cross_attn_out)
        x = self.norm2(x + cross_attn_out)

        # Feedforward
        ffn_out = self.ffn(x)
        ffn_out = self.drop3(ffn_out)
        return self.norm3(x + ffn_out)