import pandas as pd
import tensorflow as tf

# =========================
# 1. Load data
# =========================
df = pd.read_csv("training_data.csv")

sentences = df["corrupted"].tolist()

# =========================
# 2. Build vocabulary properly
# =========================
PAD = "<pad>"
START = "<start>"
END = "<end>"
UNK = "<unk>"

# get unique characters (ORDERED, not set)
chars = sorted(set("".join(sentences)))

vocab = [PAD, START, END, UNK] + chars

char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for ch, i in char2idx.items()}

vocab_size = len(vocab)

# =========================
# 3. Hyperparameters
# =========================
d_model = 64
num_heads = 4
max_len = 200

# =========================
# 4. Embedding layer
# =========================
embedding = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=d_model
)

# =========================
# 5. Positional embedding
# =========================
pos_embedding = tf.keras.layers.Embedding(
    input_dim=max_len,
    output_dim=d_model
)

def add_positional_encoding(x):
    seq_len = tf.shape(x)[1]
    positions = tf.range(seq_len)
    pos_vecs = pos_embedding(positions)
    return x + pos_vecs

# =========================
# 6. Encoder block
# =========================
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()

        # key_dim = d_model / num_heads (IMPORTANT FIX)
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn_output = self.att(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

encoder_block = EncoderBlock(d_model, num_heads)

# =========================
# 7. Encoder pipeline
# =========================
def encoder_forward(x):
    x = embedding(x)
    x = add_positional_encoding(x)
    x = encoder_block(x)   # reuse SAME layer
    return x