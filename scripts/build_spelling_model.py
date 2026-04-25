import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



# =========================
# 1. Load data
# =========================
df = pd.read_csv("training_data.csv")


corrupted_sentences = df["Corrupted"].tolist()
clean_sentences = df["Original"].tolist()
# =========================
# 2. Build vocabulary properly
# =========================
PAD = "<pad>"
START = "<start>"
END = "<end>"
UNK = "<unk>"

chars = sorted(set("".join(corrupted_sentences + clean_sentences)))

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


# =========================
# 5. Positional embedding
# =========================
def add_positional_encoding(x):
    seq_len = x.size(1)
    positions = torch.arange(seq_len, device=x.device)
    pos = pos_embedding(positions)
    return x + pos

# =========================
# 6. Encoder block
# =========================
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

# =========================
# 8. Decoder block
# =========================
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        seq_len = x.size(1)
        causal_mask = generate_causal_mask(seq_len).to(x.device)

        self_attn, _ = self.self_att(
            x, x, x,
            attn_mask=causal_mask
        )

        x = self.norm1(x + self_attn)

        cross_attn, _ = self.cross_att(x, enc_out, enc_out)
        x = self.norm2(x + cross_attn)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x

class SpellTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.encoder_block = EncoderBlock()
        self.decoder_block = DecoderBlock()

        self.output = nn.Linear(d_model, vocab_size)

    def encode(self, x):
        x = self.token_embedding(x)
        x = add_positional_encoding(x)
        return self.encoder_block(x)

    def decode(self, x, enc_out):
        x = self.token_embedding(x)
        x = add_positional_encoding(x)
        return self.decoder_block(x, enc_out)

    def forward(self, enc_inp, dec_inp):
        enc_out = self.encode(enc_inp)
        dec_out = self.decode(dec_inp, enc_out)
        return self.output(dec_out)

def encode_text(text):
    return [char2idx.get(c, char2idx[UNK]) for c in text]

encoder_inputs = []
decoder_inputs = []
decoder_targets = []

for _, row in df.iterrows():
    corrupted = row["Corrupted"]
    clean = row["Original"]

    enc = encode_text(corrupted)
    dec_in = [char2idx[START]] + encode_text(clean)
    dec_tar = encode_text(clean) + [char2idx[END]]

    encoder_inputs.append(torch.tensor(enc))
    decoder_inputs.append(torch.tensor(dec_in))
    decoder_targets.append(torch.tensor(dec_tar))
    
PAD_IDX = char2idx[PAD]

encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)

encoder_inputs = encoder_inputs.long()
decoder_inputs = decoder_inputs.long()
decoder_targets = decoder_targets.long()

print("Encoder input:")
print(encoder_inputs[:5])

print("Decoder input:")
print(decoder_inputs[:5])

print("Decoder target:")
print(decoder_targets[:5])

def generate_causal_mask(seq_len):
    # upper triangular = -inf (blocked)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

#begin training
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpellTransformer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

epochs = 40
batch_size = 16
num_samples = encoder_inputs.size(0)

for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_correct_chars = 0
    total_chars = 0
    correct_sentences = 0
    total_sentences = 0

    for i in range(0, num_samples, batch_size):

        enc_batch = encoder_inputs[i:i+batch_size].to(device)
        dec_in_batch = decoder_inputs[i:i+batch_size].to(device)
        dec_tar_batch = decoder_targets[i:i+batch_size].to(device)

        optimizer.zero_grad()

        output = model(enc_batch, dec_in_batch)

        # 1. CE loss
        output_flat = output.reshape(-1, vocab_size)
        target_flat = dec_tar_batch.reshape(-1)
        ce_loss = loss_fn(output_flat, target_flat)

        # 2. predictions
        preds = output.argmax(dim=-1)

        # 3. sentence penalty (ADD HERE)
        sentence_errors = 0
        for b in range(preds.size(0)):
            valid_len = dec_tar_batch[b] != PAD_IDX
            if not torch.equal(preds[b][valid_len], dec_tar_batch[b][valid_len]):
                sentence_errors += 1

        sentence_penalty = sentence_errors / preds.size(0)

        # 4. combined loss
        loss = ce_loss

        # 5. backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # =========================
        # 🔥 CHARACTER ACCURACY
        # =========================
        preds = output.argmax(dim=-1)

        mask = dec_tar_batch != PAD_IDX

        correct_chars = (preds == dec_tar_batch) & mask

        total_correct_chars += correct_chars.sum().item()
        total_chars += mask.sum().item()

        # =========================
        # 🔥 SENTENCE ACCURACY
        # =========================
        for b in range(preds.size(0)):
            pred_sent = preds[b]
            true_sent = dec_tar_batch[b]

            # remove padding
            valid_len = true_sent != PAD_IDX

            if torch.equal(
                pred_sent[valid_len],
                true_sent[valid_len]
            ):
                correct_sentences += 1

            total_sentences += 1

    # =========================
    # 📊 METRICS
    # =========================
    avg_loss = total_loss / (num_samples / batch_size)
    char_acc = total_correct_chars / total_chars
    sent_acc = correct_sentences / total_sentences

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {avg_loss:.4f} | "
        f"Char Acc: {char_acc:.4f} | "
        f"Sentence Acc: {sent_acc:.4f}"
    )
    
torch.save(model.state_dict(), "spell_transformer_weights2.pth")
print("Model saved!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpellTransformer().to(device)
model.load_state_dict(torch.load("spell_transformer_weights2.pth", map_location=device))
model.eval()
def predict_sentence(model, text, max_len=200):
    model.eval()

    # 1. encode input
    enc = encode_text(text)
    enc = torch.tensor(enc).unsqueeze(0).to(device)

    with torch.no_grad():

        # encoder forward
        enc_out = model.encode(enc)

        # start token
        dec_input = torch.tensor([[char2idx[START]]]).to(device)

        output_sentence = []

        for _ in range(max_len):

            # decoder forward
            dec_out = model.decode(dec_input, enc_out)

            logits = model.output(dec_out)  # (1, seq, vocab)

            next_token_logits = logits[:, -1, :]  # last step
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token == char2idx[END]:
                break

            output_sentence.append(idx2char[next_token])

            # append token to decoder input
            dec_input = torch.cat([
                dec_input,
                torch.tensor([[next_token]]).to(device)
            ], dim=1)

    return "".join(output_sentence)

test_sentence = "Ths is a smple sentnce with erors"

print("Input: ", test_sentence)
print("Output:", predict_sentence(model, test_sentence))

test_sentence = "This is a sample sentence with errors"

print("Input: ", test_sentence)
print("Output:", predict_sentence(model, test_sentence))

test_sentence = "We shold go outside and enjoi the sunshne"

print("Input: ", test_sentence)
print("Output:", predict_sentence(model, test_sentence))
