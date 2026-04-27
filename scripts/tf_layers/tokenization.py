import numpy as np


def vocab():
    """The complete vocabulary."""
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
    ]  # beginning of string, end of string
    chars = list("abcdefghijklmnopqrstuvwxyz")
    return special_tokens + chars


class Tokenization:
    def __init__(self, max_len: int = 20):
        """Tokenize input strings into individual characters. Tokenization
        turns the strings into a list of integers.

        Args:
            max_len (int, optional): _description_. Defaults to 20.
        """
        # We need a padding token (to make all inputs the same length),
        # a beginning of string, and an end of string.
        # Beginning of string is only for the decoder layer. It allows
        # us to add onto it.
        self.max_len = max_len
        self.vocab = vocab()
        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_token = {i: ch for ch, i in self.token_to_id.items()}

    def encode_input(self, txt: str) -> np.array:
        """Encode the input (possibly misspelled string). Note that the
        tokenization is the max length + 1 (for end of string)

        Args:
            txt (str): input text

        Returns:
            np.array: the tokenized string
        """
        assert len(txt) <= self.max_len
        ids = [self.token_to_id[c] for c in txt] + [self.token_to_id["<eos>"]]
        ids += [self.token_to_id["<pad>"]] * ((self.max_len + 1) - len(ids))
        return np.array(ids, dtype=np.int32)

    def encode_label(self, txt: str) -> np.array:
        """Encode the label (correctly spelled string). Note that this
        is the max length + 2 (for beginning and end of string).

        Args:
            txt (str): correctly spelled string / label

        Returns:
            np.array: the tokenized string
        """
        assert len(txt) <= self.max_len
        dec_in = [self.token_to_id["<bos>"]] + [self.token_to_id[c] for c in txt]
        dec_in += [self.token_to_id["<pad>"]] * ((self.max_len + 2) - len(dec_in))

        dec_out = [self.token_to_id[c] for c in txt] + [self.token_to_id["<eos>"]]
        dec_out += [self.token_to_id["<pad>"]] * ((self.max_len + 2) - len(dec_out))
        return np.array(dec_in, dtype=np.int32), np.array(dec_out, dtype=np.int32)

    def decode(self, arr: np.array) -> str:
        """Decode an array of tokens into a string."""
        return "".join(self.id_to_token[i] for i in arr[1:])  # skip <bos>

    # This decorator turns a method (function) into a property (as if
    # it's a variable). So you use .vocab_size as a variable rather than
    # .vocab_size()
    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad(self):
        return self.token_to_id["<pad>"]

    @property
    def bos(self):
        return self.token_to_id["<bos>"]

    @property
    def eos(self):
        return self.token_to_id["<eos>"]
