import torch
from torch import nn
from torch.nn import functional as F
from BPE import BytePairEncoding


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, n_embd: int):
        super().__init__()
        pe = torch.zeros((seq_len, n_embd), dtype=torch.float)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # This will be of shape (seq_len, 1)
        div = 10_000 ** (torch.arange(0, n_embd, 2, dtype=torch.float) / n_embd)  # This is of shape (n_embd//2)

        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor):
        return self.pe[:x.shape[1], :]  # Grab the seq_len dimension and return


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.tensor):
        return self.seq(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, device: str):
        super().__init__()
        self.n_heads = n_heads
        self.device = device

        assert n_embd % n_heads == 0, f"n_embd={n_embd} % n_heads={n_heads} != 0"
        self.head_size = n_embd // n_heads

        self.qkv_layers = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.tensor):
        B, T, C = x.shape  # (batch_size, seq_len, n_embd)
        qkv = self.qkv_layers(x)  # (batch_size, seq_len, 3 * n_embd)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.head_size)  # (batch_size, seq_len, n_heads, 3 * head_size)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, 3 * head_size)

        # Now split it into their respective matrices and apply attention mechanism
        q, k, v = qkv.chunk(3, dim=-1)  # Each matrix is now of shape (batch_size, n_heads, seq_len, head_size)

        y = (q @ k.transpose(-1, -2)) / (k.shape[-1] ** 0.5)  # After matmul last two dim, shape=(batch_size, n_heads, seq_len, seq_len)
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=self.device))  # Does T == seq_len since using padding?
        y = y.masked_fill(~mask, -1e9)
        y = F.softmax(y, dim=-1)
        values = y @ v  # Final matmul results in original shape, (batch_size, n_heads, seq_len, head_size)

        values = values.permute(0, 2, 1, 3)  # Now we revert back by doing in reverse order. Permute first
        values = values.reshape(B, T, C)  # Then reshape

        return self.proj(values)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float, device: str):
        super().__init__()
        self.mh_attention = MultiHeadSelfAttention(n_embd, n_heads, device)
        self.ln1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.tensor):
        x = x + self.dropout1(self.mh_attention(self.ln1(x)))
        x = x + self.dropout2(self.ffwd(self.ln2(x)))
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, n_embd: int, n_heads: int, n_layers: int, dropout: float, device: str, tokenizer: BytePairEncoding):
        super().__init__()
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.device = device
        self.tokenizer = tokenizer

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_enc = PositionalEncoding(seq_len, n_embd)
        self.decoder_layers = nn.Sequential(*[DecoderBlock(n_embd, n_heads, dropout, device) for _ in range(n_layers)])

        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Sharing the weights between embedding and final linear layer
        self.lm_head.weight = self.embedding.weight

    def forward(self, x: torch.tensor):
        token_emb = self.embedding(x) * (self.n_embd ** 0.5)
        pe = self.pos_enc(x)
        logits = token_emb + pe

        logits = self.decoder_layers(logits)
        logits = self.lm_head(self.ln(logits))

        return logits

    def generate(self, text: str, max_tokens: int, k=10):
        byte_repr = self.tokenizer.encode(text)
        idx = torch.tensor(byte_repr, dtype=torch.long, device=self.device)
        idx = idx.unsqueeze(0)  # To add a batch dimension

        for _ in range(max_tokens):
            trimmed = idx[:, -self.seq_len:]
            x = self(trimmed)
            logit = x[:, -1, :]

            top_k_logits, top_k_idx = torch.topk(logit, k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)

            while True:
                index = torch.multinomial(top_k_probs, num_samples=1)  # Index based on top_k
                index = top_k_idx[0, index]  # Index based on original vocab
                if index not in {self.tokenizer.PAD_token, self.tokenizer.UNK_token}:
                    break

            if index == self.tokenizer.EOS_token:
                break

            idx = torch.cat((idx, index), dim=-1)

        response = self.tokenizer.decode(idx[0].tolist())
        if response == text:  # Meaning first token is <EOS> though that should be nearly impossible
            return ""

        response = response[len(text):]  # To avoid repeating user prompt
        return response if response[0] != " " else response[1:]  # Remove unnecessary space
