import torch
import torch.nn as nn
import math


class DecoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # here we are splitting our 64 dimenssion embedding into 4 heads with 16 dimes per head
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=False)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class LapTimeTransformer(nn.Module):

    def __init__(self, input_size=11, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.1, max_seq_len=20): # 15 input and 5 prediction
        super().__init__()
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers) # 2 decoder blocks stacked sequentially
        ])

        self.output_head = nn.Linear(d_model, 1)

    def _causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=x.device)
        x = self.input_projection(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._causal_mask(seq_len, x.device)
        for layer in self.layers:
            x = layer(x, causal_mask)

        return self.output_head(x).squeeze(-1)

    def predict(self, context, horizon=5):
        self.eval()
        with torch.no_grad():
            preds = []
            current = context.clone()
            for _ in range(horizon):
                out = self.forward(current)
                next_lap_time = out[:, -1:]

                # Build next input: shift window by 1, append prediction
                # New timestep has predicted lap time + zeros for other features
                new_step = torch.zeros(current.shape[0], 1, current.shape[2], device=current.device)
                new_step[:, 0, 0] = next_lap_time.squeeze(-1)
                current = torch.cat([current[:, 1:, :], new_step], dim=1)
                preds.append(next_lap_time.squeeze(-1))

            return torch.stack(preds, dim=1)
