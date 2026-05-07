import torch
import torch.nn as nn

class DecoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),nn.ReLU(),nn.Linear(d_ff, d_model))
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

    numerical_features = 12

    def __init__(self, input_size=12, d_model=64, n_heads=4, n_layers=2, d_ff=256, dropout=0.1, max_seq_len=20, n_drivers=30, n_tracks=30):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_embedding    = nn.Embedding(max_seq_len, d_model)
        self.driver_embedding = nn.Embedding(n_drivers, d_model)
        self.track_embedding  = nn.Embedding(n_tracks,  d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for i in range(n_layers)])
        self.output_head = nn.Linear(d_model, 1)

    def causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x):
        _, seq_len, _ = x.shape

        num_feats  = x[:, :, :self.numerical_features]
        track_ids  = x[:, 0, 12].long()
        driver_ids = x[:, 0, 13].long()

        positions = torch.arange(seq_len, device=x.device)
        h = (self.input_projection(num_feats) + self.pos_embedding(positions) + self.driver_embedding(driver_ids).unsqueeze(1) + self.track_embedding(track_ids).unsqueeze(1))
        h = self.dropout(h)
        causal_mask = self.causal_mask(seq_len, h.device)
        for layer in self.layers:
            h = layer(h, causal_mask)
        return self.output_head(h).squeeze(-1)

    def predict(self, context, horizon=5):
        self.eval()
        with torch.no_grad():
            preds = []
            current = context.clone()
            for i in range(horizon):
                out = self.forward(current)
                next_lap_time = out[:, -1:]
                new_step = current[:, -1:, :].clone()
                new_step[:, 0, 0] = next_lap_time.squeeze(-1)
                current = torch.cat([current[:, 1:, :], new_step], dim=1)
                preds.append(next_lap_time.squeeze(-1))
            return torch.stack(preds, dim=1)
