"""FastSpeech 2 with a variance adaptor (duration, pitch, energy), configurable by compression strategy."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, STRATEGIES


class FFTBlock(nn.Module):
    """Multi-head attention followed by a convolutional feed-forward network."""

    def __init__(self, hidden_dim, num_heads, ff_dim, kernel_size, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_dim, ff_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(ff_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.drop(a))
        r = self.drop(F.relu(self.conv1(x.transpose(1, 2))))
        r = self.conv2(r).transpose(1, 2)
        return self.norm2(x + self.drop(r))


class VariancePredictor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        r = self.drop(self.norm1(F.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)))
        r = self.drop(self.norm2(F.relu(self.conv2(r.transpose(1, 2))).transpose(1, 2)))
        return self.linear(r).squeeze(-1)


class LengthRegulator(nn.Module):
    """Expand phoneme-level features to frame level using integer durations."""

    def forward(self, x, durations, target_len=None):
        out = [torch.repeat_interleave(x[i], durations[i].long(), dim=0) for i in range(x.size(0))]
        max_len = target_len or max(o.size(0) for o in out)
        padded = torch.zeros(x.size(0), max_len, x.size(2), device=x.device)
        for i, o in enumerate(out):
            n = min(o.size(0), max_len)
            padded[i, :n] = o[:n]
        return padded


class VarianceAdaptor(nn.Module):
    """Duration, pitch and energy predictors; the same architecture in all four strategies."""

    def __init__(self, hidden_dim, cfg):
        super().__init__()
        va = cfg.va_hidden
        self.in_proj = nn.Linear(hidden_dim, va) if hidden_dim != va else nn.Identity()
        self.out_proj = nn.Linear(va, hidden_dim) if hidden_dim != va else nn.Identity()
        self.dur_pred, self.pitch_pred, self.energy_pred = (VariancePredictor(va, cfg.dropout) for _ in range(3))
        self.lr = LengthRegulator()
        self.pitch_emb = nn.Embedding(cfg.n_pitch_bins, va)
        self.energy_emb = nn.Embedding(cfg.n_energy_bins, va)
        self.register_buffer("pitch_bins", torch.linspace(cfg.pitch_min, cfg.pitch_max, cfg.n_pitch_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(cfg.energy_min, cfg.energy_max, cfg.n_energy_bins - 1))

    def forward(self, x, gt_dur=None, gt_pitch=None, gt_energy=None, target_len=None):
        x = self.in_proj(x)
        log_dur = self.dur_pred(x)
        dur = gt_dur if gt_dur is not None else torch.clamp(torch.exp(log_dur) - 1, min=0).round().long()
        x = self.lr(x, dur, target_len)
        p_pred, e_pred = self.pitch_pred(x), self.energy_pred(x)
        p_val = torch.nan_to_num(gt_pitch if gt_pitch is not None else p_pred, nan=0.0)
        e_val = gt_energy if gt_energy is not None else e_pred
        x = x + self.pitch_emb(torch.bucketize(p_val.detach(), self.pitch_bins)) \
              + self.energy_emb(torch.bucketize(e_val.detach(), self.energy_bins))
        return self.out_proj(x), log_dur, p_pred, e_pred


class FastSpeech2(nn.Module):
    def __init__(self, strategy="baseline", cfg=CFG):
        super().__init__()
        s = STRATEGIES[strategy]
        h, nh, ff = s["hidden_dim"], s["num_heads"], s["ff_dim"]
        self.embed = nn.Embedding(cfg.vocab_size, h, padding_idx=0)
        self.register_buffer("pe", self._sinpe(4096, h))
        self.encoder = nn.ModuleList(FFTBlock(h, nh, ff, cfg.kernel_size, cfg.dropout) for _ in range(s["encoder_layers"]))
        self.va = VarianceAdaptor(h, cfg)
        self.decoder = nn.ModuleList(FFTBlock(h, nh, ff, cfg.kernel_size, cfg.dropout) for _ in range(s["decoder_layers"]))
        self.out = nn.Linear(h, cfg.n_mels)

    @staticmethod
    def _sinpe(max_len, d):
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        return pe.unsqueeze(0)

    def forward(self, phones, src_mask=None, gt_dur=None, gt_pitch=None, gt_energy=None, target_len=None):
        x = self.embed(phones) + self.pe[:, : phones.size(1)]
        for block in self.encoder:
            x = block(x, mask=src_mask)
        x, log_dur, p_pred, e_pred = self.va(x, gt_dur, gt_pitch, gt_energy, target_len)
        for block in self.decoder:
            x = block(x)
        return self.out(x), log_dur, p_pred, e_pred


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    base = count_params(FastSpeech2("baseline"))
    for k in STRATEGIES:
        n = count_params(FastSpeech2(k))
        print(f"{k:9s} {n:>10,} params  {n * 4 / 1e6:5.1f} MB  {100 * (1 - n / base):+.0f}% vs baseline")
