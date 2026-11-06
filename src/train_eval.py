"""Dataset, FastSpeech 2 loss, training loop, and evaluation (MCD and real-time factor).

STATUS: in the course notebook this ran with PLACEHOLDER phonemes (10 all-zero tokens per utterance and uniform
durations), because forced alignment was planned for a later week. The loop trains, but the text input carries no
information, so the mel/duration/pitch/energy numbers it produces are not valid TTS quality results. Replace
`LJSpeechDataset.__getitem__` with real phoneme ids and forced-alignment durations before drawing conclusions.
"""
import time
from pathlib import Path

import numpy as np
import scipy.fft
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from config import CFG, STRATEGIES
from model import FastSpeech2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LJSpeechDataset(Dataset):
    def __init__(self, ids, pre_dir):
        self.ids, self.d = ids, Path(pre_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        mel = torch.tensor(np.load(self.d / "mel" / f"{fid}.npy").T)
        pitch = torch.tensor(np.load(self.d / "pitch" / f"{fid}.npy"))
        energy = torch.tensor(np.load(self.d / "energy" / f"{fid}.npy"))
        T = mel.shape[0]
        n_ph = 10  # PLACEHOLDER: no real phonemes yet (see module docstring)
        return dict(mel=mel, pitch=pitch, energy=energy, phones=torch.zeros(n_ph, dtype=torch.long),
                    durations=torch.ones(n_ph, dtype=torch.long) * (T // n_ph))


def collate(batch):
    B, T = len(batch), max(b["mel"].shape[0] for b in batch)
    P = max(b["phones"].shape[0] for b in batch)
    out = dict(mel=torch.zeros(B, T, batch[0]["mel"].shape[1]), pitch=torch.zeros(B, T), energy=torch.zeros(B, T),
               phones=torch.zeros(B, P, dtype=torch.long), durations=torch.zeros(B, P, dtype=torch.long))
    for i, b in enumerate(batch):
        t, p = b["mel"].shape[0], b["phones"].shape[0]
        out["mel"][i, :t], out["pitch"][i, :t], out["energy"][i, :t] = b["mel"], b["pitch"][:t], b["energy"][:t]
        out["phones"][i, :p], out["durations"][i, :p] = b["phones"], b["durations"]
    return out


class FS2Loss(nn.Module):
    """L1 on mel, log-duration, pitch (voiced frames only) and energy."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mel_p, mel_g, logdur_p, dur_g, pit_p, pit_g, eng_p, eng_g):
        T = min(mel_p.size(1), mel_g.size(1))
        mel = self.l1(mel_p[:, :T], mel_g[:, :T])
        dur = self.l1(logdur_p, torch.log(dur_g.float().clamp(min=1)))
        valid = ~torch.isnan(pit_g)
        pit = self.l1(pit_p[valid], pit_g[valid]) if valid.any() else mel_p.new_zeros(1).squeeze()
        eng = self.l1(eng_p, eng_g)
        return mel + dur + pit + eng


def run_epoch(model, loader, crit, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for b in loader:
            ph, mel, pit, eng, dur = (b[k].to(DEVICE) for k in ("phones", "mel", "pitch", "energy", "durations"))
            mel_p, ld, pp, ep = model(ph, gt_dur=dur, gt_pitch=pit, gt_energy=eng, target_len=mel.size(1))
            loss = crit(mel_p, mel, ld, dur, pp, pit, ep, eng)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                opt.step(); sched.step()
            total += loss.item()
    return total / len(loader)


def compute_mcd(mel_pred, mel_ref, n_mfcc=13):
    """MCD (dB) between two (T, n_mels) log-mel arrays, via a DCT of each frame."""
    def mfcc(lm):
        return np.array([scipy.fft.dct(r, type=2, norm="ortho")[1:n_mfcc + 1] for r in lm])
    T = min(len(mel_pred), len(mel_ref))
    d = mfcc(mel_pred[:T]) - mfcc(mel_ref[:T])
    return (10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt((d ** 2).sum(axis=1)))


@torch.no_grad()
def measure_rtf(model, n_trials=40, n_phones=50):
    """Real-time factor = inference time / generated audio duration, averaged over random inputs."""
    model.eval()
    n_frames = n_phones * 10
    audio_s = n_frames * CFG.hop_length / CFG.sample_rate
    ph = torch.randint(1, CFG.vocab_size, (1, n_phones), device=DEVICE)
    dur = torch.ones(1, n_phones, dtype=torch.long, device=DEVICE) * 10
    for _ in range(5):
        model(ph, gt_dur=dur, target_len=n_frames)
    times = []
    for _ in range(n_trials):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(ph, gt_dur=dur, target_len=n_frames)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / audio_s)
    return float(np.mean(times)), float(np.std(times))


if __name__ == "__main__":
    import sys
    pre = Path(sys.argv[1])
    ids = sorted(p.stem for p in (pre / "mel").glob("*.npy"))
    np.random.seed(42); np.random.shuffle(ids)
    n_val = int(len(ids) * CFG.val_split)
    tr = DataLoader(LJSpeechDataset(ids[n_val:], pre), CFG.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    va = DataLoader(LJSpeechDataset(ids[:n_val], pre), CFG.batch_size, collate_fn=collate, num_workers=2)
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2  # the notebook's quick run used 2 epochs
    for key in STRATEGIES:
        model = FastSpeech2(key).to(DEVICE)
        opt = Adam(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.98))
        sched = OneCycleLR(opt, max_lr=CFG.learning_rate, steps_per_epoch=len(tr), epochs=epochs, pct_start=0.1)
        crit = FS2Loss()
        for ep in range(1, epochs + 1):
            print(key, ep, run_epoch(model, tr, crit, opt, sched), run_epoch(model, va, crit))
        print(key, "RTF", measure_rtf(model))
