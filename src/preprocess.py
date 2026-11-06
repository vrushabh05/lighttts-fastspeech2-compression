"""Extract log-mel, pitch (F0) and energy for every LJSpeech utterance and cache them as .npy files."""
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from config import CFG


def wav_to_mel(path, cfg=CFG):
    wav, _ = librosa.load(path, sr=cfg.sample_rate)
    mel = librosa.feature.melspectrogram(y=wav, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                         win_length=cfg.win_length, n_mels=cfg.n_mels, fmin=cfg.f_min, fmax=cfg.f_max)
    return librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)


def wav_to_pitch(path, cfg=CFG):
    try:
        import parselmouth
        pit = parselmouth.Sound(path).to_pitch(time_step=cfg.hop_length / cfg.sample_rate,
                                               pitch_floor=cfg.pitch_min, pitch_ceiling=cfg.pitch_max)
        arr = pit.selected_array["frequency"]
        arr[arr == 0] = np.nan
        return arr
    except Exception:
        wav, _ = librosa.load(path, sr=cfg.sample_rate)
        f0, _, _ = librosa.pyin(wav, fmin=cfg.pitch_min, fmax=cfg.pitch_max, sr=cfg.sample_rate, hop_length=cfg.hop_length)
        return f0


def wav_to_energy(path, cfg=CFG):
    wav, _ = librosa.load(path, sr=cfg.sample_rate)
    stft = librosa.stft(wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    return np.linalg.norm(np.abs(stft), axis=0)


def fit(a, T):
    return a[:T] if len(a) > T else np.pad(a, (0, T - len(a)))


if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2])  # LJSpeech-1.1 folder, output folder
    meta = pd.read_csv(root / "metadata.csv", sep="|", header=None, names=["file_id", "transcript", "normalized"])
    for sub in ("mel", "pitch", "energy"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for fid in meta["file_id"]:
        wav = str(root / "wavs" / f"{fid}.wav")
        m = wav_to_mel(wav)
        T = m.shape[1]
        np.save(out / "mel" / f"{fid}.npy", m.astype(np.float32))
        np.save(out / "pitch" / f"{fid}.npy", fit(wav_to_pitch(wav), T).astype(np.float32))
        np.save(out / "energy" / f"{fid}.npy", fit(wav_to_energy(wav), T).astype(np.float32))
