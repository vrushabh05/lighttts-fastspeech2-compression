"""Audio, architecture and training settings plus the four compression strategies."""
from dataclasses import dataclass


@dataclass
class Config:
    # audio
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    # text
    vocab_size: int = 256
    # baseline architecture
    kernel_size: int = 9
    dropout: float = 0.2
    # variance adaptor (identical in every strategy)
    va_hidden: int = 256
    pitch_min: float = 50.0
    pitch_max: float = 600.0
    energy_min: float = 0.0
    energy_max: float = 300.0
    n_pitch_bins: int = 256
    n_energy_bins: int = 256
    # training
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    val_split: float = 0.05


CFG = Config()

STRATEGIES = {
    "baseline": dict(name="Baseline (FastSpeech 2)", encoder_layers=4, decoder_layers=4, hidden_dim=256, num_heads=2, ff_dim=1024),
    "S1": dict(name="Light v1: layer reduction", encoder_layers=2, decoder_layers=2, hidden_dim=256, num_heads=2, ff_dim=1024),
    "S2": dict(name="Light v2: dimension reduction", encoder_layers=4, decoder_layers=4, hidden_dim=128, num_heads=1, ff_dim=512),
    "S3": dict(name="Light v3: combined", encoder_layers=2, decoder_layers=2, hidden_dim=128, num_heads=1, ff_dim=512),
}
