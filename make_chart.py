"""Draw assets/model_sizes.png from results/model_sizes.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results" / "model_sizes.csv").open()))
names = ["Baseline\n4/4 layers, 256", "S1: layers\n2/2, 256", "S2: width\n4/4, 128", "S3: both\n2/2, 128"]
vals = [int(r["params"]) / 1e6 for r in rows]
BG, INK, MUT, BLUE, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#a9a8a3"
fig, ax = plt.subplots(figsize=(7.5, 4), dpi=200, facecolor=BG)
ax.set_facecolor(BG)
bars = ax.bar(names, vals, color=[GREY, BLUE, BLUE, BLUE], width=0.55)
for b, v, r in zip(bars, vals, rows):
    lab = f"{v:.1f}M" if r["reduction_vs_baseline_pct"] == "0" else f"{v:.1f}M  (-{r['reduction_vs_baseline_pct']}%)"
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, lab, ha="center", color=INK, fontsize=9)
ax.set_ylabel("Parameters (millions)", color=MUT)
ax.set_title("Measured size of the four FastSpeech 2 variants", loc="left", fontsize=11, fontweight="bold", color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT, labelsize=8)
fig.tight_layout()
out = root / "assets" / "model_sizes.png"
fig.savefig(out, facecolor=BG)
print("wrote", out)
