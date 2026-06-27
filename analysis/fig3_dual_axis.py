"""
Figure 3: Agent initiative by model
Dual-axis grouped bar chart — оранжевый стиль.
  Left axis:  Average initiative score (1–5)
  Right axis: Fraction of action-taking rounds (%)  = доля задач с кодом C

Данные считаются из ../model_family_experiments/results_table.tsv + старая
gpt-oss/heretic пара (см. data_loader.py).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_dataset

# ── данные ────────────────────────────────────────────────────────────────────
models, scores, fracs, is_abl = load_dataset()

# ── цвета ─────────────────────────────────────────────────────────────────────
AMBER       = "#D97706"   # тёмно-оранжевый  → score bars
AMBER_LIGHT = "#FCD34D"   # светло-жёлтый    → fraction bars

# ── настройка фигуры ──────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False})
fig, ax1 = plt.subplots(figsize=(14, 6), dpi=180)
fig.patch.set_facecolor("white")
ax2 = ax1.twinx()

x = np.arange(len(models))
w = 0.40

# ── бары ──────────────────────────────────────────────────────────────────────
b1 = ax1.bar(x - w/2, scores, width=w, color=AMBER,       label="Average initiative score", zorder=3)
b2 = ax2.bar(x + w/2, fracs,  width=w, color=AMBER_LIGHT, label="Fraction of action-taking rounds", zorder=3)

# ── разделители между парами семейств ──────────────────────────────────────────
for i in range(2, len(models), 2):
    ax1.axvline(i - 0.5, color="#eee", linestyle="-", linewidth=1, zorder=0)

# ── аннотации ─────────────────────────────────────────────────────────────────
for bar, val in zip(b1, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}",
             ha="center", va="bottom", fontsize=8.5, color=AMBER, fontweight="bold")

for bar, val in zip(b2, fracs):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.012, f"{val:.0%}",
             ha="center", va="bottom", fontsize=8.5, color="#92610A", fontweight="bold")

# ── маркеры original/abliterated под подписями ─────────────────────────────────
xticklabels = [f"{'😈' if a else '😇'} {m}" for m, a in zip(models, is_abl)]

# ── оси ───────────────────────────────────────────────────────────────────────
ax1.set_ylabel("Average initiative score (1–5)", color=AMBER, fontsize=12)
ax1.set_ylim(0, 5.8)
ax1.tick_params(axis="y", labelcolor=AMBER)
ax1.set_xticks(x)
ax1.set_xticklabels(xticklabels, fontsize=9, rotation=30, ha="right")
ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
ax1.spines["left"].set_color(AMBER)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_color("#ccc")

ax2.set_ylabel("Fraction of action-taking rounds", color="#92610A", fontsize=12)
ax2.set_ylim(0, 1.12)
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax2.tick_params(axis="y", labelcolor="#92610A")
ax2.spines["right"].set_color(AMBER_LIGHT)
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)

# ── заголовок и легенда ───────────────────────────────────────────────────────
ax1.set_title("Agent initiative by model — Original (😇) vs Abliterated (😈)",
              fontsize=14, fontweight="bold", pad=12)

handles = [b1, b2]
labels  = ["Average initiative score", "Fraction of action-taking rounds"]
ax1.legend(handles, labels, loc="upper right", fontsize=10,
           framealpha=0.9, edgecolor="#ddd")

fig.tight_layout()
fig.savefig("fig3_dual_axis.png", dpi=180, bbox_inches="tight")
print("Saved: fig3_dual_axis.png")
