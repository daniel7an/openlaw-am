"""Token-length distribution of a chunks.jsonl — run after every chunking change.

Lengths are measured in the exact embed format ("passage: {title}\n{text}", special
tokens included), i.e. the number the 512 ceiling actually applies to. Bins are
fixed (width 25, 0..525) so plots from different chunking strategies line up
bar-for-bar and can be compared side by side.

Usage:
    uv run python chunk_stats.py                                   # data/chunks.jsonl
    uv run python chunk_stats.py data/chunks.jsonl=511-sentence-safe old.jsonl=400-token
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from config import get

MODEL_LIMIT = get("chunking.model_limit")
OUT_DIR = Path("results")

BIN_W = 25
BINS = list(range(0, MODEL_LIMIT + 2 * BIN_W, BIN_W))

# dataviz reference palette (validated): categorical slots 1-2, light chrome.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def token_lengths(path: Path, tk) -> list[int]:
    chunks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    return [len(tk(f"passage: {c['title']}\n{c['text']}")["input_ids"]) for c in chunks]


def summarize(lens: list[int]) -> str:
    s = sorted(lens)
    med, p95 = s[len(s) // 2], s[int(len(s) * 0.95)]
    return f"n={len(s)}  median={med}  p95={p95}  max={s[-1]}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = sys.argv[1:] or ["data/chunks.jsonl=current"]
    inputs = [(Path(a.split("=")[0]), a.split("=")[1] if "=" in a else Path(a).stem) for a in args]
    if len(inputs) > 2:
        sys.exit("at most 2 corpora per plot — more stops being readable as grouped bars")

    tk = AutoTokenizer.from_pretrained(get("embedding.model", env="OPENLAW_EMBED_MODEL"))
    series = [(label, token_lengths(path, tk)) for path, label in inputs]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    n = len(series)
    group_w = BIN_W * 0.82  # 2px-ish surface gap between adjacent bins
    for i, (label, lens) in enumerate(series):
        counts = [0] * (len(BINS) - 1)
        for v in lens:
            counts[min(v // BIN_W, len(counts) - 1)] += 1
        w = group_w / n
        xs = [b + BIN_W * 0.09 + i * w for b in BINS[:-1]]
        ax.bar(xs, counts, width=w * 0.94, align="edge", color=SERIES[i],
               label=f"{label}   ({summarize(lens)})", zorder=3)

    ax.axvline(MODEL_LIMIT, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(MODEL_LIMIT - 6, ax.get_ylim()[1], f"embed ceiling {MODEL_LIMIT}",
            ha="right", va="top", fontsize=8, color=MUTED)

    ax.set_title("Chunk token lengths (embed format)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("tokens", color=INK_2, fontsize=9)
    ax.set_ylabel("chunks", color=INK_2, fontsize=9)
    ax.set_xticks(BINS[::2])
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
    ax.margins(x=0.01)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"chunk_tokens_{'_vs_'.join(l for l, _ in series)}.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")
    for label, lens in series:
        over = sum(1 for v in lens if v > MODEL_LIMIT)
        print(f"  {label}: {summarize(lens)}  over-{MODEL_LIMIT}: {over}")


if __name__ == "__main__":
    main()
