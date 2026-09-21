"""Draw the decode attention-kernel microbenchmark from the file written by attn_microbench.py.

For every KV length and method the chart shows two bottom-aligned bars, the data-movement
time D (wide) and the cache-resident compute time C (narrow, in front), and a marker at the
measured latency T. The label above a group is the latency of the BF16 kernel divided by
the latency of HACK.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MOVEMENT, MOVEMENT_DARK = "#eb6834", "#a8431a"
COMPUTE, COMPUTE_DARK = "#2a78d6", "#184f95"
INK, SECONDARY_INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHODS = {"hack": ("HACK", None), "bf16": ("BF16", "////"), "dequant": ("Dequant", "\\\\\\\\")}
BAR_WIDTH = 0.26


def length_label(tokens: int) -> str:
    return f"{tokens // 1024}K" if tokens % 1024 == 0 else str(tokens)


def draw_method(ax, x: float, row: dict, hatch: str | None) -> None:
    ax.bar(x, row["D_us"], BAR_WIDTH, color=MOVEMENT, hatch=hatch, edgecolor=MOVEMENT_DARK, linewidth=0, zorder=2)
    ax.bar(x, row["C_us"], BAR_WIDTH * 0.5, color=COMPUTE, hatch=hatch, edgecolor=COMPUTE_DARK, linewidth=0, zorder=3)
    ax.bar(x, row["C_us"], BAR_WIDTH * 0.5, fill=False, edgecolor="white", linewidth=1.0, zorder=4)
    half = BAR_WIDTH * 0.5
    ax.plot([x - half, x + half], [row["T_us"]] * 2, color=INK, linewidth=1.6, solid_capstyle="butt", zorder=5)


def legend_handles(methods: list[str]) -> list:
    handles = []
    for method in methods:
        name, hatch = METHODS[method]
        movement = dict(facecolor=MOVEMENT, edgecolor=MOVEMENT_DARK, label=f"{name}: data movement D")
        compute = dict(facecolor=COMPUTE, edgecolor=COMPUTE_DARK, label=f"{name}: compute C")
        handles += [Patch(hatch=hatch, linewidth=0, **movement), Patch(hatch=hatch, linewidth=0, **compute)]
    handles.append(Line2D([0], [0], color=INK, linewidth=1.6, label="latency T"))
    return handles


def style_axes(ax, lengths: list[int], log_scale: bool) -> None:
    ax.set_xticks(range(len(lengths)), [length_label(n) for n in lengths])
    ax.set_xlabel("KV length (tokens)", color=SECONDARY_INK)
    ax.set_ylabel("Time per decode step (µs)", color=SECONDARY_INK)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot(results: list[dict], title: str, log_scale: bool):
    lengths = sorted({row["kv_length"] for row in results})
    methods = [m for m in METHODS if any(row["method"] == m for row in results)]
    table = {(row["method"], row["kv_length"]): row for row in results}
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    top = max(max(row["T_us"], row["D_us"], row["C_us"]) for row in results)
    for group, length in enumerate(lengths):
        present = [m for m in methods if (m, length) in table]
        for slot, method in enumerate(present):
            x = group + (slot - (len(present) - 1) / 2) * (BAR_WIDTH + 0.06)
            draw_method(ax, x, table[(method, length)], METHODS[method][1])
        if ("hack", length) in table and ("bf16", length) in table:
            ratio = table[("bf16", length)]["T_us"] / table[("hack", length)]["T_us"]
            peak = max(max(table[(m, length)][key] for key in ("T_us", "D_us", "C_us")) for m in present)
            label = dict(xytext=(0, 4), textcoords="offset points", ha="center", color=SECONDARY_INK, fontsize=8)
            ax.annotate(f"{ratio:.2f}×", (group, peak), **label)
    style_axes(ax, lengths, log_scale)
    if not log_scale:
        ax.set_ylim(0, top * 1.45)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    legend = dict(ncols=3, fontsize=7.5, frameon=False, labelcolor=SECONDARY_INK, loc="upper left")
    ax.legend(handles=legend_handles(methods), **legend)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="JSON file written by attn_microbench.py")
    parser.add_argument("--output", default="attn_microbench.pdf")
    parser.add_argument("--log", action="store_true", help="logarithmic time axis")
    args = parser.parse_args()
    with open(args.results) as file:
        data = json.load(file)
    environment = data["environment"]
    title = f"Decode attention kernel, {environment['gpu']}, {environment['steps']} output tokens per measurement"
    figure = plot(data["results"], title, args.log)
    figure.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
