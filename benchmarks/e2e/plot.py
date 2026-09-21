"""Bar charts of end-to-end runs, drawn only from the `summary.json` files written by metrics.py.

    python benchmarks/e2e/plot.py --output-dir figures runs/baseline_10gbps runs/hack_10gbps

Writes latency.pdf (mean and P99 of JCT, TTFT, TPOT, one panel per metric), decomposition.pdf
(mean JCT split into phases), resources.pdf (KV bytes transferred per request, peak memory in use
on the decode GPU) and plotted_values.csv with every number shown in the figures.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PHASES = ("queueing", "prefill", "kv_transfer", "decode", "other")
PHASE_LABELS = ("Queueing", "Prefill", "KV transfer", "Decode", "Other")
LATENCY_PANELS = (("jct", "JCT (s)", 1.0), ("ttft", "TTFT (s)", 1.0), ("tpot", "TPOT (ms)", 1000.0))


def load_runs(run_dirs: list[Path]) -> list[tuple[str, dict]]:
    if len(run_dirs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs fit in one figure")
    return [(path.name, json.loads((path / "summary.json").read_text())) for path in run_dirs]


def style_axes(ax, value_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    base = "bottom" if value_axis == "y" else "left"
    ax.spines[base].set_visible(True)
    ax.spines[base].set_color(AXIS)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def format_value(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}"


def label_bar(ax, x: float, value: float) -> None:
    ax.text(x, value, format_value(value), ha="center", va="bottom", fontsize=8, color=SECONDARY)


def plot_latency(runs: list[tuple[str, dict]], output: Path, table: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(LATENCY_PANELS), figsize=(3.4 * len(LATENCY_PANELS), 3.2), facecolor=SURFACE)
    width = 0.34
    for ax, (metric, title, factor) in zip(axes, LATENCY_PANELS):
        style_axes(ax, "y")
        for index, (name, summary) in enumerate(runs):
            stats = summary.get(metric) or {}
            for offset, statistic, alpha in ((-width / 2, "mean", 1.0), (width / 2, "p99", 0.55)):
                value = stats.get(statistic)
                if value is None:
                    continue
                value *= factor
                ax.bar(index + offset, value, width, color=SERIES[index], alpha=alpha, edgecolor=SURFACE, linewidth=2)
                label_bar(ax, index + offset, value)
                table.append({"figure": "latency", "run": name, "quantity": f"{metric}_{statistic}", "value": value})
        ax.set_xticks(range(len(runs)), [name for name, _ in runs], rotation=20, ha="right")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].text(0.0, 1.16, "solid: mean, light: P99", transform=axes[0].transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def plot_decomposition(runs: list[tuple[str, dict]], output: Path, table: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(runs) + 1.6), facecolor=SURFACE)
    style_axes(ax, "x")
    for row, (name, summary) in enumerate(runs):
        left = 0.0
        for phase, label, color in zip(PHASES, PHASE_LABELS, SERIES):
            value = max(0.0, summary.get("decomposition_mean", {}).get(phase, 0.0))
            ax.barh(row, value, 0.5, left=left, color=color, edgecolor=SURFACE, linewidth=2,
                    label=label if row == 0 else None)  # fmt: skip
            left += value
            table.append({"figure": "decomposition", "run": name, "quantity": phase, "value": value})
        ax.text(left, row, " " + format_value(left) + " s", va="center", fontsize=8, color=SECONDARY)
    ax.set_yticks(range(len(runs)), [name for name, _ in runs])
    ax.invert_yaxis()
    ax.set_xlabel("Mean JCT (s)", fontsize=9, color=SECONDARY)
    ax.legend(ncols=len(PHASES), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8,
              labelcolor=SECONDARY)  # fmt: skip
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def plot_resources(runs: list[tuple[str, dict]], output: Path, table: list[dict]) -> None:
    panels = (
        ("KV transferred per request (MiB)", lambda s: (s.get("transfer_bytes") or {}).get("mean", 0.0) / 2**20),
        ("Peak memory in use, decode GPU (MiB)",
         lambda s: s.get("memory", {}).get("decode_peak_memory_in_use_mib", 0.0)),  # fmt: skip
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2), facecolor=SURFACE)
    for ax, (title, read) in zip(axes, panels):
        style_axes(ax, "y")
        for index, (name, summary) in enumerate(runs):
            value = read(summary)
            ax.bar(index, value, 0.5, color=SERIES[index], edgecolor=SURFACE, linewidth=2)
            label_bar(ax, index, value)
            table.append({"figure": "resources", "run": name, "quantity": title, "value": value})
        ax.set_xticks(range(len(runs)), [name for name, _ in runs], rotation=20, ha="right")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(output, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories that contain summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", default="pdf", choices=("pdf", "png", "svg"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs, table = load_runs(args.run_dirs), []
    plot_latency(runs, args.output_dir / f"latency.{args.format}", table)
    plot_decomposition(runs, args.output_dir / f"decomposition.{args.format}", table)
    plot_resources(runs, args.output_dir / f"resources.{args.format}", table)
    with open(args.output_dir / "plotted_values.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "run", "quantity", "value"])
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
