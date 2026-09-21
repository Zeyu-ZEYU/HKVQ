"""Print the accuracy results as one table per task (rows: methods, columns: models)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/accuracy"))
    args = parser.parse_args()
    tables = defaultdict(dict)
    for path in sorted(args.results.glob("*.json")):
        r = json.loads(path.read_text())
        tables[(r["task"], r["metric"])][(r["method"], r["model"].split("/")[-1])] = (r["score"], r["examples"])
    for (task, metric), cells in tables.items():
        models = sorted({m for _, m in cells})
        methods = sorted({m for m, _ in cells}, key=lambda m: (m != "baseline", m))
        print(f"\n{task} ({metric})")
        print(f"{'method':14s}" + "".join(f"{m:>24s}" for m in models))
        for method in methods:
            row = "".join(
                f"{cells[(method, model)][0] * 100:17.2f}% (n={cells[(method, model)][1]})" if (method, model) in cells else f"{'-':>24s}"
                for model in models
            )
            print(f"{method:14s}{row}")


if __name__ == "__main__":
    main()
