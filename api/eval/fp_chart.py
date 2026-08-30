"""The false-positive reduction chart.

Shows what a detector threshold move actually buys: false positives removed
against recall retained. The recalibration endpoint picks its cut the same way,
so the point marked "after" is the one reviewer feedback would produce, not a
number chosen to make the curve look good.

    python -m api.eval.fp_chart --report bench/report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="bench/report.json")
    ap.add_argument("--out", default="bench/fp_reduction.png")
    ap.add_argument("--keep-recall", type=float, default=0.80,
                    help="recall floor the threshold move must respect")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads(pathlib.Path(a.report).read_text())
    curves = rep.get("fp_curve") or {}
    labels = [k for k in ("bias", "contradicted", "unverifiable", "pii")
              if k in curves and any(r["fp"] for r in curves[k])]
    if not labels:
        raise SystemExit("no curves with false positives in the report")

    fig, axes = plt.subplots(1, len(labels), figsize=(5.0 * len(labels), 4.8),
                             squeeze=False)
    summary = {}

    for ax, lab in zip(axes[0], labels):
        c = curves[lab]
        base = c[0]
        # the cut recalibration would choose: fewest false positives that still
        # clears the recall floor
        ok = [r for r in c if r["recall"] >= a.keep_recall * base["recall"]]
        after = min(ok, key=lambda r: r["fp"]) if ok else base

        fps = [r["fp"] for r in c]
        recs = [r["recall"] for r in c]
        ax.plot(fps, recs, "-", color="#c9c9c9", lw=1.2, zorder=1)
        sc = ax.scatter(fps, recs, c=[r["thr"] for r in c], cmap="viridis",
                        s=22, zorder=2)
        ax.scatter([base["fp"]], [base["recall"]], s=150, facecolors="none",
                   edgecolors="#B4551C", lw=2, zorder=3)
        ax.annotate(f"before  thr {base['thr']:.2f}\n{base['fp']} FP, recall {base['recall']:.2f}",
                    (base["fp"], base["recall"]), textcoords="offset points",
                    xytext=(-10, -34), fontsize=8, color="#B4551C", ha="right",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.scatter([after["fp"]], [after["recall"]], s=150, marker="*",
                   color="#1F6F6B", zorder=4)
        ax.annotate(f"after  thr {after['thr']:.2f}\n{after['fp']} FP, recall {after['recall']:.2f}",
                    (after["fp"], after["recall"]), textcoords="offset points",
                    xytext=(14, 18), fontsize=8, color="#1F6F6B",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

        drop = (1 - after["fp"] / base["fp"]) * 100 if base["fp"] else 0.0
        ax.set_title(f"{lab}\n{drop:.0f}% fewer false positives, "
                     f"recall {base['recall']:.2f} to {after['recall']:.2f}",
                     fontsize=10, pad=12)
        ax.set_xlabel("false positives on the benign set")
        ax.set_ylabel("recall")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(-0.08, 1.18)
        ax.margins(x=0.14)
        summary[lab] = {"before": base, "after": after, "fp_drop_pct": round(drop, 1)}

    cb = fig.colorbar(sc, ax=axes[0].tolist(), label="confidence threshold",
                      fraction=0.012, pad=0.035)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("False-positive reduction from a threshold move "
                 f"(benign set, n={rep['sets']['benign']['n']})",
                 fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.80, wspace=0.30, right=0.90)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    pathlib.Path(a.out.replace(".png", ".json")).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: {"thr": f"{v['before']['thr']} -> {v['after']['thr']}",
                          "fp": f"{v['before']['fp']} -> {v['after']['fp']}",
                          "recall": f"{v['before']['recall']} -> {v['after']['recall']}",
                          "fp_drop_pct": v["fp_drop_pct"]}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
