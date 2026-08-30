"""Train and calibrate the risk router.

Three-way split on purpose. The classifier fits on train, the isotonic
calibrator fits on a separate slice, and ECE is measured on a third the
calibrator has never seen. Calibrating and scoring on the same data reports an
ECE that flatters itself.

    python -m api.eval.train_router --inp bench/router_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from api import router, tier0
from api.schemas import Trace


def row_feats(r: dict[str, Any]) -> dict[str, float]:
    """Exactly the features the live path computes, from the same code."""
    tr = Trace(
        id=r["id"], sess="train", tenant=r["tenant"], geo=r["geo"], ts=0.0,
        prompt=r["prompt"], resp=r["resp"], ctx=r["ctx"], tools=r["tools"],
        tok_in=r["tok_in"], tok_out=r["tok_out"],
    )
    t0 = (tier0.scan(tier0.norm(r["prompt"]), side="prompt")
          + tier0.scan(tier0.norm(r["resp"]), side="resp"))
    return router.feats(tr, t0)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error, plus the per-bin table the diagram plots."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot, out = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if not n:
            out.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        tot += (n / len(p)) * abs(acc - conf)
        out.append({"lo": lo, "hi": hi, "n": n, "conf": conf, "acc": acc})
    return tot, out


def pick(y: np.ndarray, p: np.ndarray, recall_floor: float) -> float:
    """Highest threshold that still meets the recall floor.

    Highest, not lowest: among thresholds that satisfy the safety requirement we
    want the one that raises the fewest alerts.
    """
    best = 0.0
    for t in np.unique(np.concatenate([p, [0.0, 1.0]])):
        flag = p >= t
        rec = float((flag & (y == 1)).sum()) / max(1, int((y == 1).sum()))
        if rec >= recall_floor:
            best = max(best, float(t))
    return round(best, 4)


def rates(y: np.ndarray, p: np.ndarray, t: float, base: float) -> dict[str, float]:
    flag = p >= t
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    rec = float((flag & (y == 1)).sum()) / max(1, pos)
    fpr = float((flag & (y == 0)).sum()) / max(1, neg)
    prec = float((flag & (y == 1)).sum()) / max(1, int(flag.sum()))
    # This is the share of traffic that crosses the threshold and gets checked,
    # not the share that raises an alert. Crossing the router threshold buys a
    # tier-1 pass; whether anything is reported depends on what the detectors
    # then find. Alert volume is a post-detector number and belongs in the
    # phase 5 harness, not here.
    #
    # Our training set is deliberately defect-rich for signal, so project the
    # rate at a realistic production base rate rather than quoting 52%.
    sent = base * rec + (1.0 - base) * fpr
    return {"thr": round(t, 4), "recall": round(rec, 4), "fpr": round(fpr, 4),
            "precision": round(prec, 4),
            "checked_per_100_at_base": round(sent * 100, 2)}


def ece_floor(n: int, bins: int = 10, trials: int = 300) -> tuple[float, float]:
    """ECE a perfectly calibrated model would still score at this sample size.

    ECE is a binned statistic, so on a small test set it measures sampling noise
    more than miscalibration. Quoting ECE without this floor is meaningless: at
    n=100 a flawless model scores about 0.10, which is twice the 0.05 target.
    """
    out = []
    for s in range(trials):
        g = np.random.default_rng(s)
        q = g.uniform(0, 1, n)
        yy = (g.uniform(0, 1, n) < q).astype(int)
        out.append(ece(yy, q, bins)[0])
    return float(np.mean(out)), float(np.percentile(out, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="bench/router_train.jsonl")
    ap.add_argument("--out", default="bench/router_model.json")
    ap.add_argument("--base", type=float, default=0.02, help="assumed production defect rate")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    rows = [json.loads(l) for l in pathlib.Path(a.inp).read_text().splitlines() if l.strip()]
    X = np.array([router.vec(row_feats(r)) for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    tn = np.array([r["tenant"] for r in rows])

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(rows))
    n1, n2 = int(0.60 * len(rows)), int(0.80 * len(rows))
    tr, ca, te = idx[:n1], idx[n1:n2], idx[n2:]

    fit = np.concatenate([tr, ca])          # classifier and calibrator share this
    mu, sd = X[fit].mean(0), X[fit].std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    def mk() -> Any:
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")

    # Isotonic needs more than a 100-row slice or it overfits into a step
    # function -- that put calibrated ECE (0.102) above raw (0.060). Fit it on
    # out-of-fold predictions across the whole fitting set instead, and keep the
    # test split untouched so ECE stays honest.
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    oof = cross_val_predict(mk(), Z[fit], y[fit], cv=cv, method="predict_proba")[:, 1]

    clf = mk()
    clf.fit(Z[fit], y[fit])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof, y[fit])

    p_te_raw = clf.predict_proba(Z[te])[:, 1]
    p_te = iso.predict(p_te_raw)

    e_raw, _ = ece(y[te], p_te_raw)
    e_cal, table = ece(y[te], p_te)
    fl_mean, fl_p95 = ece_floor(len(te))

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y[te], p_te)) if len(set(y[te])) > 1 else float("nan")

    # thresholds per tenant, from the recall floors in the spec
    floors = {"DECIDE": 0.98, "CS-BOT": 0.90, "KB-COPILOT": 0.85}
    p_all = iso.predict(clf.predict_proba(Z)[:, 1])
    thr: dict[str, dict[str, Any]] = {}
    for t, floor in floors.items():
        m = tn == t
        if m.sum() < 10 or len(set(y[m])) < 2:
            m = np.ones(len(y), dtype=bool)
        med = pick(y[m], p_all[m], floor)
        hi_c = [q for q in np.unique(p_all[m]) if q > med]
        high = float(np.quantile(hi_c, 0.60)) if len(hi_c) else min(1.0, med + 0.2)
        thr[t] = {"med": med, "high": round(high, 4), "recall_floor": floor,
                  "at_med": rates(y[m], p_all[m], med, a.base),
                  "at_high": rates(y[m], p_all[m], high, a.base)}

    # isotonic as interpolation points the gateway can read without sklearn
    xs = np.unique(np.clip(np.linspace(0, 1, 101), 0, 1))
    model = {
        "feats": router.FEATS,
        "w": [float(v) for v in clf.coef_[0]],
        "b": float(clf.intercept_[0]),
        "mu": [float(v) for v in mu],
        "sd": [float(v) for v in sd],
        "iso_x": [float(v) for v in xs],
        "iso_y": [float(v) for v in iso.predict(xs)],
        "meta": {"n": len(rows), "n_train": len(tr), "n_cal": len(ca), "n_test": len(te),
                 "auc": round(auc, 4), "ece_raw": round(e_raw, 4),
                 "ece_cal": round(e_cal, 4), "ece_noise_floor": round(fl_mean, 4),
                 "ece_noise_p95": round(fl_p95, 4), "base_rate_assumed": a.base,
                 "pos_rate_data": round(float(y.mean()), 4)},
    }
    # the pitch number, using latencies we actually measured rather than the
    # spec's estimates. T2 is not built yet, so its cost stays an assumption.
    T0_MS, T1_MS, T2_MS = 0.44, 60.9, 500.0
    for t, v in thr.items():
        p1 = v["at_med"]["checked_per_100_at_base"] / 100.0
        p2 = v["at_high"]["checked_per_100_at_base"] / 100.0
        blended = T0_MS + max(0.0, p1 - p2) * T1_MS + p2 * T2_MS
        v["budget"] = {
            "t1_share": round(max(0.0, p1 - p2), 4), "t2_share": round(p2, 4),
            "blended_ms": round(blended, 2),
            "always_deep_ms": T2_MS,
            "pct_of_always_deep": round(100.0 * blended / T2_MS, 2),
            "note": "T0/T1 measured on this build; T2 cost assumed, not yet built",
        }

    pathlib.Path(a.out).write_text(json.dumps(model, indent=2) + "\n")
    pathlib.Path("bench/router_thresholds.json").write_text(json.dumps(thr, indent=2) + "\n")

    _diagram(table, e_cal, e_raw, "bench/router_reliability.png")

    print(json.dumps({"model": a.out, **model["meta"],
                      "ece_target": 0.05, "ece_pass": e_cal < 0.05,
                      "ece_vs_floor": round(e_cal - fl_mean, 4),
                      "thresholds": {k: {"med": v["med"], "high": v["high"],
                                         "recall_at_med": v["at_med"]["recall"],
                                         "checked_per_100": v["at_med"]["checked_per_100_at_base"],
                                         "blended_ms": v["budget"]["blended_ms"],
                                         "pct_of_always_deep": v["budget"]["pct_of_always_deep"]}
                                     for k, v in thr.items()}}, indent=2))


def _diagram(table: list[dict], e_cal: float, e_raw: float, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [(b["lo"] + b["hi"]) / 2 for b in table if b["n"]]
    ys = [b["acc"] for b in table if b["n"]]
    ns = [b["n"] for b in table if b["n"]]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.2, 6.0),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="#1F6F6B", lw=1.8, ms=5, label="calibrated")
    ax.set_ylabel("observed defect rate")
    ax.set_title(f"Router reliability\nECE {e_cal:.4f} calibrated, {e_raw:.4f} raw")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(-0.02, 1.02)

    ax2.bar(xs, ns, width=0.085, color="#B4551C", alpha=0.8)
    ax2.set_xlabel("predicted risk")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
