import React from "react";

const TIER_COLOR = ["#3f9d6f", "#2e86c9", "#c9962e", "#9a5bc4"];

/**
 * The persistent header. Deliberately not a gauge: a single 0-100 risk number
 * would imply the system decides by score, which it does not. What matters to
 * an operator is what verification actually cost and where it was spent.
 */
export default function Header({ sum, series, health }) {
  const tiers = sum?.tiers || {};
  const total = Object.values(tiers).reduce((a, b) => a + b, 0) || 1;
  const segs = [0, 1, 2, 3].map((t) => ({
    t, n: tiers[String(t)] || 0, pct: ((tiers[String(t)] || 0) / total) * 100,
  }));
  const buckets = series?.buckets || [];
  const peak = Math.max(1, ...buckets.map((b) => b.p95 || 0));
  const det = health?.det === "up" || health?.det === "mock";

  return (
    <div className="hdr">
      <div className="brand">
        ControlPlane.ai<span>{sum?.n ?? 0} traces / {sum?.window ?? "-"}</span>
      </div>

      <div className="metric">
        <div className="k">overhead p50</div>
        <div className="v">{sum?.overhead?.p50 ?? "-"}<small> ms</small></div>
      </div>
      <div className="metric">
        <div className="k">overhead p95</div>
        <div className="v">{sum?.overhead?.p95 ?? "-"}<small> ms</small></div>
      </div>
      <div className="metric">
        <div className="k">tier 1 rate</div>
        <div className="v">{sum ? (sum.t1_rate * 100).toFixed(1) : "-"}<small> %</small></div>
      </div>
      <div className="metric">
        <div className="k">alerts / 100</div>
        <div className="v">{sum?.alerts_per_100 ?? "-"}</div>
      </div>

      <div className="tierbar">
        <div className="row">
          {segs.map((s) => (
            <div key={s.t} className="seg" title={`tier ${s.t}: ${s.n}`}
                 style={{ width: `${s.pct}%`, background: TIER_COLOR[s.t] }} />
          ))}
        </div>
        <div className="legend">
          {segs.map((s) => (
            <span key={s.t}>
              <i style={{ background: TIER_COLOR[s.t] }} />T{s.t} {s.n}
            </span>
          ))}
        </div>
      </div>

      <div className="metric">
        <div className="k">p95 over time</div>
        <div className="spark" title="verification overhead p95 per bucket">
          {buckets.map((b, i) => (
            <div key={i} style={{ height: `${Math.max(2, ((b.p95 || 0) / peak) * 100)}%` }} />
          ))}
        </div>
      </div>

      <div className="metric">
        <div className="k">detectors</div>
        <div className="v" style={{ color: det ? "var(--allow)" : "var(--block)", fontSize: 13 }}>
          {det ? "up" : "down"}
        </div>
      </div>
    </div>
  );
}
