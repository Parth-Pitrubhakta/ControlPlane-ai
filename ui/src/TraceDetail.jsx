import React from "react";

/**
 * Renders the response with findings highlighted inline.
 *
 * Spans can overlap -- a sentence can be both unverifiable and carry PII -- so
 * this cuts the text at every span boundary and tags each slice with whatever
 * covers it, rather than nesting elements and hoping.
 */
export function Highlighted({ text, findings, side = "resp" }) {
  if (!text) return <div className="empty">no text</div>;
  const fs = (findings || []).filter(
    (f) => (f.side || "resp") === side && f.dim !== "cost" &&
           f.span && f.span[1] > f.span[0] && f.span[1] <= text.length
  );
  if (!fs.length) return <div className="resp">{text}</div>;

  const cuts = new Set([0, text.length]);
  fs.forEach((f) => { cuts.add(f.span[0]); cuts.add(f.span[1]); });
  const pts = [...cuts].sort((a, b) => a - b);

  const parts = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    if (a >= b) continue;
    const covering = fs.filter((f) => f.span[0] <= a && b <= f.span[1]);
    const slice = text.slice(a, b);
    if (!covering.length) { parts.push(<span key={a}>{slice}</span>); continue; }
    const worst = covering.reduce((m, f) => (f.sev > m.sev ? f : m), covering[0]);
    const title = covering
      .map((f) => `${f.label} sev ${f.sev} conf ${f.conf}${f.evid ? ` [${f.evid}]` : ""} - ${f.det}`)
      .join("\n");
    parts.push(<mark key={a} className={`dim-${worst.dim}`} title={title}>{slice}</mark>);
  }
  return <div className="resp">{parts}</div>;
}

export function FindingList({ findings, onVerdict, verdicts }) {
  if (!findings?.length) return <div className="empty">no findings</div>;
  return (
    <div className="fnd">
      {findings.map((f, i) => (
        <div className="item" key={i}>
          <span className={`chip dim-${f.dim}`}>{f.label}</span>
          <span className="sev">sev {f.sev}</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="meta">
              conf {f.conf} / {f.det} / {f.side || "resp"} [{f.span?.[0]}:{f.span?.[1]}]
              {f.evid ? ` / evidence ${f.evid}` : ""}
            </div>
          </div>
          {onVerdict && (
            <div style={{ display: "flex", gap: 4 }}>
              <button className={`btn ok ${verdicts?.[i] === true ? "on" : ""}`}
                      onClick={() => onVerdict(i, true)}>agree</button>
              <button className={`btn no ${verdicts?.[i] === false ? "on" : ""}`}
                      onClick={() => onVerdict(i, false)}>disagree</button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function TraceDetail({ t }) {
  if (!t) return <div className="panel"><div className="empty">select a trace</div></div>;
  const lat = t.lat || {};
  return (
    <div className="panel">
      <h3>
        <span>trace {t.id}</span>
        <span className={`act ${t.act}`}>{t.act}</span>
      </h3>
      <dl className="kv">
        <dt>tenant</dt><dd>{t.tenant} / {t.geo}</dd>
        <dt>policy</dt><dd>{t.pol_ver || "-"}</dd>
        <dt>tier</dt><dd>{t.tier} (risk {t.risk})</dd>
        <dt>latency</dt>
        <dd>
          t0 {lat.t0 ?? "-"} / t1 {lat.t1 ?? "-"} / decide {lat.decide ?? "-"} ms
          {lat.shadow != null ? ` / shadow ${lat.shadow} (${lat.shadow_n} win)` : ""}
        </dd>
        <dt>tokens</dt><dd>{t.tok_in} in / {t.tok_out} out / ${t.cost}</dd>
        <dt>session</dt><dd>{t.sess}</dd>
        {t.tools?.length ? (<><dt>tools</dt><dd>{t.tools.join(", ")}</dd></>) : null}
      </dl>

      <h3>prompt</h3>
      <Highlighted text={t.prompt} findings={t.fnd} side="prompt" />

      <h3>response</h3>
      <Highlighted text={t.resp} findings={t.fnd} side="resp" />

      <h3>findings ({t.fnd?.length || 0})</h3>
      <FindingList findings={t.fnd} />

      {t.ovr && (
        <>
          <h3>reviewer override</h3>
          <div className="resp"><pre>{JSON.stringify(t.ovr, null, 2)}</pre></div>
        </>
      )}
    </div>
  );
}
