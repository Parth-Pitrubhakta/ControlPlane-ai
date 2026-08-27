import React, { useEffect, useState } from "react";
import { getQueue, postVerdict, getFeedbackStats, recalibrate } from "./api.js";
import { Highlighted, FindingList } from "./TraceDetail.jsx";

export default function Review({ onChanged }) {
  const [rows, setRows] = useState([]);
  const [sel, setSel] = useState(null);
  const [verdicts, setVerdicts] = useState({});
  const [actAgree, setActAgree] = useState(null);
  const [stats, setStats] = useState(null);
  const [recal, setRecal] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const load = async () => {
    try {
      const [q, s] = await Promise.all([getQueue(), getFeedbackStats()]);
      setRows(q.rows); setStats(s);
      if (sel && !q.rows.find((r) => r.id === sel.id)) setSel(null);
    } catch (e) { setErr(String(e)); }
  };
  useEffect(() => { load(); }, []);

  const pick = (r) => { setSel(r); setVerdicts({}); setActAgree(null); setRecal(null); };

  const submit = async () => {
    if (!sel) return;
    setBusy(true); setErr("");
    try {
      await postVerdict(sel.id, {
        reviewer: "reviewer",
        act_agree: actAgree,
        items: Object.entries(verdicts).map(([idx, agree]) => ({ idx: +idx, agree })),
      });
      setSel(null); setVerdicts({});
      await load(); onChanged?.();
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  const runRecal = async (dry) => {
    setBusy(true); setErr("");
    try { setRecal(await recalibrate(dry)); await load(); onChanged?.(); }
    catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  const allJudged = sel && Object.keys(verdicts).length === (sel.fnd?.length || 0);

  return (
    <div className="split">
      <div className="panel">
        <h3><span>review queue</span><span className="note">{rows.length} waiting</span></h3>
        {rows.length === 0 ? (
          <div className="empty">nothing escalated or blocked is waiting</div>
        ) : (
          <table>
            <thead><tr><th>trace</th><th>tenant</th><th>action</th><th>findings</th><th>policy</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className={`click ${sel?.id === r.id ? "sel" : ""}`} onClick={() => pick(r)}>
                  <td className="num dim">{r.id.slice(0, 10)}</td>
                  <td>{r.tenant}/{r.geo}</td>
                  <td><span className={`act ${r.act}`}>{r.act}</span></td>
                  <td className="num">{r.fnd?.length || 0}</td>
                  <td className="num dim">{r.pol_ver}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <h3><span>feedback so far</span></h3>
        {!stats?.verdicts ? (
          <div className="empty">no verdicts recorded yet</div>
        ) : (
          <table>
            <thead><tr><th>label</th><th>judged</th><th>false pos</th><th>fp rate</th><th>highest fp conf</th></tr></thead>
            <tbody>
              {Object.entries(stats.by_label).map(([k, v]) => (
                <tr key={k}>
                  <td><span className="chip">{k}</span></td>
                  <td className="num">{v.n}</td>
                  <td className="num">{v.fp}</td>
                  <td className="num">{(v.fp_rate * 100).toFixed(0)}%</td>
                  <td className="num dim">{v.fp_conf_max ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div style={{ padding: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <button className="btn" disabled={busy} onClick={() => runRecal(true)}>preview recalibration</button>
          <button className="btn pri" disabled={busy} onClick={() => runRecal(false)}>recalibrate and publish</button>
          <span className="note">moves detector thresholds, writes a new policy version</span>
        </div>
        {recal && (
          <div className="resp"><pre>{JSON.stringify(recal, null, 2)}</pre></div>
        )}
      </div>

      <div className="panel">
        {!sel ? (
          <div className="empty">select a queued trace to review</div>
        ) : (
          <>
            <h3>
              <span>{sel.id}</span>
              <span className={`act ${sel.act}`}>{sel.act}</span>
            </h3>
            <div className="note" style={{ padding: "8px 14px" }}>
              {sel.tenant}/{sel.geo} &middot; policy {sel.pol_ver} &middot; risk {sel.risk}
            </div>
            <h3>response the model produced</h3>
            <Highlighted text={sel.resp} findings={sel.fnd} />
            <h3>was each finding correct?</h3>
            <FindingList
              findings={sel.fnd}
              verdicts={verdicts}
              onVerdict={(i, agree) => setVerdicts((v) => ({ ...v, [i]: agree }))}
            />
            <div style={{ padding: 12, display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <span className="note">action {sel.act} was</span>
              <button className={`btn ok ${actAgree === true ? "on" : ""}`} onClick={() => setActAgree(true)}>right</button>
              <button className={`btn no ${actAgree === false ? "on" : ""}`} onClick={() => setActAgree(false)}>wrong</button>
              <button className="btn pri" disabled={busy || !allJudged} onClick={submit}>
                {allJudged ? "submit verdict" : `judge all ${sel.fnd?.length || 0} findings`}
              </button>
            </div>
            {err && <div className="err" style={{ padding: "0 14px 12px" }}>{err}</div>}
          </>
        )}
      </div>
    </div>
  );
}
