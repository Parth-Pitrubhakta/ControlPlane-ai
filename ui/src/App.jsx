import React, { useCallback, useEffect, useRef, useState } from "react";
import { getSummary, getSeries, getTraces, getTrace, getHealth, getQueue } from "./api.js";
import Header from "./Header.jsx";
import TraceDetail from "./TraceDetail.jsx";
import Review from "./Review.jsx";
import Policies from "./Policies.jsx";

const POLL_MS = 3000;

function Traces() {
  const [rows, setRows] = useState([]);
  const [sel, setSel] = useState(null);
  const [tenant, setTenant] = useState("");
  const [act, setAct] = useState("");
  const [live, setLive] = useState(true);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await getTraces({ limit: 60, tenant, act });
      setRows(r.rows); setErr("");
    } catch (e) { setErr(String(e)); }
  }, [tenant, act]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!live) return;
    const h = setInterval(load, POLL_MS);
    return () => clearInterval(h);
  }, [live, load]);

  const open = async (id) => {
    try { setSel(await getTrace(id)); } catch (e) { setErr(String(e)); }
  };

  return (
    <>
      <div className="bar">
        <select value={tenant} onChange={(e) => setTenant(e.target.value)}>
          <option value="">all tenants</option>
          <option>CS-BOT</option><option>KB-COPILOT</option><option>DECIDE</option>
        </select>
        <select value={act} onChange={(e) => setAct(e.target.value)}>
          <option value="">all actions</option>
          <option>allow</option><option>annotate</option><option>edit</option>
          <option>block</option><option>escalate</option>
        </select>
        <button className={`btn ${live ? "on" : ""}`} onClick={() => setLive((v) => !v)}>
          {live ? "live" : "paused"}
        </button>
        <button className="btn" onClick={load}>refresh</button>
        {err && <span className="err">{err}</span>}
      </div>

      <div className="split">
        <div className="panel">
          <h3><span>traces</span><span className="note">{rows.length}</span></h3>
          {rows.length === 0 ? (
            <div className="empty">no traces yet. send a request to the gateway on port 8080.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>action</th><th>tenant</th><th>tier</th><th>risk</th>
                  <th>findings</th><th>response</th><th>ms</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className={`click ${sel?.id === r.id ? "sel" : ""}`} onClick={() => open(r.id)}>
                    <td><span className={`act ${r.act}`}>{r.act}</span></td>
                    <td className="dim">{r.tenant}/{r.geo}</td>
                    <td className="num">{r.tier}</td>
                    <td className="num dim">{r.risk?.toFixed?.(2) ?? "-"}</td>
                    <td>
                      <div className="chips">
                        {r.chips.slice(0, 4).map(([dim, label], i) => (
                          <span key={i} className={`chip dim-${dim}`}>{label}</span>
                        ))}
                        {r.chips.length > 4 && <span className="chip">+{r.chips.length - 4}</span>}
                      </div>
                    </td>
                    <td className="trunc dim">{r.resp || <em>empty</em>}</td>
                    <td className="num">{r.overhead}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <TraceDetail t={sel} />
      </div>
    </>
  );
}

export default function App() {
  const [tab, setTab] = useState("traces");
  const [sum, setSum] = useState(null);
  const [series, setSeries] = useState(null);
  const [health, setHealth] = useState(null);
  const [queueN, setQueueN] = useState(0);
  const [window_, setWindow] = useState("1h");
  const bump = useRef(0);

  const refresh = useCallback(async () => {
    try {
      const [s, se, h, q] = await Promise.all([
        getSummary(window_), getSeries(window_), getHealth(), getQueue(),
      ]);
      setSum(s); setSeries(se); setHealth(h); setQueueN(q.n);
    } catch { /* header degrades rather than blanking the page */ }
  }, [window_]);

  useEffect(() => { refresh(); }, [refresh, bump.current]);
  useEffect(() => {
    const h = setInterval(refresh, POLL_MS);
    return () => clearInterval(h);
  }, [refresh]);

  return (
    <div className="app">
      <Header sum={sum} series={series} health={health} />
      <div className="tabs">
        <button className={tab === "traces" ? "on" : ""} onClick={() => setTab("traces")}>traces</button>
        <button className={tab === "review" ? "on" : ""} onClick={() => setTab("review")}>
          review<span className="count">{queueN}</span>
        </button>
        <button className={tab === "policies" ? "on" : ""} onClick={() => setTab("policies")}>policies</button>
        <div style={{ flex: 1 }} />
        <select value={window_} onChange={(e) => setWindow(e.target.value)}
                style={{ alignSelf: "center", marginRight: 4 }}>
          <option value="5m">last 5m</option>
          <option value="1h">last 1h</option>
          <option value="24h">last 24h</option>
          <option value="all">all</option>
        </select>
      </div>
      <div className="body">
        {tab === "traces" && <Traces />}
        {tab === "review" && <Review onChanged={refresh} />}
        {tab === "policies" && <Policies onChanged={refresh} />}
      </div>
    </div>
  );
}
