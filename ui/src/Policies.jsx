import React, { useEffect, useState } from "react";
import { getPolicies, putPolicy } from "./api.js";

export default function Policies({ onChanged }) {
  const [rows, setRows] = useState([]);
  const [key, setKey] = useState("");
  const [text, setText] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const load = async () => {
    const p = await getPolicies();
    setRows(p.rows);
    if (!key && p.rows.length) select(p.rows[0], p.rows);
  };
  useEffect(() => { load(); }, []);

  const select = (r) => {
    setKey(`${r.tenant}/${r.geo}`);
    setText(JSON.stringify(r, null, 2));
    setMsg(""); setErr("");
  };

  const save = async () => {
    setMsg(""); setErr("");
    let doc;
    try { doc = JSON.parse(text); }
    catch (e) { setErr(`not valid JSON: ${e.message}`); return; }
    try {
      const r = await putPolicy(doc);
      if (r.error) { setErr(r.error); return; }
      setMsg(`published ${r.pol_ver}`);
      await load(); onChanged?.();
    } catch (e) { setErr(String(e)); }
  };

  return (
    <div className="split">
      <div className="panel">
        <h3><span>active policies</span><span className="note">{rows.length}</span></h3>
        <table>
          <thead><tr><th>tenant</th><th>geo</th><th>version</th><th>med</th><th>high</th><th>pii floor</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.tenant}/${r.geo}`} className={`click ${key === `${r.tenant}/${r.geo}` ? "sel" : ""}`}
                  onClick={() => select(r)}>
                <td>{r.tenant}</td>
                <td>{r.geo}</td>
                <td className="num dim">{r.ver}</td>
                <td className="num">{r.thr?.med ?? "-"}</td>
                <td className="num">{r.thr?.high ?? "-"}</td>
                <td><span className={`act ${r.floors?.pii}`}>{r.floors?.pii}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note" style={{ padding: 12 }}>
          Editing publishes a new version. Bump `ver` first, or the gateway keeps the
          one it already has. Old versions stay readable so any past decision can be replayed.
        </div>
      </div>

      <div className="panel">
        <h3><span>edit {key || "-"}</span></h3>
        <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
          <textarea rows={26} value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button className="btn pri" onClick={save}>publish new version</button>
            {msg && <span className="note" style={{ color: "var(--allow)" }}>{msg}</span>}
            {err && <span className="err">{err}</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
