const j = async (url, opt) => {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

export const getSummary = (w = "1h", tenant) =>
  j(`/api/metrics/summary?window=${w}${tenant ? `&tenant=${tenant}` : ""}`);
export const getSeries = (w = "1h") => j(`/api/metrics/series?window=${w}&buckets=32`);
export const getTraces = (q = {}) => {
  const p = new URLSearchParams(Object.entries(q).filter(([, v]) => v !== "" && v != null));
  return j(`/api/traces?${p}`);
};
export const getTrace = (id) => j(`/api/traces/${id}`);
export const getQueue = () => j(`/api/review/queue?limit=50`);
export const getFeedbackStats = () => j(`/api/review/stats`);
export const postVerdict = (id, body) =>
  j(`/api/review/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
export const getPolicies = () => j(`/api/policies`);
export const putPolicy = (doc) =>
  j(`/api/policies`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(doc),
  });
export const recalibrate = (dry) => j(`/api/recalibrate?dry=${dry ? "true" : "false"}`, { method: "POST" });
export const getHealth = () => j(`/health`);
