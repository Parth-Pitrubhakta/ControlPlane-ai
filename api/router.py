"""Risk router: predicts how much verification a response is worth.

What this is not: a risk gauge. The float it emits selects a tier and is kept
on the trace for offline analysis. It never reaches decide.py, and no action is
ever derived from it (invariant 3).

Black-box by default (invariant 6). Every feature below is computable from the
request and the response text alone -- no logprobs, no hidden states. The two
logprob features are declared but only populated when CAP_LOGPROBS is on, and
the model is trained without them so the system works with the flag off.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
from typing import Any

import structlog

from api.schemas import Finding, Policy, Trace

log = structlog.get_logger("router")

CAP_LOGPROBS = os.getenv("CAP_LOGPROBS", "0") == "1"
MODEL_PATH = os.getenv("ROUTER_MODEL", "bench/router_model.json")

TENANTS = ("CS-BOT", "KB-COPILOT", "DECIDE")
GEOS = ("IN", "EU", "US")

# Ordered feature vector. Changing this list invalidates a trained model, so the
# trained file carries a copy and load() refuses a mismatch.
FEATS: list[str] = [
    "t_cs", "t_kb", "t_dec",          # tenant, one-hot
    "g_in", "g_eu", "g_us",           # geo, one-hot
    "sens",                           # sensitive-topic score of the question
    "plen",                           # prompt length, log scaled
    "t0_pii", "t0_inj",               # tier 0 counts
    "rk_max", "rk_mean",              # retrieval scores
    "ndocs",                          # how much evidence we actually have
    "rlen",                           # response length, log scaled
    "ent_dens",                       # proper-noun density
    "num_dens",                       # numeric density: numbers are checkable claims
    "ncite",                          # explicit citations in the response
    "nhedge",                         # hedging language
    "ntool",                          # tool calls
    "tok_out",                        # output tokens, log scaled
    # Two features beyond the spec's list, added on evidence. Without them the
    # router is structurally blind to grounding defects: nothing else here can
    # see whether a claim is supported, so a stale-wiki answer looks identical
    # to a correct one and AUC sat at 0.76. Both are black-box (invariant 6) and
    # cost microseconds.
    "num_unsup",                      # share of asserted numbers found in no chunk
    "num_unsup_n",                    # how many, absolute: a ratio alone is
                                      # length-confounded on short answers
    "ovl",                            # content-word overlap with the context
]
FEATS_LP: list[str] = ["lp_mean", "lp_min"]

_SENS = re.compile(
    r"(?i)\b(?:refund|charge|billing|invoice|payment|card|bank|account|password|"
    r"cancel|delete|terminate|legal|lawsuit|liable|contract|warranty|claim|"
    r"medical|prescription|diagnos\w+|insurance|tax|salary|aadhaar|pan\b|kyc|"
    r"complaint|escalat\w+|compensation|breach|fraud)\b"
)
_HEDGE = re.compile(
    r"(?i)\b(?:i think|i believe|probably|possibly|might|may be|appears? to|"
    r"seems? to|as far as i know|generally|typically|usually|should be|"
    r"i am not sure|cannot confirm)\b"
)
_CITE = re.compile(r"\[(?:source|ref|doc)[:\]]|\(\s*(?:source|see)\s*:", re.I)
_NUM = re.compile(r"\b\d[\d,.]*\b")
_WORD = re.compile(r"\b[\w'-]+\b")
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")


def _log1p(x: float, s: float = 1.0) -> float:
    return math.log1p(max(0.0, x) / s)


_STOP = frozenset(
    "the a an and or but if of to in on for with is are was were be been being "
    "this that these those it its as at by from you your we our they their i "
    "will would can could may might should do does did not no yes have has had "
    "there here what which who when where how".split()
)


def _content(txt: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(txt)
            if len(w) > 2 and w.lower() not in _STOP}


def _nums(txt: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(txt)}


def feats(
    tr: Trace,
    t0: list[Finding] | None = None,
    lp: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract the feature dict. Pure, and cheap enough to run on every request."""
    t0 = t0 or []
    resp, prompt = tr.resp, tr.prompt
    words = _WORD.findall(resp)
    nw = max(1, len(words))
    scores = [float(c.get("score", 0.0) or 0.0) for c in tr.ctx]

    f: dict[str, float] = {
        "t_cs": 1.0 if tr.tenant == "CS-BOT" else 0.0,
        "t_kb": 1.0 if tr.tenant == "KB-COPILOT" else 0.0,
        "t_dec": 1.0 if tr.tenant == "DECIDE" else 0.0,
        "g_in": 1.0 if tr.geo == "IN" else 0.0,
        "g_eu": 1.0 if tr.geo == "EU" else 0.0,
        "g_us": 1.0 if tr.geo == "US" else 0.0,
        "sens": min(1.0, len(_SENS.findall(prompt)) / 3.0),
        "plen": _log1p(len(prompt), 100.0),
        "t0_pii": float(sum(1 for x in t0 if x.label == "pii")),
        "t0_inj": float(sum(1 for x in t0 if x.label == "inject")),
        "rk_max": max(scores) if scores else 0.0,
        "rk_mean": (sum(scores) / len(scores)) if scores else 0.0,
        "ndocs": _log1p(len(tr.ctx)),
        "rlen": _log1p(len(resp), 100.0),
        "ent_dens": len(_PROPER.findall(resp)) / nw,
        "num_dens": len(_NUM.findall(resp)) / nw,
        "ncite": float(len(_CITE.findall(resp))),
        "nhedge": float(len(_HEDGE.findall(resp))),
        "ntool": float(len(tr.tools)),
        "tok_out": _log1p(tr.tok_out, 100.0),
    }

    ctx_txt = " ".join(str(c.get("text", "")) for c in tr.ctx)
    rn, cn = _nums(resp), _nums(ctx_txt)
    # a number stated in the answer that appears in no retrieved chunk is the
    # cheapest available signal that something is being asserted unsupported
    f["num_unsup"] = (len(rn - cn) / len(rn)) if rn else 0.0
    f["num_unsup_n"] = _log1p(len(rn - cn))
    rw, cw = _content(resp), _content(ctx_txt)
    f["ovl"] = (len(rw & cw) / len(rw)) if rw else 0.0
    if CAP_LOGPROBS and lp:
        f["lp_mean"] = float(lp.get("mean", 0.0))
        f["lp_min"] = float(lp.get("min", 0.0))
    return f


def vec(f: dict[str, float], names: list[str] | None = None) -> list[float]:
    return [float(f.get(k, 0.0)) for k in (names or FEATS)]


class Router:
    """Logistic regression plus isotonic calibration, stored as plain JSON.

    Deliberately not a pickle: the gateway must be able to load a model trained
    by a different sklearn build, and a policy artefact you cannot read is a
    policy artefact you cannot audit.
    """

    def __init__(self) -> None:
        self.names: list[str] = list(FEATS)
        self.w: list[float] = []
        self.b: float = 0.0
        self.mu: list[float] = []
        self.sd: list[float] = []
        self.iso_x: list[float] = []
        self.iso_y: list[float] = []
        self.meta: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return bool(self.w)

    def load(self, path: str | None = None) -> bool:
        p = pathlib.Path(path or MODEL_PATH)
        if not p.exists():
            log.warning("router_model_missing", path=str(p))
            return False
        d = json.loads(p.read_text())
        if d.get("feats") != self.names:
            log.warning("router_feature_mismatch", path=str(p))
            return False
        self.w = d["w"]
        self.b = d["b"]
        self.mu = d["mu"]
        self.sd = d["sd"]
        self.iso_x = d.get("iso_x", [])
        self.iso_y = d.get("iso_y", [])
        self.meta = d.get("meta", {})
        return True

    def _raw(self, x: list[float]) -> float:
        z = self.b
        for i, v in enumerate(x):
            sd = self.sd[i] if self.sd[i] else 1.0
            z += self.w[i] * ((v - self.mu[i]) / sd)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def _cal(self, p: float) -> float:
        """Isotonic calibration by linear interpolation over the fitted steps."""
        if not self.iso_x:
            return p
        if p <= self.iso_x[0]:
            return self.iso_y[0]
        if p >= self.iso_x[-1]:
            return self.iso_y[-1]
        lo, hi = 0, len(self.iso_x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.iso_x[mid] <= p:
                lo = mid
            else:
                hi = mid
        x0, x1 = self.iso_x[lo], self.iso_x[hi]
        y0, y1 = self.iso_y[lo], self.iso_y[hi]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

    def risk(self, f: dict[str, float]) -> float:
        """Calibrated probability that this response hides something worth finding."""
        if not self.ready:
            return 0.5      # untrained: no opinion, so let the tier floor decide
        return round(self._cal(self._raw(vec(f, self.names))), 6)


_R = Router()


def load(path: str | None = None) -> bool:
    return _R.load(path)


def ready() -> bool:
    return _R.ready


def meta() -> dict[str, Any]:
    return {"ready": _R.ready, "feats": len(_R.names), "logprobs": CAP_LOGPROBS,
            **_R.meta}


def risk(tr: Trace, t0: list[Finding] | None = None) -> float:
    return _R.risk(feats(tr, t0))


def tier(risk_v: float, pol: Policy, floor: int = 0) -> int:
    """Map a risk score to a verification tier using this tenant's thresholds.

    Thresholds live in the policy document, not here, so recalibration ships as
    a new policy version rather than a code change.
    """
    med = float(pol.thr.get("med", 0.3))
    high = float(pol.thr.get("high", 0.7))
    t = 2 if risk_v >= high else 1 if risk_v >= med else 0
    return max(t, floor)


def need_for(t: int) -> list[str]:
    """Which detectors a tier turns on.

    Tier 2 swaps nli for t2. They answer the same question -- is this grounded --
    but t2 decomposes the response into claims and retrieves evidence per claim,
    which is the expensive way and the accurate one.
    """
    if t >= 2:
        return ["t2", "safety", "bias"]
    if t >= 1:
        return ["nli", "safety", "bias"]
    return []
