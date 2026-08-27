"""Shared vocabulary for the gateway. Frozen in phase 1.

det/serve.py mirrors Finding exactly. Everything downstream reads these.
"""

from typing import Literal

from pydantic import BaseModel, Field

Dim = Literal["perf", "resp", "cost"]
Act = Literal["allow", "annotate", "edit", "block", "escalate"]

# most-restrictive-wins ordering
RANK: dict[str, int] = {
    "allow": 0,
    "annotate": 1,
    "edit": 2,
    "block": 3,
    "escalate": 4,
}


class Finding(BaseModel):
    span: tuple[int, int]        # char offsets into the response text
    dim: Dim
    label: str                   # contradicted | unverifiable | pii | bias | unsafe | inject | cost_anom
    sev: int                     # 0..3
    conf: float                  # 0..1
    evid: str | None = None      # supporting doc id, or matched pattern name
    det: str                     # detector name + version, e.g. "nli-deberta-v1"


class Trace(BaseModel):
    id: str
    sess: str
    tenant: str                  # CS-BOT | KB-COPILOT | DECIDE
    geo: str                     # IN | EU | US
    ts: float
    prompt: str
    resp: str
    ctx: list[dict] = Field(default_factory=list)   # {id, text, score}
    tok_in: int = 0
    tok_out: int = 0
    tools: list[str] = Field(default_factory=list)
    cost: float = 0.0
    lat: dict = Field(default_factory=dict)         # {gen, t0, t1, t2, total} in ms
    tier: int = 0
    risk: float = 0.0            # router output - analysis only, never drives action
    fnd: list[Finding] = Field(default_factory=list)
    act: Act = "allow"
    pol_ver: str = ""
    ovr: dict | None = None      # reviewer override


class Policy(BaseModel):
    tenant: str
    geo: str
    ver: str
    effective_from: float
    lat_budget_ms: int
    thr: dict                    # {"med": 0.3, "high": 0.7}
    floors: dict                 # label -> Act, e.g. {"pii": "block"}
    escalate_if: dict            # {"sev": 3, "irrev_tool": true}
    sample: dict                 # {"t2": 0.05, "t3": 0.01}
    retention_days: int
