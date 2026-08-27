"""Mirror of api/schemas.py Finding. Kept byte-compatible on purpose.

Lives in its own module rather than in serve.py so the detector modules can
import it without a cycle back through the app. serve.py re-exports it.
"""

from typing import Literal

from pydantic import BaseModel

Dim = Literal["perf", "resp", "cost"]
Side = Literal["prompt", "resp"]


class Finding(BaseModel):
    span: tuple[int, int]
    side: Side = "resp"
    dim: Dim
    label: str
    sev: int
    conf: float
    evid: str | None = None
    det: str
