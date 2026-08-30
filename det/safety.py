"""Safety via IBM Granite Guardian.

Substituted for Llama Guard 3 / ShieldGemma, both of which are gated=manual on
the Hub and cannot be downloaded without a licence accepted under a real
account. Granite Guardian is ungated and purpose-built for the same job.

It is a causal LM used as a classifier: build the guardian prompt, take one
forward pass, and read the Yes/No logits at the final position. No generation.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from det.batcher import Batcher
from det.schema import Finding

MODEL = os.getenv("SAFETY_MODEL", "ibm-granite/granite-guardian-3.0-2b")
DEV = os.getenv("DET_DEVICE_SAFETY") or os.getenv("DET_DEVICE", "cuda")
VER = "granite-guardian-3.0-2b-v1"
RISK = os.getenv("SAFETY_RISK", "harm")
THR = float(os.getenv("SAFETY_THR", "0.6"))
THR_HI = float(os.getenv("SAFETY_THR_HI", "0.9"))

_tok: Any = None
_mod: Any = None
_yes = _no = -1
_bat: Batcher[tuple[str, str, str], float] | None = None


def load() -> None:
    global _tok, _mod, _yes, _no, _bat
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    _yes = _tok.convert_tokens_to_ids("Yes")
    _no = _tok.convert_tokens_to_ids("No")
    if _yes is None or _no is None:
        raise RuntimeError("guardian tokenizer has no Yes/No ids")
    _fwd([("hello", "Refunds take three business days.", RISK)])  # warm cuda
    _bat = Batcher(_fwd, "safety")
    _bat.start()


def ready() -> bool:
    return _mod is not None


def info() -> dict[str, Any]:
    return {"model": MODEL, "ver": VER, "ready": ready(), "dev": DEV, "risk": RISK,
            "batch": dict(_bat.stat) if _bat else {}}


def _prompt(prompt: str, resp: str, risk: str) -> str:
    msgs = [{"role": "user", "content": prompt or ""},
            {"role": "assistant", "content": resp}]
    return _tok.apply_chat_template(
        msgs, guardian_config={"risk_name": risk},
        add_generation_prompt=True, tokenize=False,
    )


def _fwd(items: list[tuple[str, str, str]]) -> list[float]:
    txt = [_prompt(p, r, k) for p, r, k in items]
    # left padding so the final position is the real last token for every row
    b = _tok(txt, return_tensors="pt", padding=True, padding_side="left",
             truncation=True, max_length=2048).to(DEV)
    with torch.no_grad():
        lg = _mod(**b).logits[:, -1, :]
    p = torch.stack([lg[:, _yes], lg[:, _no]], dim=-1).float().softmax(-1).cpu()
    return [float(r[0]) for r in p]


async def check(resp: str, prompt: str = "") -> list[Finding]:
    if not resp.strip():
        return []
    assert _bat is not None
    p = await _bat.submit((prompt, resp, RISK))
    if p < THR:
        return []
    return [
        Finding(
            span=(0, len(resp)),
            dim="resp",
            label="unsafe",
            sev=3 if p >= THR_HI else 2,
            conf=round(p, 4),
            evid=f"risk:{RISK}",
            det=VER,
        )
    ]
