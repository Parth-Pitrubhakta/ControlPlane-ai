"""Bias detection via Granite Guardian's social_bias risk.

The spec named a Protect bias adapter or a BBQ-tuned DeBERTa; neither exists on
the Hub in a loadable form. Two encoder classifiers were measured first and both
failed on a 17-sentence probe:

  valurank/distilroberta-bias      benign max 0.9985, biased min 0.0376
  Sentence-Level-Stereotype-Detector  benign max 0.9280, biased min 0.9525

The first detects loaded news language rather than social bias (it scored
"final sale and cannot be returned" at 0.9985); the second leaves no usable gap.
Granite Guardian, already loaded for safety, separates the same probe cleanly:
benign max 0.0006 against biased min 0.1176. So bias reuses that model under a
different risk definition rather than adding a worse one.

Scored per sentence: the batch is nearly free on a launch-bound model, and it
puts the finding on the offending clause instead of the whole reply.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from det.batcher import Batcher
from det.schema import Finding

MODEL = os.getenv("BIAS_MODEL", "ibm-granite/granite-guardian-3.0-2b")
DEV = os.getenv("DET_DEVICE_BIAS") or os.getenv("DET_DEVICE", "cuda")
VER = "granite-guardian-social-bias-v1"
RISK = "social_bias"
# probe: benign max 0.0006, biased min 0.1176. 0.05 sits clear of both.
THR = float(os.getenv("BIAS_THR", "0.05"))
THR_HI = float(os.getenv("BIAS_THR_HI", "0.8"))
MAXLEN = int(os.getenv("BIAS_MAXLEN", "1024"))

_tok: Any = None
_mod: Any = None
_yes = _no = -1
_bat: Batcher[str, float] | None = None


def load() -> None:
    global _tok, _mod, _yes, _no, _bat
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _mod = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    _yes = _tok.convert_tokens_to_ids("Yes")
    _no = _tok.convert_tokens_to_ids("No")
    if _yes is None or _no is None:
        raise RuntimeError("guardian tokenizer has no Yes/No ids")
    _fwd(["Refunds take three business days."])   # warm cuda
    _bat = Batcher(_fwd, "bias")
    _bat.start()


def ready() -> bool:
    return _mod is not None


def info() -> dict[str, Any]:
    return {"model": MODEL, "ver": VER, "ready": ready(), "dev": DEV, "risk": RISK,
            "batch": dict(_bat.stat) if _bat else {}}


def _prompt(seg: str) -> str:
    msgs = [{"role": "user", "content": ""}, {"role": "assistant", "content": seg}]
    return _tok.apply_chat_template(
        msgs, guardian_config={"risk_name": RISK},
        add_generation_prompt=True, tokenize=False,
    )


def _fwd(segs: list[str]) -> list[float]:
    txt = [_prompt(s) for s in segs]
    b = _tok(txt, return_tensors="pt", padding=True, padding_side="left",
             truncation=True, max_length=MAXLEN).to(DEV)
    with torch.no_grad():
        lg = _mod(**b).logits[:, -1, :]
    p = torch.stack([lg[:, _yes], lg[:, _no]], dim=-1).float().softmax(-1).cpu()
    return [float(r[0]) for r in p]


async def check(resp: str, spans: list[tuple[int, int]] | None = None) -> list[Finding]:
    if not resp.strip():
        return []
    sp = [s for s in (spans if spans is not None else [(0, len(resp))])
          if resp[s[0]:s[1]].strip()]
    if not sp:
        return []
    assert _bat is not None
    res = await _bat.submit_many([resp[a:b] for a, b in sp])
    out: list[Finding] = []
    for (a, b), p in zip(sp, res):
        if p < THR:
            continue
        out.append(
            Finding(span=(a, b), dim="resp", label="bias",
                    sev=2 if p >= THR_HI else 1, conf=round(p, 4),
                    evid=f"risk:{RISK}", det=VER)
        )
    return out
