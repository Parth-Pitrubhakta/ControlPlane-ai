"""Tier 2 claim handling. Run with the cp-vllm env (imports torch).

The parser and the span locator are pure, and both have already broken once:
the parser silently turned tier 2 into a slower tier 1, and a claim whose span
cannot be placed would make the UI highlight the wrong words.
"""

import pytest

from det import t2


# ------------------------------------------------------------------ parsing

def test_parses_one_array_per_line():
    """What Qwen actually returns. A greedy [.*] match spans all of these and
    parses as nothing, which is how tier 2 silently degraded to sentences."""
    out = ('["Unopened items may be returned within 30 days."]\n'
           '["Refunds are issued within 3 business days."]\n'
           '["Standard delivery takes 3 to 5 business days."]')
    got = t2._parse_claims(out)
    assert len(got) == 3
    assert got[0].startswith("Unopened items")


def test_parses_a_single_array():
    assert t2._parse_claims('["One claim.", "Two claims."]') == ["One claim.", "Two claims."]


def test_parses_a_fenced_array():
    got = t2._parse_claims('```json\n["One claim.", "Two claims."]\n```')
    assert got == ["One claim.", "Two claims."]


def test_falls_back_to_bullets():
    got = t2._parse_claims("- The warranty is three years.\n- Refunds take three days.")
    assert len(got) == 2
    assert got[0] == "The warranty is three years."


def test_ignores_empty_arrays():
    assert t2._parse_claims("[]") == []


def test_never_raises_on_junk():
    for junk in ("", "I cannot help with that", "[[[", '{"a": 1}'):
        assert isinstance(t2._parse_claims(junk), list)


# ----------------------------------------------------------------- locating

RESP = ("Thank you for contacting us. Electronics carry a warranty of 3 years. "
        "Refunds are issued within 3 business days.")


def test_locates_a_verbatim_claim():
    sp = t2._locate(RESP, "Electronics carry a warranty of 3 years.")
    assert sp and RESP[sp[0]:sp[1]] == "Electronics carry a warranty of 3 years."


def test_locates_despite_whitespace_differences():
    sp = t2._locate(RESP, "Electronics  carry a warranty   of 3 years.")
    assert sp is not None
    assert "warranty" in RESP[sp[0]:sp[1]]


def test_falls_back_to_the_overlapping_sentence_when_paraphrased():
    sp = t2._locate(RESP, "Refunds issued business days within three")
    assert sp is not None
    assert "Refunds" in RESP[sp[0]:sp[1]]


def test_returns_none_when_the_claim_is_not_in_the_response():
    """Better no finding than one pointing at the wrong text: an edit would
    delete words the model never wrote."""
    assert t2._locate(RESP, "The moon is made of cheese and orbits Jupiter") is None


def test_returns_none_for_an_empty_claim():
    assert t2._locate(RESP, "   ") is None


def test_located_spans_are_always_inside_the_response():
    for c in ("Electronics carry a warranty of 3 years.",
              "Refunds are issued within 3 business days.",
              "Thank you for contacting us."):
        sp = t2._locate(RESP, c)
        if sp:
            assert 0 <= sp[0] < sp[1] <= len(RESP)


# ------------------------------------------------------------------ wiring

async def test_no_context_means_unverifiable_never_contradicted(monkeypatch):
    """Invariant 2 holds in tier 2 exactly as it does in tier 1."""
    async def fake(resp):
        return [("Electronics carry a warranty of 3 years.", (28, 68))]
    monkeypatch.setattr(t2, "decompose", fake)
    out = await t2.check(RESP, [])
    assert out and all(f.label == "unverifiable" and f.sev == 1 for f in out)


async def test_empty_response_yields_nothing():
    assert await t2.check("", [{"id": "d", "text": "x"}]) == []
