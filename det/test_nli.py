"""Three-valued grounding logic. Run with the cp-vllm env (imports torch).

The forward pass is stubbed: these tests pin the decision rules, not the model.
"""

import pytest

from det import nli


class FakeBat:
    """Stands in for the batcher, returning scripted (p_entail, p_contra)."""

    def __init__(self, out):
        self.out = out
        self.seen = []

    async def submit_many(self, pairs):
        self.seen = pairs
        return self.out[: len(pairs)]


@pytest.fixture
def stub(monkeypatch):
    def go(scores):
        b = FakeBat(scores)
        monkeypatch.setattr(nli, "_bat", b)
        monkeypatch.setattr(nli, "TOPK", 0)
        return b
    return go


CTX = [{"id": "warranty.md#1", "text": "Electronics carry a warranty of 3 years."}]


def test_sents_offsets_are_exact():
    t = "First one. Second one! Third?"
    sp = nli.sents(t)
    assert [t[a:b] for a, b in sp] == ["First one.", "Second one!", "Third?"]


def test_sents_ignores_whitespace_and_stubs():
    assert nli.sents("   ") == []
    assert nli.sents("A. .. B.") == [(0, 2), (6, 8)]


def test_sents_handles_missing_final_punctuation():
    t = "No trailing period here"
    assert [t[a:b] for a, b in nli.sents(t)] == [t]


async def test_no_context_is_unverifiable_never_contradicted():
    f = await nli.check("The centre opens at 9am. It closes at 6pm.", [])
    assert len(f) == 2
    assert {x.label for x in f} == {"unverifiable"}
    assert all(x.sev == 1 and x.dim == "perf" for x in f)


async def test_contradiction_above_threshold(stub):
    stub([(0.01, 0.99)])
    f = await nli.check("Electronics carry a warranty of 5 years.", CTX)
    assert len(f) == 1
    assert f[0].label == "contradicted"
    assert f[0].sev == 2
    assert f[0].evid == "warranty.md#1"


async def test_entailment_above_threshold_emits_nothing(stub):
    stub([(0.95, 0.01)])
    assert await nli.check("Electronics carry a warranty of 3 years.", CTX) == []


async def test_middle_ground_is_unverifiable(stub):
    stub([(0.4, 0.3)])
    f = await nli.check("Electronics are covered.", CTX)
    assert len(f) == 1 and f[0].label == "unverifiable" and f[0].sev == 1


async def test_contradiction_beats_entailment_when_corpus_disagrees(stub):
    # one chunk entails, another contradicts: the conservative reading wins
    ctx = CTX + [{"id": "wiki.md#1", "text": "Electronics carry a warranty of 5 years."}]
    stub([(0.95, 0.01), (0.01, 0.99)])
    f = await nli.check("Electronics carry a warranty of 3 years.", ctx)
    assert len(f) == 1
    assert f[0].label == "contradicted"
    assert f[0].evid == "wiki.md#1"


async def test_evidence_points_at_the_right_chunk(stub):
    ctx = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"},
           {"id": "c", "text": "gamma"}]
    stub([(0.1, 0.2), (0.1, 0.95), (0.1, 0.3)])
    f = await nli.check("One sentence only.", ctx)
    assert f[0].evid == "b"


async def test_spans_index_the_response(stub):
    t = "Fine sentence. Electronics carry a warranty of 5 years."
    stub([(0.9, 0.01), (0.01, 0.99)])
    f = await nli.check(t, CTX)
    assert len(f) == 1
    a, b = f[0].span
    assert t[a:b] == "Electronics carry a warranty of 5 years."


async def test_unverifiable_never_exceeds_sev_1(stub):
    # invariant 2: unverifiable is capped at annotate, so it must never carry
    # the severity that would let a policy escalate it to a block
    stub([(0.0, 0.0)] * 3)
    t = "One thing. Two things. Three things."
    for f in await nli.check(t, CTX):
        if f.label == "unverifiable":
            assert f.sev == 1


async def test_empty_response_yields_nothing():
    assert await nli.check("", CTX) == []


async def test_pairs_are_premise_context_hypothesis_sentence(stub):
    b = stub([(0.5, 0.1)])
    await nli.check("A claim here.", CTX)
    assert b.seen == [(CTX[0]["text"], "A claim here.")]
