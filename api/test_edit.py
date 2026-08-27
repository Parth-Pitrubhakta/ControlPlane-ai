"""The four edits. The key property is that nothing here can invent text."""

import re

from api import edit
from api.schemas import Finding


def f(label, span, sev=2, evid=None, side="resp", dim=None):
    d = dim or ("perf" if label in ("contradicted", "unverifiable") else "resp")
    return Finding(span=span, side=side, dim=d, label=label, sev=sev,
                   conf=0.9, evid=evid, det="test-v1")


def test_redacts_a_pii_span():
    t = "Your card 4111 1111 1111 1111 is on file."
    a = t.index("4111")
    out, ops = edit.run(t, [f("pii", (a, a + 19), evid="card")], "edit")
    assert "4111" not in out
    assert edit.REDACTION in out
    assert ops[0]["op"] == "redact"


def test_deletes_a_contradicted_sentence():
    t = "Refunds take three days. The warranty is 5 years. Track it in the app."
    a = t.index("The warranty")
    b = t.index(" Track")
    out, _ = edit.run(t, [f("contradicted", (a, b), evid="warranty.md")], "edit")
    assert "5 years" not in out
    assert "Refunds take three days." in out
    assert "Track it in the app." in out


def test_appends_a_citation_for_the_contradicting_doc():
    t = "The warranty is 5 years."
    out, _ = edit.run(t, [f("contradicted", (0, len(t)), evid="warranty.md")], "edit")
    assert "[source: warranty.md]" in out


def test_appends_a_hedge_for_unverifiable():
    t = "Our Bengaluru centre opens at 9am."
    out, _ = edit.run(t, [f("unverifiable", (0, len(t)), sev=1)], "edit")
    assert out.startswith(t)
    assert "could not be verified" in out


def test_nothing_happens_unless_the_action_is_edit():
    t = "Your card 4111 1111 1111 1111 is on file."
    for act in ("allow", "annotate", "block", "escalate"):
        out, ops = edit.run(t, [f("pii", (10, 29))], act)
        assert out == t and ops == []


def test_prompt_side_findings_are_never_edited():
    t = "A clean response."
    out, ops = edit.run(t, [f("pii", (0, 5), side="prompt")], "edit")
    assert out == t
    assert not [o for o in ops if o["op"] == "redact"]


def test_overlapping_spans_do_not_corrupt_offsets():
    t = "Card 4111 1111 1111 1111 and mail a@b.example here."
    fs = [f("pii", (5, 24), evid="card"), f("pii", (34, 45), evid="email")]
    out, ops = edit.run(t, list(reversed(fs)), "edit")
    assert "4111" not in out and "a@b.example" not in out
    assert out.count(edit.REDACTION) == 2


def test_out_of_range_span_is_ignored_not_fatal():
    t = "Short."
    out, ops = edit.run(t, [f("pii", (100, 200))], "edit")
    assert out == t


def test_edit_is_idempotent():
    t = "The warranty is 5 years."
    fs = [f("contradicted", (0, len(t)), evid="warranty.md")]
    once, _ = edit.run(t, fs, "edit")
    twice, _ = edit.run(once, [], "edit")
    assert twice == once


def test_edit_is_deterministic():
    t = "Card 4111 1111 1111 1111 on file. The warranty is 5 years."
    fs = [f("pii", (5, 24), evid="card"),
          f("contradicted", (34, len(t)), evid="warranty.md")]
    runs = {edit.run(t, fs, "edit")[0] for _ in range(20)}
    assert len(runs) == 1


def test_output_never_contains_invented_words():
    """The non-generative property, stated as a test.

    Every word in the output must come from the input or from the module's
    frozen constants. If this ever fails, something is generating text.
    """
    t = ("Refunds take three business days. Your card 4111 1111 1111 1111 is on "
         "file. The warranty is 5 years. We open at 9am on Saturdays.")
    fs = [f("pii", (t.index("4111"), t.index("4111") + 19), evid="card"),
          f("contradicted", (t.index("The warranty"), t.index(" We open")),
            evid="warranty.md"),
          f("unverifiable", (t.index("We open"), len(t)), sev=1)]
    out, _ = edit.run(t, fs, "edit")

    allowed = set(re.findall(r"[\w']+", t))
    allowed |= set(re.findall(r"[\w']+", edit.REDACTION + edit.HEDGE))
    allowed |= set(re.findall(r"[\w']+", edit.CITE_OPEN + "warranty.md" + edit.CITE_CLOSE))
    produced = set(re.findall(r"[\w']+", out))
    assert produced <= allowed, f"invented: {produced - allowed}"


def test_plan_is_pure_and_reports_what_it_will_do():
    fs = [f("pii", (0, 4)), f("contradicted", (5, 9), evid="d1"),
          f("unverifiable", (10, 14), sev=1)]
    ops = edit.plan(fs, "edit")
    assert [o["op"] for o in ops] == ["redact", "delete", "cite", "hedge"]


def test_deleting_everything_is_flagged_rather_than_dressed_up():
    t = "The warranty is 5 years."
    out, ops = edit.run(t, [f("contradicted", (0, len(t)), evid="warranty.md")], "edit")
    assert edit.emptied(ops), "an edit that removes the whole answer must say so"
    assert not out.startswith(" ")


def test_partial_delete_is_not_flagged_as_emptied():
    t = "Refunds take three days. The warranty is 5 years."
    a = t.index("The warranty")
    out, ops = edit.run(t, [f("contradicted", (a, len(t)), evid="warranty.md")], "edit")
    assert not edit.emptied(ops)
    assert "Refunds take three days." in out
