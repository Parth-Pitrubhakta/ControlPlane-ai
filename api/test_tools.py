from api import tools


def test_read_only_verbs():
    for n in ("crm.contact.read", "kb.search", "orders.list", "get_invoice",
              "lookup-customer", "describe_policy"):
        assert tools.cls(n) == "ro", n


def test_irreversible_verbs():
    for n in ("billing.refund", "account.delete", "email.send", "payments.transfer",
              "deploy_release", "subscription.cancel", "user.ban"):
        assert tools.cls(n) == "irrev", n


def test_reversible_verbs():
    for n in ("ticket.create", "contact.update", "note.write", "draft_reply"):
        assert tools.cls(n) == "rev", n


def test_unknown_tool_is_assumed_to_have_a_side_effect():
    assert tools.cls("frobnicate_widget") == "rev"


def test_has_irrev_and_all_ro():
    assert tools.has_irrev(["kb.search", "billing.refund"])
    assert not tools.has_irrev(["kb.search", "ticket.create"])
    assert tools.all_ro(["kb.search", "orders.list"])
    assert not tools.all_ro(["kb.search", "ticket.create"])
    assert not tools.all_ro([])


def test_read_beats_write_when_both_words_appear():
    # "read" is checked first on purpose: a read-only lookup on a write API
    # should not be classified as a mutation
    assert tools.cls("orders.read_after_write") == "ro"
