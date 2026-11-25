"""Corpus, scoring and reply parsing of bench/local_relation_extraction.py (no model)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import local_relation_extraction as lre

E = lre.Entity


def _note(text: str, entities: list[lre.Entity], relations: list[lre.Relation]) -> lre.Note:
    return lre.Note("n", "plain", text, tuple(entities), tuple(relations))


TEXT = "checkout-api depends on ledger-db."
API, DB = E(0, 12, "service"), E(24, 33, "datastore")
GOLD = _note(TEXT, [API, DB], [lre.Relation(API, "depends_on", DB)])


def test_the_committed_fixture_is_what_the_templates_generate() -> None:
    generated = [lre.note_to_json(n) for n in lre.generate()]
    committed = [json.loads(line) for line in lre.FIXTURE.read_text().splitlines()]
    assert committed == generated


def test_every_label_is_a_span_of_its_note_and_fits_the_declared_types() -> None:
    notes = lre.load_notes(lre.FIXTURE)
    assert {n.category for n in notes} == set(lre.CATEGORIES)
    assert len({n.text for n in notes}) == len(notes)
    for raw, note in zip(lre.FIXTURE.read_text().splitlines(), notes, strict=True):
        for entity in json.loads(raw)["entities"]:
            assert note.text[entity["start"] : entity["end"]] == entity["text"]
            assert entity["type"] in lre.ENTITY_TYPES
        assert all(lre.signature_allows(r) for r in note.relations)
        if note.category == "undeclared":
            assert not note.relations


def test_exact_candidates_need_no_correction() -> None:
    c = lre.score_note(GOLD, lre.Extraction(list(GOLD.entities), list(GOLD.relations)))
    assert (c.entity_tp, c.relation_tp, c.corrections) == (2, 1, 0)
    assert lre.summarize(c)["span_accuracy"] == 1.0


def test_a_reversed_relation_is_one_wrong_direction_and_one_correction() -> None:
    reversed_ = lre.Relation(E(0, 12, "datastore"), "depends_on", E(24, 33, "service"))
    wrong_types = lre.score_note(GOLD, lre.Extraction([], [reversed_]))
    assert wrong_types.wrong_direction == 0, "endpoint types differ: not the labeled relation"

    two_services = _note(
        "cart-service depends on rate-limiter.",
        [E(0, 12, "service"), E(24, 36, "service")],
        [lre.Relation(E(0, 12, "service"), "depends_on", E(24, 36, "service"))],
    )
    flipped = lre.Relation(E(24, 36, "service"), "depends_on", E(0, 12, "service"))
    c = lre.score_note(two_services, lre.Extraction(list(two_services.entities), [flipped]))
    assert (c.relation_fp, c.relation_fn, c.wrong_direction, c.corrections) == (1, 1, 1, 1)


def test_a_wrong_boundary_costs_span_accuracy_not_the_relation() -> None:
    short = E(0, 8, "service")
    got = lre.Extraction([short, DB], [lre.Relation(short, "depends_on", DB)])
    c = lre.score_note(GOLD, got)
    assert (c.entity_tp, c.entity_fp, c.entity_fn) == (1, 1, 1)
    assert (c.span_exact, c.span_overlap, c.relation_tp, c.corrections) == (1, 2, 1, 1)


def test_a_skipped_note_scores_as_no_candidates_and_lowers_coverage() -> None:
    c = lre.score_note(GOLD, None)
    assert (c.answered, c.entity_fn, c.relation_fn, c.corrections) == (0, 2, 1, 3)
    assert lre.summarize(c)["coverage"] == 0.0


def test_the_rules_arm_drops_a_negated_dependency() -> None:
    arm = lre.RulesArm()
    assert arm.extract("Checkout-api no longer depends on ledger-db.").relations == []
    kept = arm.extract("The payments team owns checkout-api.").relations
    assert [(r.head.type, r.type, r.tail.type) for r in kept] == [("service", "owned_by", "team")]


def test_a_chat_reply_becomes_spans_and_drops_what_the_note_does_not_hold() -> None:
    reply = (
        'Here you go: {"entities": [{"text": "Checkout-API", "type": "service"},'
        ' {"text": "ledger-db", "type": "datastore"}, {"text": "billing", "type": "service"},'
        ' {"text": "ledger-db", "type": "region"}],'
        ' "relations": [{"head": "checkout-api", "type": "depends_on", "tail": "ledger-db"},'
        ' {"head": "ledger-db", "type": "depends_on", "tail": "checkout-api"},'
        ' {"head": "checkout-api", "type": "deployed_to", "tail": "ledger-db"}]}'
    )
    got = lre.parse_chat_reply(TEXT, reply)
    assert got.entities == [API, DB]
    assert got.relations == [lre.Relation(API, "depends_on", DB)]


def test_an_arm_whose_model_cannot_be_loaded_is_reported_unavailable() -> None:
    args = argparse.Namespace(chat_base_url="http://127.0.0.1:9", chat_model="", chat_timeout_s=1)
    result = lre.run_arm("chat", [GOLD], args)
    assert result == {
        "arm": "chat",
        "available": False,
        "reason": "no --chat-model",
        "coverage": 0.0,
    }
    assert "unavailable: no --chat-model" in lre.render_report([result])
