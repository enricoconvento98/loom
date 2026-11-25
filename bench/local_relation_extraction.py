"""Local entity and relation extraction on labeled service notes.

Compares extractors on one labeled corpus of synthetic service ownership and
dependency notes, with a fixed set of entity types (service, datastore, team,
person) and relation types (owned_by, depends_on). Offline: no Loom server and
no hosted provider. Each arm runs in its own process, so its peak memory and
cold start are its own.

Arms (`--arms`, comma-separated):

  writer  the labels themselves, as a writer passing `entities` and `relations`
          on `memory.set` supplies them: the reference, at no compute.
  rules   regular expressions written with the corpus templates. It checks the
          pipeline end to end; its scores say nothing about unseen notes.
  spans   a span model for entities and a relation model over their pairs,
          both zero-shot on the declared types (`--span-model`,
          `--relation-model`). Needs `pip install gliner glirel` and the model
          files, from the hub or from a local directory.
  chat    a local chat model behind an OpenAI-compatible endpoint
          (`--chat-base-url`, `--chat-model`), asked for the declared types as
          JSON. Its compute is the model server's, so memory is not reported.

Reported per arm: entity precision and recall (exact span and type), span
accuracy (exact boundaries among predictions that overlap a labeled entity of
their type), relation precision and recall (endpoints matched by overlap),
wrong-direction relations, corrections per note (the edits that turn the
candidates into the labels: a proxy for the time spent correcting them),
notes per second, CPU seconds per note, cold start and peak memory. Coverage
is the share of notes the arm answered; a note it skipped scores as no
candidates, and an arm whose model cannot be loaded is reported unavailable.

The corpus (`bench/fixtures/service_notes.jsonl`) holds plain statements,
inverted word order, several relations in one note, negated dependencies,
ambiguous names, and notes that state only relations outside the declared
set. `--write-fixture` regenerates it from the templates below.

    PYTHONPATH=$PWD .venv/bin/python bench/local_relation_extraction.py \
        --arms writer,rules,spans,chat --chat-base-url http://localhost:11434/v1 \
        --chat-model <model> --out bench/out/local_relation_extraction.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "service_notes.jsonl"

ENTITY_TYPES = ("service", "datastore", "team", "person")
# relation -> (head types, tail types)
RELATION_SIGNATURE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "owned_by": (("service", "datastore"), ("team", "person")),
    "depends_on": (("service",), ("service", "datastore")),
}
RELATION_TYPES = tuple(RELATION_SIGNATURE)
CATEGORIES = ("plain", "inverted", "multi", "negated", "ambiguous", "undeclared")


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    type: str

    def overlaps(self, other: Entity) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Relation:
    head: Entity
    type: str
    tail: Entity


@dataclass(frozen=True)
class Note:
    id: str
    category: str
    text: str
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


@dataclass
class Extraction:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


class ExtractorUnavailable(RuntimeError):
    """The arm's model cannot be loaded or reached; no note is extracted."""


class NoteSkipped(RuntimeError):
    """The arm gave no usable answer for one note."""


def signature_allows(rel: Relation) -> bool:
    heads, tails = RELATION_SIGNATURE[rel.type]
    return rel.head.type in heads and rel.tail.type in tails and rel.head != rel.tail


# --------------------------------------------------------------------------- corpus

_SERVICES = (
    "checkout-api auth-gateway billing-worker search-indexer inventory-api pricing-engine "
    "report-builder notification-service cart-service fraud-scorer image-resizer "
    "export-scheduler rate-limiter profile-api shipping-planner invoice-renderer"
).split()
_DATASTORES = (
    "ledger-db orders-postgres events-kafka session-redis metrics-clickhouse catalog-db "
    "audit-postgres queue-redis"
).split()
_TEAMS = [
    f"{n} team" for n in ("payments", "platform", "growth", "data infra", "storefront", "risk")
]
_PERSONS = (
    "Mara Lindqvist",
    "Tomas Okafor",
    "Ines Carvalho",
    "Devi Ranganathan",
    "Jonas Petrov",
    "Aiko Morland",
)
# One-word service names that are also ordinary words or team names.
_AMBIGUOUS = "search billing atlas beacon identity compass".split()
_FILLERS = "eu-west us-east Go Rust Kotlin staging".split()

# (category, template, relations as (head slot, type, tail slot)). A slot's first
# letter is its type: s service, d datastore, t team, p person, x filler (no entity).
# `{ts}` is "<the note's first service> team".
_TEMPLATES: tuple[tuple[str, str, tuple[tuple[str, str, str], ...]], ...] = (
    ("plain", "{s} is owned by the {t}.", (("s", "owned_by", "t"),)),
    ("plain", "{d} is owned by the {t}.", (("d", "owned_by", "t"),)),
    ("plain", "{s} belongs to the {t}.", (("s", "owned_by", "t"),)),
    ("plain", "Owner of {s}: {p}.", (("s", "owned_by", "p"),)),
    ("plain", "{s} depends on {s2}.", (("s", "depends_on", "s2"),)),
    ("plain", "{s} reads from {d} on every request.", (("s", "depends_on", "d"),)),
    ("plain", "{s} calls {s2} to authorise each request.", (("s", "depends_on", "s2"),)),
    ("plain", "{s} needs {d} to start.", (("s", "depends_on", "d"),)),
    ("inverted", "The {t} owns {s}.", (("s", "owned_by", "t"),)),
    ("inverted", "{p} is the owner of {s}.", (("s", "owned_by", "p"),)),
    ("inverted", "{s2} is a dependency of {s}.", (("s", "depends_on", "s2"),)),
    ("inverted", "{d} backs {s}.", (("s", "depends_on", "d"),)),
    ("inverted", "{s2} is called by {s} during checkout.", (("s", "depends_on", "s2"),)),
    (
        "multi",
        "{s} is owned by the {t} and depends on {d}.",
        (("s", "owned_by", "t"), ("s", "depends_on", "d")),
    ),
    (
        "multi",
        "The {t} owns {s}, which calls {s2} and reads from {d}.",
        (("s", "owned_by", "t"), ("s", "depends_on", "s2"), ("s", "depends_on", "d")),
    ),
    (
        "multi",
        "{s} depends on {s2} and {d}; {p} is the owner.",
        (("s", "depends_on", "s2"), ("s", "depends_on", "d"), ("s", "owned_by", "p")),
    ),
    ("negated", "{s} no longer depends on {s2}.", ()),
    ("negated", "{s} does not depend on {d}; the migration removed that call.", ()),
    ("negated", "The {t} is not the owner of {s}.", ()),
    ("negated", "{s} stopped reading from {d} after the cutover.", ()),
    (
        "negated",
        "{s} was moved away from the {t}; {p} is the owner now.",
        (("s", "owned_by", "p"),),
    ),
    ("ambiguous", "The {ts} owns {s}.", (("s", "owned_by", "ts"),)),
    ("ambiguous", "{s} depends on {s2}.", (("s", "depends_on", "s2"),)),
    ("ambiguous", "{s} is owned by the {t}, not the {ts}.", (("s", "owned_by", "t"),)),
    ("ambiguous", "{s} reads from {d}.", (("s", "depends_on", "d"),)),
    ("undeclared", "{s} is deployed to {x}.", ()),
    ("undeclared", "{s} is written in {x}.", ()),
    ("undeclared", "The {t} reviewed the {s} runbook.", ()),
    ("undeclared", "{p} asked about {d} retention.", ()),
    ("undeclared", "{s} and {s2} share a dashboard.", ()),
    ("undeclared", "The deploy window moves to Tuesday.", ()),
)
_SLOT_TYPE = {"s": "service", "d": "datastore", "t": "team", "p": "person"}
_SLOT_RE = re.compile(r"\{(\w+)\}")
_PER_CATEGORY = {
    "plain": 80,
    "inverted": 40,
    "multi": 30,
    "negated": 40,
    "ambiguous": 25,
    "undeclared": 25,
}


def _slot_values(category: str, slots: list[str], rng: random.Random) -> dict[str, str]:
    """A value for each slot, distinct within the note."""
    services = _AMBIGUOUS if category == "ambiguous" else _SERVICES
    pools = {"s": services, "d": _DATASTORES, "t": _TEAMS, "p": _PERSONS, "x": _FILLERS}
    values: dict[str, str] = {}
    for slot in slots:
        if slot == "ts":
            continue
        free = [v for v in pools[slot[0]] if v not in values.values()]
        values[slot] = rng.choice(free)
    if "ts" in slots:
        values["ts"] = f"{values['s']} team"
    return values


def render(category: str, template: str, relations: Any, rng: random.Random) -> Note:
    """One note from a template, with the character span of every entity slot."""
    slots = _SLOT_RE.findall(template)
    values = _slot_values(category, slots, rng)
    text, spans, at = "", {}, 0
    for m in _SLOT_RE.finditer(template):
        text += template[at : m.start()]
        slot, value = m.group(1), values[m.group(1)]
        if slot[0] in _SLOT_TYPE:
            spans[slot] = Entity(len(text), len(text) + len(value), _SLOT_TYPE[slot[0]])
        text += value
        at = m.end()
    text += template[at:]
    text = text[0].upper() + text[1:]
    rels = tuple(Relation(spans[h], t, spans[o]) for h, t, o in relations)
    entities = tuple(sorted(spans.values(), key=lambda e: e.start))
    return Note("", category, text, entities, rels)


def generate(seed: int = 7) -> list[Note]:
    """The corpus: each category's templates in turn, no text twice."""
    rng = random.Random(seed)
    notes: list[Note] = []
    seen: set[str] = set()
    for category, count in _PER_CATEGORY.items():
        templates = [t for t in _TEMPLATES if t[0] == category]
        made = attempts = 0
        while made < count and attempts < count * 50:
            _, template, relations = templates[attempts % len(templates)]
            attempts += 1
            note = render(category, template, relations, rng)
            if note.text in seen:
                continue
            seen.add(note.text)
            made += 1
            notes.append(
                Note(f"{category}-{made:03d}", category, note.text, note.entities, note.relations)
            )
    return notes


def note_to_json(note: Note) -> dict[str, Any]:
    index = {e: i for i, e in enumerate(note.entities)}
    return {
        "id": note.id,
        "category": note.category,
        "text": note.text,
        "entities": [
            {"start": e.start, "end": e.end, "type": e.type, "text": note.text[e.start : e.end]}
            for e in note.entities
        ],
        "relations": [
            {"head": index[r.head], "type": r.type, "tail": index[r.tail]} for r in note.relations
        ],
    }


def note_from_json(raw: dict[str, Any]) -> Note:
    entities = tuple(Entity(e["start"], e["end"], e["type"]) for e in raw["entities"])
    relations = tuple(
        Relation(entities[r["head"]], r["type"], entities[r["tail"]]) for r in raw["relations"]
    )
    return Note(raw["id"], raw["category"], raw["text"], entities, relations)


def load_notes(path: Path) -> list[Note]:
    return [note_from_json(json.loads(line)) for line in path.read_text().splitlines() if line]


def write_fixture(path: Path) -> int:
    notes = generate()
    path.write_text("".join(json.dumps(note_to_json(n)) + "\n" for n in notes))
    return len(notes)


# --------------------------------------------------------------------------- scoring


@dataclass
class Counts:
    notes: int = 0
    answered: int = 0
    entity_tp: int = 0
    entity_fp: int = 0
    entity_fn: int = 0
    span_exact: int = 0
    span_overlap: int = 0
    relation_tp: int = 0
    relation_fp: int = 0
    relation_fn: int = 0
    wrong_direction: int = 0
    corrections: int = 0

    def add(self, other: Counts) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _matches(pred: Relation, gold: Relation) -> bool:
    """Same relation type, each endpoint overlapping its labeled span."""
    return (
        pred.type == gold.type
        and pred.head.overlaps(gold.head)
        and pred.tail.overlaps(gold.tail)
        and pred.head.type == gold.head.type
        and pred.tail.type == gold.tail.type
    )


def score_note(note: Note, got: Extraction | None) -> Counts:
    """One note's counts; `got` None is a skipped note, scored as no candidates."""
    c = Counts(notes=1, answered=int(got is not None))
    got = got or Extraction()
    gold_entities, pred_entities = set(note.entities), set(got.entities)
    c.entity_tp = len(gold_entities & pred_entities)
    c.entity_fp = len(pred_entities - gold_entities)
    c.entity_fn = len(gold_entities - pred_entities)
    # A prediction that overlaps a labeled entity of its type found it; exact or not.
    missed = set(gold_entities - pred_entities)
    boundary_fixes = 0
    for p in sorted(pred_entities - gold_entities, key=lambda e: e.start):
        near = next((g for g in missed if g.type == p.type and g.overlaps(p)), None)
        if near is not None:
            missed.discard(near)
            boundary_fixes += 1
    c.span_exact = c.entity_tp
    c.span_overlap = c.entity_tp + boundary_fixes

    open_gold = list(dict.fromkeys(note.relations))
    unmatched: list[Relation] = []
    for rel in dict.fromkeys(got.relations):
        hit = next((g for g in open_gold if _matches(rel, g)), None)
        if hit is None:
            unmatched.append(rel)
        else:
            open_gold.remove(hit)
            c.relation_tp += 1
    c.relation_fp, c.relation_fn = len(unmatched), len(open_gold)
    for rel in unmatched:
        flipped = Relation(rel.tail, rel.type, rel.head)
        if any(_matches(flipped, g) for g in note.relations):
            c.wrong_direction += 1
    # One edit moves a boundary or flips a direction; the rest are adds and deletes.
    flips = min(c.wrong_direction, c.relation_fn)
    c.corrections = (
        c.entity_fp + c.entity_fn - boundary_fixes + c.relation_fp + c.relation_fn - flips
    )
    return c


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize(c: Counts) -> dict[str, Any]:
    return {
        "notes": c.notes,
        "coverage": _ratio(c.answered, c.notes),
        "entity_precision": _ratio(c.entity_tp, c.entity_tp + c.entity_fp),
        "entity_recall": _ratio(c.entity_tp, c.entity_tp + c.entity_fn),
        "span_accuracy": _ratio(c.span_exact, c.span_overlap),
        "relation_precision": _ratio(c.relation_tp, c.relation_tp + c.relation_fp),
        "relation_recall": _ratio(c.relation_tp, c.relation_tp + c.relation_fn),
        "relation_false_positives": c.relation_fp,
        "wrong_direction": c.wrong_direction,
        "corrections_per_note": _ratio(c.corrections, c.notes),
    }


# --------------------------------------------------------------------------- arms


class WriterArm:
    """The labels, as a writer who passes entities and relations supplies them."""

    in_process = True

    def __init__(self, notes: list[Note]) -> None:
        self._by_text = {n.text: n for n in notes}
        self.version = "labels"

    def load(self) -> None:
        return None

    def extract(self, text: str) -> Extraction:
        note = self._by_text[text]
        return Extraction(list(note.entities), list(note.relations))


_TEAM_RE = re.compile(r"\b[Tt]he ((?:[a-z]+ ){1,2}team)\b")
_PERSON_RE = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
_HYPHENATED_RE = re.compile(r"\b[A-Za-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
_DATASTORE_SUFFIXES = ("db", "postgres", "kafka", "redis", "clickhouse")
_NEGATION_RE = re.compile(r"\b(no longer|not|stopped|away from)\b")
# What stands between two entities -> (relation, whether the first one is the head).
_BETWEEN: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"\b(is owned by|belongs to)\b"), "owned_by", True),
    (re.compile(r"\b(owns|is the owner of)\b"), "owned_by", False),
    (re.compile(r"^: $"), "owned_by", True),
    (re.compile(r"\b(depends on|reads from|calls|needs)\b"), "depends_on", True),
    (re.compile(r"\b(is a dependency of|is called by|backs)\b"), "depends_on", False),
)


class RulesArm:
    """Patterns written with the templates: a check of the pipeline, not a model."""

    in_process = True
    version = "rules"

    def load(self) -> None:
        return None

    def _entities(self, text: str) -> list[Entity]:
        found = [Entity(m.start(1), m.end(1), "team") for m in _TEAM_RE.finditer(text)]
        found += [Entity(m.start(), m.end(), "person") for m in _PERSON_RE.finditer(text)]
        for m in _HYPHENATED_RE.finditer(text):
            kind = "datastore" if m.group().rsplit("-", 1)[1] in _DATASTORE_SUFFIXES else "service"
            found.append(Entity(m.start(), m.end(), kind))
        return sorted(found, key=lambda e: e.start)

    def extract(self, text: str) -> Extraction:
        entities = self._entities(text)
        relations: list[Relation] = []
        for first, second in zip(entities, entities[1:], strict=False):
            between = text[first.end : second.start]
            if _NEGATION_RE.search(between):
                continue
            for pattern, rel_type, first_is_head in _BETWEEN:
                if pattern.search(between):
                    head, tail = (first, second) if first_is_head else (second, first)
                    relation = Relation(head, rel_type, tail)
                    if signature_allows(relation):
                        relations.append(relation)
                    break
        return Extraction(entities, relations)


# The relation model's own tokenizer, so token positions agree with it.
_TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")
_RELATION_LABELS = {"owned by": "owned_by", "depends on": "depends_on"}


class SpansArm:
    """A span model for entities, then a relation model over the entity pairs."""

    in_process = True

    def __init__(self, span_model: str, relation_model: str, threshold: float) -> None:
        self._span_id, self._relation_id, self._threshold = span_model, relation_model, threshold
        self.version = f"{span_model} + {relation_model}"
        self._spans: Any = None
        self._relations: Any = None

    def load(self) -> None:
        try:
            import gliner
            import glirel

            self._spans = gliner.GLiNER.from_pretrained(self._span_id)
            self._relations = glirel.GLiREL.from_pretrained(self._relation_id)
        except Exception as e:  # noqa: BLE001 — a missing package or model is a reported outcome
            raise ExtractorUnavailable(f"{type(e).__name__}: {e}") from e
        self.version += f" (gliner {gliner.__version__}, glirel {glirel.__version__})"

    def extract(self, text: str) -> Extraction:
        found = self._spans.predict_entities(text, list(ENTITY_TYPES), threshold=self._threshold)
        entities = [Entity(int(e["start"]), int(e["end"]), str(e["label"])) for e in found]
        tokens = list(_TOKEN_RE.finditer(text))
        by_position: dict[tuple[int, int], Entity] = {}
        ner: list[list[Any]] = []
        for e in entities:
            inside = [i for i, t in enumerate(tokens) if t.start() >= e.start and t.end() <= e.end]
            if inside:
                by_position[(inside[0], inside[-1] + 1)] = e
                ner.append([inside[0], inside[-1], e.type, text[e.start : e.end]])
        relations: list[Relation] = []
        if len(ner) >= 2:
            scored = self._relations.predict_relations(
                [t.group() for t in tokens],
                list(_RELATION_LABELS),
                threshold=self._threshold,
                ner=ner,
                top_k=1,
            )
            for r in scored:
                head = by_position.get((r["head_pos"][0], r["head_pos"][1]))
                tail = by_position.get((r["tail_pos"][0], r["tail_pos"][1]))
                if head is None or tail is None:
                    continue
                relation = Relation(head, _RELATION_LABELS[r["label"]], tail)
                if signature_allows(relation) and relation not in relations:
                    relations.append(relation)
        return Extraction(entities, relations)


_CHAT_PROMPT = """\
Extract entities and relations from the note. Use only these types.

Entity types: service, datastore, team, person.
Relation types:
  owned_by: head is a service or datastore, tail is the team or person that owns it now.
  depends_on: head is a service, tail is the service or datastore it depends on now.

Copy each entity's text exactly as it appears in the note. Leave out a relation the note
negates or states as ended, and any relation of another type. Return only JSON:
{"entities": [{"text": "...", "type": "..."}],
 "relations": [{"head": "...", "type": "...", "tail": "..."}]}
"""


class ChatArm:
    """A local chat model behind an OpenAI-compatible endpoint."""

    in_process = False

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model, self._timeout_s = model, timeout_s
        self.version = model
        self._client: Any = None

    def _ask(self, text: str) -> str:
        body = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _CHAT_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        key = os.environ.get("OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        reply = self._client.post(self._url, json=body, headers=headers)
        reply.raise_for_status()
        return str(reply.json()["choices"][0]["message"]["content"])

    def load(self) -> None:
        if not self._model:
            raise ExtractorUnavailable("no --chat-model")
        try:
            import httpx

            self._client = httpx.Client(timeout=self._timeout_s)
            self._ask("checkout-api depends on ledger-db.")
        except Exception as e:  # noqa: BLE001 — an unreachable model server is a reported outcome
            raise ExtractorUnavailable(f"{type(e).__name__}: {e}") from e

    def extract(self, text: str) -> Extraction:
        try:
            return parse_chat_reply(text, self._ask(text))
        except Exception as e:  # noqa: BLE001 — one bad reply skips its note, not the run
            raise NoteSkipped(f"{type(e).__name__}: {e}") from e


def parse_chat_reply(text: str, reply: str) -> Extraction:
    """The reply's JSON as spans of `text`; an entity not found in the note is dropped."""
    data = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
    by_surface: dict[str, Entity] = {}
    taken: list[Entity] = []
    lowered = text.lower()
    for raw in data.get("entities") or []:
        surface, kind = str(raw.get("text", "")), str(raw.get("type", ""))
        if not surface or kind not in ENTITY_TYPES or surface.lower() in by_surface:
            continue
        at = lowered.find(surface.lower())
        while at >= 0:
            entity = Entity(at, at + len(surface), kind)
            if not any(entity.overlaps(t) for t in taken):
                by_surface[surface.lower()] = entity
                taken.append(entity)
                break
            at = lowered.find(surface.lower(), at + 1)
    relations: list[Relation] = []
    for raw in data.get("relations") or []:
        head = by_surface.get(str(raw.get("head", "")).lower())
        tail = by_surface.get(str(raw.get("tail", "")).lower())
        if head is None or tail is None or raw.get("type") not in RELATION_SIGNATURE:
            continue
        relation = Relation(head, str(raw["type"]), tail)
        if signature_allows(relation) and relation not in relations:
            relations.append(relation)
    return Extraction(sorted(taken, key=lambda e: e.start), relations)


ARMS = ("writer", "rules", "spans", "chat")


def build_arm(name: str, notes: list[Note], args: argparse.Namespace) -> Any:
    if name == "writer":
        return WriterArm(notes)
    if name == "rules":
        return RulesArm()
    if name == "spans":
        return SpansArm(args.span_model, args.relation_model, args.threshold)
    if name == "chat":
        return ChatArm(args.chat_base_url, args.chat_model, args.chat_timeout_s)
    raise SystemExit(f"unknown arm {name!r}; choose from {', '.join(ARMS)}")


# --------------------------------------------------------------------------- running


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def run_arm(name: str, notes: list[Note], args: argparse.Namespace) -> dict[str, Any]:
    """Load one arm and score every note; called in the arm's own process."""
    arm = build_arm(name, notes, args)
    started = time.perf_counter()
    try:
        arm.load()
    except ExtractorUnavailable as e:
        return {"arm": name, "available": False, "reason": str(e), "coverage": 0.0}
    load_s = time.perf_counter() - started
    total, by_category = Counts(), {c: Counts() for c in CATEGORIES}
    first_s = wall_s = 0.0
    cpu_started = time.process_time()
    skipped: list[str] = []
    for i, note in enumerate(notes):
        t0 = time.perf_counter()
        try:
            got: Extraction | None = arm.extract(note.text)
        except NoteSkipped as e:
            got = None
            skipped.append(f"{note.id}: {e}"[:200])
        took = time.perf_counter() - t0
        wall_s += took
        if i == 0:
            first_s = took
        counts = score_note(note, got)
        total.add(counts)
        by_category[note.category].add(counts)
    cpu_s = time.process_time() - cpu_started
    out: dict[str, Any] = {"arm": name, "available": True, "version": arm.version}
    out.update(summarize(total))
    out["by_category"] = {c: summarize(v) for c, v in by_category.items() if v.notes}
    out["skipped"] = skipped
    out["notes_per_s"] = round(len(notes) / wall_s, 1) if wall_s else None
    out["cold_start_s"] = round(load_s + first_s, 3)
    # A chat arm's compute is the model server's, not this process's.
    out["cpu_s_per_note"] = round(cpu_s / len(notes), 5) if arm.in_process and notes else None
    out["peak_rss_mb"] = _peak_rss_mb() if arm.in_process else None
    return out


def _child_argv(name: str, result_path: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--arm",
        name,
        "--result-json",
        result_path,
        "--dataset",
        str(args.dataset),
        "--span-model",
        args.span_model,
        "--relation-model",
        args.relation_model,
        "--threshold",
        str(args.threshold),
        "--chat-base-url",
        args.chat_base_url,
        "--chat-model",
        args.chat_model,
        "--chat-timeout-s",
        str(args.chat_timeout_s),
    ]


def run_arms(names: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Each arm in a fresh interpreter, so memory and cold start are its own."""
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            path = str(Path(tmp) / f"{name}.json")
            done = subprocess.run(_child_argv(name, path, args), check=False)
            if done.returncode != 0 or not Path(path).exists():
                raise SystemExit(f"arm {name!r} failed with exit code {done.returncode}")
            results.append(json.loads(Path(path).read_text()))
    return results


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if not isinstance(value, float):
        return str(value)
    if value > 1:
        return f"{value:,.1f}"
    return f"{value:.3f}" if value >= 0.001 or value == 0 else f"{value:.1e}"


_COLUMNS = (
    ("arm", "arm"),
    ("coverage", "coverage"),
    ("entity P", "entity_precision"),
    ("entity R", "entity_recall"),
    ("span acc", "span_accuracy"),
    ("rel P", "relation_precision"),
    ("rel R", "relation_recall"),
    ("wrong dir", "wrong_direction"),
    ("fixes/note", "corrections_per_note"),
    ("notes/s", "notes_per_s"),
    ("cpu s/note", "cpu_s_per_note"),
    ("cold start s", "cold_start_s"),
    ("peak MB", "peak_rss_mb"),
)


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(title for title, _ in _COLUMNS) + " |",
        "|" + " --- |" * len(_COLUMNS),
    ]
    for r in results:
        if not r["available"]:
            lines.append(f"| {r['arm']} | 0 | unavailable: {r['reason'][:120]} |")
            continue
        lines.append("| " + " | ".join(_cell(r.get(key)) for _, key in _COLUMNS) + " |")
    lines += ["", "Relation false positives by category:", ""]
    lines += ["| arm | " + " | ".join(CATEGORIES) + " |", "|" + " --- |" * (len(CATEGORIES) + 1)]
    for r in results:
        if r["available"]:
            cells = [
                _cell(r["by_category"].get(c, {}).get("relation_false_positives"))
                for c in CATEGORIES
            ]
            lines.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hardware() -> dict[str, Any]:
    return {
        "machine": platform.machine(),
        "system": platform.platform(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=FIXTURE)
    parser.add_argument("--arms", default="writer,rules")
    parser.add_argument("--span-model", default="urchade/gliner_small-v2.1")
    parser.add_argument("--relation-model", default="jackboyla/glirel-large-v0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chat-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--chat-model", default="")
    parser.add_argument("--chat-timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    parser.add_argument("--write-fixture", action="store_true", help="regenerate --dataset")
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.write_fixture:
        print(f"wrote {write_fixture(args.dataset)} notes -> {args.dataset}")
        return 0
    if args.arm:
        result = run_arm(args.arm, load_notes(args.dataset), args)
        Path(args.result_json).write_text(json.dumps(result))
        return 0
    results = run_arms([a.strip() for a in args.arms.split(",") if a.strip()], args)
    print(render_report(results))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"dataset": str(args.dataset), "hardware": hardware(), "arms": results}
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
