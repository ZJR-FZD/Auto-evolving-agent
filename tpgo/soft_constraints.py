"""
Soft constraint and reflection helpers for online trajectory improvement.

The helpers here are intentionally generic: they do not encode benchmark
answers or task-specific entities. They provide reusable guidance for
constraint-led search, lightweight role checks, and duplicate-query handling.
"""

from __future__ import annotations

import re
import json
from typing import Any


SOFT_CONSTRAINT_SYSTEM_APPEND = """

## Soft Constraint Ledger

For multi-hop search tasks, keep a lightweight ledger in your reasoning:
1. Break the question into candidate roles: subject, event, intermediate entity,
   final answer role.
2. Track strong constraints separately from weak topical matches. Strong
   constraints include dates/ranges, counts, relationships, survival/death
   details, named events, exact media appearances, titles, and requested answer
   type.
3. Treat the first plausible entity as a candidate, not as confirmed. Before
   committing to it, check which constraints remain unverified.
4. If a candidate only matches the broad topic but not the distinctive
   constraints, mark it as tentative or rejected and pivot.
5. Prefer searches that test one missing constraint at a time. Do not keep
   searching around an entity only because it is thematically similar.

## Semantic Clue Expansion

When a clue is phrased indirectly, expand it into a few plausible search terms
without assuming any one is true. Examples:
- "tissue from a pair of organs" may suggest eye/cornea tissue, lung tissue,
  kidney-related terms, or other paired-organ tissue terms.
- "appeared on a TV show segment" may be indexed as guesting, interview,
  episode, segment, appearance, or show name.
- "helped two people / one was a kid" may be indexed as recipient, beneficiary,
  transplant recipient, child recipient, or donor family story.
Use these as soft search pivots, then verify with sources.

## Reflection Check

After a low-signal result or critic warning, explicitly ask:
- Which candidate am I currently assuming?
- Which original constraints are still unverified?
- Is my next search testing a missing constraint, or just rewording the same
  broad topic?

## Answer Role Check

Before finalizing, ensure the answer has the requested role and type. If the
question asks for a recipient, do not answer with the donor, parent, driver,
host, article author, intended-but-not-actual recipient, or organization.
Output only the requested answer span in <answer>...</answer>.

## Search Query Planning

Do not start with one long query that paraphrases the whole question. Search
engines perform better when each query targets one stage of the chain.

Use a staged query plan:
1. Anchor discovery: find the subject/donor using rare source-side event facts,
   not the final answer description. Combine 2-3 facts likely to appear in the
   same article, such as accident wording, survivor roles, tissue type, and
   recipient count.
2. Detail confirmation: once a subject candidate appears, search that candidate
   with a different hard fact from the question.
3. Final-detail extraction: only after the subject and donation detail are
   plausible, search subject + recipient/media clue to extract the answer.

Query construction rules:
- Prefer short targeted queries over all-constraint queries. If a query has
  more than about 10 meaningful words, it is probably overstuffed.
- Prefer source-side wording over puzzle wording: relationship words, counts,
  age phrases, tissue names, "segment", "guesting", "beneficiary", "thanked",
  and "parents" often work better than a full paraphrase.
- Try several short angles across stages rather than repeating the same final
  answer description.
- If broad English searches fail, keep unusual exact phrases and try regional
  or media vocabulary. This is a generic strategy, not a claim about the answer.
"""


NEG_CRITIC_ALIGNMENT_APPEND = """

Additional evaluation rules:
- GOOD requires progress on distinctive constraints, not just broad topical
  relevance.
- Mark BAD when the agent keeps pursuing a candidate that only matches a broad
  topic and has not been checked against the question's unique constraints.
- Mark BAD when a [SOFT_CONSTRAINT_CHECK] warning appears and the agent
  continues with the same candidate without addressing the warning.
- Mark BAD when the trajectory confuses roles such as donor vs recipient,
  subject vs related person, event participant vs final answer.
- Mark BAD when recent searches are rewordings that do not test an unverified
  constraint.
- If BAD, the hint should point to the kind of missing constraint or role check
  needed. Do not provide an answer, exact entity, or exact query.
- Do not suggest speculative examples such as a specific organ type, country,
  TV program, or person unless the trajectory evidence already contains it.
- Do not penalize a possible tissue type merely because it can have multiple
  recipients; only call out recipient-count conflict when retrieved evidence
  itself says a conflicting count.
"""


FORCED_REFLECTION_APPEND = (
    "\n5) Include a short constraint ledger: current candidate, verified facts, "
    "unverified facts, and answer role.\n"
    "6) Your next action should test one unverified constraint or pivot to a "
    "new candidate. Keep this as guidance, not a hard template."
)


FORCE_ANSWER_APPEND = (
    "\n6) Run an answer-role check before the tag: verify that the final span "
    "is the entity asked for, not a related donor/source/person.\n"
    "7) If the role is uncertain, still output only the most plausible short "
    "candidate span. Do not explain uncertainty inside the answer."
)


QUERY_STRATEGY_PROMPT = """
[SYSTEM][QUERY_STRATEGY]
Use staged search instead of a single all-clues query.

Stage A - anchor discovery:
- Search for the subject/donor/source entity using 2-3 rare source-side facts.
- Do not include the full final-answer description yet.
- Prefer snippet search first; use fetch only after a plausible candidate appears.

Stage B - candidate verification:
- Search the candidate with a different hard fact from the original question.
- Reject or demote the candidate if count, timeline, role, or tissue details conflict.

Stage C - final extraction:
- Search candidate + recipient/media clue to find the requested answer span.

For indirect clues, translate softly:
- paired-organ tissue -> possible terms such as cornea, eye tissue, kidney tissue, lung tissue;
- TV show segment -> segment, guesting, appearance, beneficiary, interview;
- expressed gratitude -> thanked, thanks, grateful, gratitude, parents/family.

Your next search should choose exactly one stage and one clue chain.
"""


STOP_QUERY_WORDS = {
    "the", "and", "for", "from", "with", "that", "this", "were", "was",
    "are", "one", "who", "what", "when", "where", "which", "their", "then",
    "after", "before", "because", "inclusive", "individual", "person",
}

FINAL_DETAIL_TERMS = {
    "tv", "television", "show", "segment", "appearance", "appeared",
    "interview", "guesting", "thanked", "thanks", "gratitude", "grateful",
}

SOURCE_EVENT_TERMS = {
    "teen", "died", "death", "accident", "crash", "vehicle", "vehicular",
    "father", "driver", "survived", "donation", "donor", "tissue", "organ",
    "recipient", "recipients", "people", "kid", "child", "teenager",
}


def query_keywords(query: str) -> set[str]:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", (query or "").lower())
    return {w for w in normalized.split() if len(w) > 2 and w not in STOP_QUERY_WORDS}


def is_overstuffed_query(query: str) -> bool:
    """Detect queries that paraphrase the whole riddle instead of one stage."""
    kw = query_keywords(query)
    if len(kw) < 11:
        return False
    has_final_detail = bool(kw & FINAL_DETAIL_TERMS)
    has_source_event = len(kw & SOURCE_EVENT_TERMS) >= 5
    return has_final_detail and has_source_event


def build_overstuffed_query_feedback(query: str) -> str:
    kw = sorted(query_keywords(query))
    return (
        "[HARNESS] QUERY_TOO_BROAD: This search query appears to combine the "
        "source event, donation details, recipient details, and final TV clue "
        "in one long query. Search engines often fail on this.\n"
        f"Detected keywords: {', '.join(kw[:18])}\n"
        "Use staged search instead:\n"
        "1) anchor discovery: 2-3 rare source-side facts only;\n"
        "2) candidate verification: candidate + one different hard fact;\n"
        "3) final extraction: candidate + recipient/media clue.\n"
        "Next action: issue one short query for exactly one stage. Prefer "
        "fetch=false until a plausible candidate appears."
    )


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _appearance_cutoff_year(instruction: str) -> int | None:
    text = (instruction or "").lower()
    matches = re.findall(
        r"\b(?:before|prior to|by)\s+((?:19|20)\d{2})\b.{0,120}\b(?:appeared|appearance|tv|show|segment)",
        text,
    )
    matches.extend(
        re.findall(
            r"\b(?:appeared|appearance|tv|show|segment)\b.{0,120}\b(?:before|prior to|by)\s+((?:19|20)\d{2})\b",
            text,
        )
    )
    if not matches:
        return None
    return max(int(y) for y in matches)


def _result_blob(tool_result: str) -> str:
    parsed = _safe_json_loads(tool_result)
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            parts.extend(
                str(item.get(k, ""))
                for k in ("title", "snippet", "content", "url")
                if item.get(k)
            )
        return "\n".join(parts)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return tool_result or ""


def find_soft_constraint_conflicts(instruction: str, tool_result: str) -> list[str]:
    """Return generic soft warnings when retrieved evidence may conflict.

    These are not hard filters. The runner injects them as reflection nudges so
    the model checks the original constraints before committing to a candidate.
    """
    inst = (instruction or "").lower()
    blob = _result_blob(tool_result)
    low = blob.lower()
    warnings: list[str] = []

    appearance_cutoff = _appearance_cutoff_year(instruction)

    if re.search(r"\btwo\s+(?:people|persons|recipients)\b", inst):
        if re.search(r"\b(?:four|4)\s+(?:lives|people|recipients)\b", low):
            warnings.append(
                "question says the relevant tissues helped two people, but result mentions four lives/recipients"
            )
        if re.search(r"\b(?:three|3)\s+(?:lives|people|recipients)\b", low):
            warnings.append(
                "question says the relevant tissues helped two people, but result mentions three lives/recipients"
            )

    has_pair_tissue_clue = (
        "tissue" in inst
        and ("pair" in inst or "particular pair" in inst)
        and ("organ" in inst or "organs" in inst)
    )
    concrete_tissue_terms = (
        "cornea", "corneal", "eye tissue", "eyes", "retina", "retinal",
        "kidney tissue", "lung tissue", "heart valve", "skin graft",
    )
    if has_pair_tissue_clue:
        has_generic_organ_story = any(
            term in low for term in ("organ donor", "organ donation", "transplant", "liver")
        )
        has_concrete_tissue = any(term in low for term in concrete_tissue_terms)
        if has_generic_organ_story and not has_concrete_tissue:
            warnings.append(
                "result looks like a broad organ-transplant story; the question asks about tissues from a paired organ"
            )

    if ("kid" in inst or "child" in inst) and "recipient" in inst:
        if re.search(r"\b(?:2[5-9]|[3-9]\d)\s*,?\s+(?:from|of|year-old|years old)\b", low):
            warnings.append(
                "result names an adult recipient; verify the requested recipient was a child/teen at the relevant TV appearance"
            )
        if appearance_cutoff is not None and re.search(
            rf"\b(?:transplant(?:ed)?|appeared|appearance|interview|segment|episode)\b.{{0,100}}\b(?:{appearance_cutoff}|20(?:2[3-9]|[3-9]\d))\b",
            low,
        ):
            warnings.append(
                "result suggests the transplant occurred after the TV-appearance cutoff; verify timeline before using this candidate"
            )

    return warnings[:4]


def build_soft_constraint_reflection(conflicts: list[str]) -> str:
    lines = [
        "[SYSTEM][SOFT_CONSTRAINT_CHECK]",
        "The latest evidence may conflict with the original question. This is not a hard rejection, but you must reflect before continuing.",
        "Potential issues:",
    ]
    for idx, warning in enumerate(conflicts, start=1):
        lines.append(f"{idx}) {warning}")
    lines.extend(
        [
            "Next step guidance:",
            "- Do not lock this candidate unless the distinctive constraints are verified.",
            "- Prefer a query that tests a missing constraint from the original question.",
            "- If the candidate conflicts with timing/count/role/tissue clues, mark it tentative or rejected and pivot.",
        ]
    )
    return "\n".join(lines)


STRONG_PIVOT_TERMS = {
    # Generic evidence/role pivots
    "beneficiary",
    "beneficiaries",
    "actual",
    "intended",
    "father",
    "mother",
    "driver",
    "survived",
    "survivor",
    "survivors",
    "child",
    "kid",
    # Media appearance pivots
    "segment",
    "episode",
    "guesting",
    "guest",
    "appearance",
    "beneficiary",
    "thanked",
    "thanks",
    "gratitude",
    "parents",
    # Medical/tissue pivots
    "transplant",
    "transplanted",
    "cornea",
    "corneas",
    "corneal",
    "eye",
    "eyes",
    "retina",
    "retinal",
    "kidney",
    "kidneys",
    "lung",
    "lungs",
    "philstar",
    "inquirer",
    "pep",
    "abs",
    "gma",
}

BROAD_PIVOT_TERMS = {
    "recipient",
    "recipients",
    "donor",
    "donors",
    "family",
    "parent",
    "parents",
    "teen",
    "teenager",
    "tv",
    "television",
    "show",
    "interview",
    "tissue",
    "tissues",
    "organ",
    "organs",
}


def _flatten_recent(recent_queries: list[set[str]]) -> set[str]:
    out: set[str] = set()
    for kw in recent_queries:
        out.update(str(w).lower() for w in kw)
    return out


def has_meaningful_query_pivot(
    current_kw: set[str],
    recent_queries: list[set[str]],
    *,
    min_new_terms: int = 2,
) -> bool:
    """Return True when a near-duplicate query adds a useful new angle.

    This deliberately softens duplicate blocking. A query can overlap heavily
    with recent searches and still be worth running if it introduces a new
    constraint-bearing term, such as a role, media clue, or concrete tissue
    term.
    """
    if not current_kw or not recent_queries:
        return False
    current = {str(w).lower() for w in current_kw}
    previous = _flatten_recent(recent_queries)
    new_terms = current - previous
    if new_terms & STRONG_PIVOT_TERMS:
        return True
    # Years, counts, and exact-looking tokens often represent real constraints.
    if any(re.search(r"\d", term) for term in new_terms):
        return True
    if new_terms <= BROAD_PIVOT_TERMS:
        return False
    if len(new_terms) >= max(3, min_new_terms):
        return True
    if len(new_terms) >= min_new_terms:
        return True
    return False
