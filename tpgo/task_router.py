"""Rule-based task routing for multi-hop web-search QA.

This module is meta-control only. It classifies a question into a broad
search task type and returns search guidance, risk tags, and answer-format
hints. It does not call external APIs, read gold answers, or solve the task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SUPPORTED_TASK_TYPES = (
    "company_finance",
    "person_bio_family",
    "film_tv_game",
    "sports_match",
    "book_blog_article",
    "archive_permit_history",
    "science_species",
    "music_artist",
    "general_multihop",
)


@dataclass(frozen=True)
class TaskRoute:
    """Routing metadata for a question.

    Attributes are intentionally advisory. Downstream agents may use them to
    choose search templates or source preferences, but must still perform
    evidence-based retrieval before answering.
    """

    task_type: str
    confidence: float
    search_keywords: list[str]
    preferred_sources: list[str]
    risk_tags: list[str]
    answer_format_hint: str
    notes: str


def _norm(text: str) -> str:
    """Normalize text for simple rule matching."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _hits(text: str, patterns: tuple[str, ...]) -> int:
    """Count regex/substring pattern hits in normalized text."""
    total = 0
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], text, flags=re.IGNORECASE):
                total += 1
        elif pattern in text:
            total += 1
    return total


def _extract_keywords(question: str, task_type: str, limit: int = 10) -> list[str]:
    """Extract short, non-answer search seed terms from the question."""
    q = _norm(question)
    quoted = re.findall(r'"([^"]{3,80})"', question or "")
    years = re.findall(r"\b(?:18|19|20)\d{2}\b", question or "")
    proper_chunks = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", question or "")

    domain_terms = {
        "company_finance": (
            "annual report", "10-k", "stock repurchase", "share repurchase",
            "investor relations", "fiscal year",
        ),
        "person_bio_family": (
            "father", "mother", "parents", "family", "died", "death",
            "accident", "cornea", "recipient", "beneficiary",
        ),
        "film_tv_game": (
            "film", "director", "cast", "episode", "television", "game",
            "developer", "producer",
        ),
        "sports_match": (
            "match", "goal", "free kick", "score", "minute", "final",
            "qualifier", "team",
        ),
        "book_blog_article": (
            "book", "author", "blog", "article", "essay", "published",
            "review", "abstract",
        ),
        "archive_permit_history": (
            "archive", "permit", "planning", "council", "excavation",
            "archaeology", "heritage",
        ),
        "science_species": (
            "species", "scientific name", "abstract", "genus", "taxonomy",
            "journal", "doi",
        ),
        "music_artist": (
            "song", "album", "artist", "band", "single", "lyrics",
            "label", "producer",
        ),
        "general_multihop": ("date", "name", "source", "record"),
    }

    seeds: list[str] = []
    for bucket in (quoted, years, proper_chunks, domain_terms.get(task_type, ())):
        for item in bucket:
            term = re.sub(r"\s+", " ", str(item).strip())
            if not term:
                continue
            if term.lower() in {"what", "which", "who", "when", "where"}:
                continue
            if term not in seeds:
                seeds.append(term)
            if len(seeds) >= limit:
                return seeds
    return seeds


def _answer_format_hint(question: str, task_type: str) -> str:
    """Infer the expected answer shape from the question wording."""
    q = _norm(question)
    if "first and last name" in q or "first name" in q or "last name" in q:
        return "person name; output first and last name only when possible"
    if task_type == "person_bio_family" and (
        q.startswith("who ") or " which child " in q or "recipient" in q
    ):
        return "person name; verify the role before output"
    if "company" in q or task_type == "company_finance":
        return "company name, ticker, numeric amount, fiscal year, or short finance fact as requested"
    if "scientific name" in q:
        return "binomial scientific name"
    if "species" in q and task_type == "science_species":
        return "species name or scientific name"
    if "score" in q or task_type == "sports_match":
        return "team/player name, scoreline, minute, or match fact as requested"
    if "director" in q:
        return "person name"
    if "title" in q:
        return "exact title"
    if task_type == "music_artist":
        return "artist, song, album, label, or date as requested"
    return "concise answer span matching the requested entity or fact"


def _preferred_sources(task_type: str) -> list[str]:
    """Return source preferences for a route."""
    return {
        "company_finance": [
            "company investor relations",
            "SEC EDGAR 10-K/10-Q",
            "annual report PDF",
            "exchange filings",
        ],
        "person_bio_family": [
            "reputable news articles",
            "official family/foundation pages",
            "TV network or program pages",
            "newspaper archives",
        ],
        "film_tv_game": [
            "IMDb",
            "official film/TV/game pages",
            "publisher/developer pages",
            "reputable entertainment databases",
        ],
        "sports_match": [
            "official league pages",
            "club/team pages",
            "match reports",
            "sports statistics databases",
        ],
        "book_blog_article": [
            "publisher pages",
            "author pages",
            "journal/blog archives",
            "library or ISBN records",
        ],
        "archive_permit_history": [
            "council planning portals",
            "archive catalogues",
            "heritage reports",
            "archaeology PDFs",
        ],
        "science_species": [
            "journal article pages",
            "PubMed/PMC",
            "Biodiversity Heritage Library",
            "GBIF/Catalogue of Life/NCBI taxonomy",
        ],
        "music_artist": [
            "official artist pages",
            "label pages",
            "Discogs/MusicBrainz",
            "chart databases",
        ],
        "general_multihop": [
            "primary sources",
            "official pages",
            "reputable news",
            "reference databases",
        ],
    }[task_type]


def _risk_tags(question: str, task_type: str) -> list[str]:
    """Infer likely retrieval/extraction risks from the question and route."""
    q = _norm(question)
    tags: list[str] = []
    if any(word in q for word in ("same name", "not to be confused", "also known", "alias")):
        tags.append("ambiguous_name_collision")
    if re.search(r"\b(before|after|between|prior to|inclusive|later|earlier)\b", q):
        tags.append("temporal_constraint_mismatch")
    if re.search(r"\b\d+(?:st|nd|rd|th)?\b", q):
        tags.append("numeric_extraction_error")
    if any(word in q for word in ("first and last", "exact", "format", "only")):
        tags.append("answer_format_error")
    if task_type in {"company_finance", "science_species", "archive_permit_history"}:
        tags.append("insufficient_primary_evidence")
    if task_type in {"person_bio_family", "film_tv_game", "music_artist"}:
        tags.append("ambiguous_name_collision")
    if len(q.split()) > 70:
        tags.append("overlong_reasoning_chain")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def classify_question(question: str) -> TaskRoute:
    """Classify a question into a broad search task route.

    The implementation is rule-based and deterministic. It only inspects the
    question text and returns advisory routing metadata.
    """
    q = _norm(question)
    if not q:
        return TaskRoute(
            task_type="general_multihop",
            confidence=0.2,
            search_keywords=[],
            preferred_sources=_preferred_sources("general_multihop"),
            risk_tags=["source_retrieval_failure"],
            answer_format_hint="concise answer span matching the requested entity or fact",
            notes="empty question; fallback route",
        )

    rules: dict[str, tuple[str, ...]] = {
        "company_finance": (
            "stock repurchase", "share repurchase", "buyback", "10-k",
            "annual report", "investor relations", "fiscal year", "sec filing",
            "revenue", "dividend", "shares outstanding",
        ),
        "person_bio_family": (
            "father", "mother", "parents", "family", "spouse", "children",
            "died", "death", "accident", "vehicular", "survived", "donated",
            "cornea", "recipient", "beneficiary", "first and last name",
        ),
        "film_tv_game": (
            "film", "movie", "director", "screenplay", "actor", "actress",
            "episode", "tv show", "television series", "video game",
            "game director", "developer", "publisher",
        ),
        "sports_match": (
            "match", "goal", "free kick", "free-kick", "penalty", "score",
            "minute", "qualifier", "final", "semi-final", "team", "club",
            "league", "tournament",
        ),
        "book_blog_article": (
            "book", "novel", "author", "publisher", "isbn", "blog",
            "article", "essay", "review", "abstract", "chapter", "magazine",
        ),
        "archive_permit_history": (
            "archive", "permit", "planning", "council", "excavation",
            "archaeological", "archaeology", "test pits", "heritage",
            "historic", "listed building",
        ),
        "science_species": (
            "species", "scientific name", "genus", "taxonomy", "taxon",
            "abstract", "doi", "journal", "specimen", "holotype",
            "phylogenetic",
        ),
        "music_artist": (
            "song", "album", "artist", "band", "single", "lyrics",
            "record label", "composer", "producer", "singer", "track",
        ),
    }

    scores = {task_type: _hits(q, patterns) for task_type, patterns in rules.items()}

    # Priority override: species/scientific-name questions can contain
    # "abstract", but should route to science before article/blog.
    if scores["science_species"] > 0 and any(term in q for term in ("species", "scientific name", "genus", "taxonomy")):
        task_type = "science_species"
    else:
        task_type = max(scores, key=lambda key: (scores[key], key))
        if scores[task_type] == 0:
            task_type = "general_multihop"

    score = scores.get(task_type, 0)
    confidence = min(0.95, 0.35 + 0.15 * score) if task_type != "general_multihop" else 0.35
    if task_type == "science_species" and scores["science_species"] >= 2:
        confidence = max(confidence, 0.78)

    return TaskRoute(
        task_type=task_type,
        confidence=round(confidence, 3),
        search_keywords=_extract_keywords(question, task_type),
        preferred_sources=_preferred_sources(task_type),
        risk_tags=_risk_tags(question, task_type),
        answer_format_hint=_answer_format_hint(question, task_type),
        notes="rule-based route; advisory only; do not answer without retrieved evidence",
    )
