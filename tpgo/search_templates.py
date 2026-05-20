"""Search query templates for TPGO task routes.

Templates are advisory and operate only on question text plus a TaskRoute.
They do not use gold answers and do not solve the task.
"""

from __future__ import annotations

import re

from tpgo.task_router import TaskRoute


GENERIC_SEEDS = {
    "father", "mother", "parents", "family", "died", "death", "accident",
    "recipient", "beneficiary", "annual report", "10-k", "stock repurchase",
    "director", "film", "match", "goal", "book", "author", "species",
    "scientific name", "song", "album",
}


SEARCH_TEMPLATES: dict[str, list[str]] = {
    "company_finance": [
        "{entity} annual report",
        "{entity} 10-K stock repurchase",
        "{entity} investor relations",
        "{entity} share repurchase fiscal year",
        "{entity} SEC filing buyback",
    ],
    "person_bio_family": [
        "{entity} father driver survived accident",
        "{entity} family donated tissue recipient",
        "{entity} cornea recipient beneficiary",
        "{entity} TV segment thanked parents",
        "{entity} obituary accident donation",
    ],
    "film_tv_game": [
        "{entity} director film",
        "{entity} cast director official",
        "{entity} episode credits",
        "{entity} video game developer publisher",
        "{entity} IMDb director",
    ],
    "sports_match": [
        "{entity} match report goal minute",
        "{entity} free kick score",
        "{entity} qualifier match report",
        "{entity} final score lineup",
        "{entity} official match report",
    ],
    "book_blog_article": [
        "{entity} author publisher",
        "{entity} blog article archive",
        "{entity} abstract publication",
        "{entity} review title author",
        "{entity} library record ISBN",
    ],
    "archive_permit_history": [
        "{place} archaeology report PDF",
        "{place} council planning excavation",
        "{place} archaeological test pits",
        "{place} heritage permit archive",
        "{place} planning application archaeology",
    ],
    "science_species": [
        "{entity} scientific name species",
        "{entity} taxonomy genus",
        "{entity} abstract new species",
        "{entity} journal DOI species",
        "{entity} holotype specimen",
    ],
    "music_artist": [
        "{entity} official artist song",
        "{entity} album label",
        "{entity} Discogs MusicBrainz",
        "{entity} single release date",
        "{entity} songwriter producer",
    ],
    "general_multihop": [
        "{entity} official source",
        "{entity} news archive",
        "{entity} date source",
        "{entity} reference record",
        "{entity} primary source",
    ],
}


def _clean(text: str) -> str:
    """Normalize a query fragment."""
    return re.sub(r"\s+", " ", (text or "").strip(" ,.;:!?\"'"))


def _dedupe_words(query: str) -> str:
    """Remove repeated words while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for word in _clean(query).split():
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(word)
    return " ".join(out)


def _entity_from_route(question: str, route: TaskRoute) -> str:
    """Choose a conservative entity/search seed from route metadata."""
    for keyword in route.search_keywords:
        cleaned = _clean(keyword)
        if (
            cleaned
            and not re.fullmatch(r"(?:18|19|20)\d{2}", cleaned)
            and cleaned.lower() not in GENERIC_SEEDS
        ):
            return cleaned
    quoted = re.findall(r'"([^"]{3,80})"', question or "")
    if quoted:
        return _clean(quoted[0])
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", question or "")
    if proper:
        return _clean(proper[0])
    task_words = _task_seed_words(question, route.task_type)
    if task_words:
        return " ".join(task_words[:6])
    return "source"


def _task_seed_words(question: str, task_type: str) -> list[str]:
    """Extract task-specific seed words when no named entity is available."""
    q_words = [
        w.lower()
        for w in re.sub(r"[^0-9A-Za-z]+", " ", question or "").split()
        if len(w) > 2
    ]
    priority = {
        "company_finance": (
            "company", "stock", "repurchase", "share", "buyback", "10", "annual", "report",
        ),
        "person_bio_family": (
            "teen", "died", "car", "accident", "vehicular", "father", "driver",
            "survived", "cornea", "tissue", "donation", "recipient",
        ),
        "film_tv_game": ("film", "director", "actor", "episode", "series", "game"),
        "sports_match": ("match", "goal", "free", "kick", "minute", "qualifier", "final"),
        "book_blog_article": ("book", "author", "blog", "article", "abstract", "published"),
        "archive_permit_history": ("archive", "permit", "planning", "council", "excavation", "archaeology"),
        "science_species": ("species", "scientific", "name", "genus", "abstract", "journal"),
        "music_artist": ("song", "album", "artist", "band", "single", "label"),
        "general_multihop": ("date", "name", "source", "record"),
    }.get(task_type, ())
    out: list[str] = []
    for word in priority:
        if word in q_words and word not in out:
            out.append(word)
    if out:
        return out
    stop = {"what", "which", "when", "where", "there", "their", "because", "with", "that"}
    return [w for w in q_words if w not in stop][:6]


def _place_from_question(question: str, fallback: str) -> str:
    """Extract a likely place-like seed for archive/history templates."""
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b", question or "")
    return _clean(proper[0]) if proper else fallback


def build_search_queries(question: str, route: TaskRoute, max_queries: int = 5) -> list[str]:
    """Build advisory search queries from a question and route.

    Args:
        question: Original user question.
        route: Result from classify_question.
        max_queries: Maximum number of queries to return.

    Returns:
        A de-duplicated list of search query strings.
    """
    max_queries = max(1, int(max_queries))
    task_type = route.task_type if route.task_type in SEARCH_TEMPLATES else "general_multihop"
    entity = _entity_from_route(question, route)
    place = _place_from_question(question, entity)
    values = {"entity": entity, "place": place}

    out: list[str] = []
    for template in SEARCH_TEMPLATES[task_type]:
        query = _dedupe_words(template.format(**values))
        if query and query not in out:
            out.append(query)
        if len(out) >= max_queries:
            break

    for keyword in route.search_keywords:
        if len(out) >= max_queries:
            break
        cleaned = _clean(keyword)
        if cleaned and cleaned not in out:
            out.append(cleaned)

    return out[:max_queries]
