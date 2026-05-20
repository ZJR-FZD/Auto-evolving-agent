"""Tests for the lightweight TPGO task router."""

from __future__ import annotations

from tpgo.search_templates import build_search_queries
from tpgo.task_router import classify_question


def test_stock_repurchase_routes_to_company_finance() -> None:
    """Stock repurchase questions should use finance routing."""
    route = classify_question(
        "Which company disclosed the largest stock repurchase in its 2021 10-K annual report?"
    )
    assert route.task_type == "company_finance"
    assert "insufficient_primary_evidence" in route.risk_tags
    queries = build_search_queries("Acme stock repurchase 10-K annual report", route)
    assert any("10-K" in q or "annual report" in q for q in queries)


def test_cornea_parent_crash_routes_to_person_bio_family() -> None:
    """Donation/family/accident questions should use person biography routing."""
    route = classify_question(
        "A teen died in a vehicular accident, his father and driver survived, "
        "and his parents donated corneas to two recipients. Which child recipient "
        "later appeared on a TV segment and thanked the family?"
    )
    assert route.task_type == "person_bio_family"
    assert "ambiguous_name_collision" in route.risk_tags
    assert "person name" in route.answer_format_hint


def test_movie_director_routes_to_film_tv_game() -> None:
    """Director questions should route to film/TV/game."""
    route = classify_question("Who directed the 1998 film whose lead actor later appeared in the TV series?")
    assert route.task_type == "film_tv_game"


def test_match_goal_free_kick_routes_to_sports_match() -> None:
    """Match, goal, minute, and free-kick clues should route to sports."""
    route = classify_question(
        "In which match did a player score a 95th-minute free-kick goal in a qualifier?"
    )
    assert route.task_type == "sports_match"
    assert "numeric_extraction_error" in route.risk_tags


def test_species_abstract_prefers_science_over_article() -> None:
    """Species/scientific-name questions with abstract should prefer science."""
    route = classify_question(
        "According to the abstract of the 2014 paper, what is the scientific name of the new species?"
    )
    assert route.task_type == "science_species"
    assert route.confidence >= 0.78
