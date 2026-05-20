"""Generic distilled search strategy guidance.

This module is static prompt text only. It does not call a stronger model, does
not inspect benchmark answers, and does not persist memory. The guidance is
generic meta-control for text web-search tasks.
"""

from __future__ import annotations


DISTILLED_32B_STRATEGY_PROMPT = """
[SYSTEM][DISTILLED_SEARCH_STRATEGY]
This is generic search-process guidance only. Do not answer from this block.

Search discipline:
1. First identify the requested answer role and answer shape: person, title,
   organization, number/date, place, species, match fact, or other short span.
2. Split the question into 2-3 clue chains. Start with the rarest source-side
   facts, not with a full paraphrase of the question.
3. Use short staged queries:
   - anchor discovery: identify the main subject/candidate;
   - constraint check: test one distinctive missing constraint;
   - extraction: search candidate plus the final answer role.
4. Treat the first plausible entity as tentative. Verify it against a date,
   count, relationship, role, or source-specific clue before finalizing.
5. If two searches in a row are low-signal, pivot to a different clue chain:
   different entity role, exact phrase, source type, language/region term, or
   official/primary source.
6. Avoid long "all clues at once" queries. They often retrieve topical but
   wrong matches.
7. Before final answer, silently run a role check: the final span must be the
   entity/fact requested, not a related source, donor, author, parent, team,
   host, article, or organization unless that is exactly what was asked.
8. If evidence is weak near the step limit, output the most plausible short
   candidate span. Do not output process notes or uncertainty prose.
"""


def distilled_strategy_prompt() -> str:
    """Return static generic strategy guidance for text web-search tasks."""
    return DISTILLED_32B_STRATEGY_PROMPT
