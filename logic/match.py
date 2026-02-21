"""
logic/match.py — Keyword matching for "find" mode.

Performs case-insensitive substring matching between OCR text
and a user query.
"""


def find_best_match(texts: list[str], query: str) -> int:
    """
    Search each text in `texts` for `query` (case-insensitive substring).
    Returns the index of the first match, or -1 if none found.

    Args:
        texts: list of OCR strings, one per candidate box.
        query: the search term (e.g. "milk").

    Returns:
        Index into `texts` of first match, or -1.
    """
    q = query.lower().strip()
    for i, text in enumerate(texts):
        if q in text.lower():
            return i
    return -1
