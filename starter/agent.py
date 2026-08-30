from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
OVERRIDE_RE = re.compile(r"\b(actually|ignore|instead|rather than|change of plans)\b", re.I)
NO_PREF_RE = re.compile(
    r"(don't have(?: an? additional)? preference|no preference|use your judg(?:e)?ment|"
    r"not quite right yet|ask me about one specific attribute)",
    re.I,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "need", "not", "no", "preference", "additional", "still", "yet",
    "actually", "ignore", "earlier", "what", "matters", "those", "options",
    "quite", "right", "ask", "about", "one", "specific", "don't", "dont",
    "judgment", "judgement", "use", "your", "tell", "detail", "details",
    "single", "best", "more", "any", "another", "later", "maybe", "just",
}

QUESTION_TEXT = {
    "feature": "What product feature matters most to you?",
    "material": "What material would feel best for this?",
    "use_case": "What activity or situation will you use it for?",
    "budget": "Do you have a budget range in mind?",
    "color": "Do you have a color in mind?",
    "size": "Is there a sizing or fit requirement I should prioritize?",
    "style": "What style or cut are you after?",
    "other": "What single detail should I use to narrow this down?",
}

QUESTION_PRIORITY = ("feature", "material", "use_case", "budget", "color", "size", "style", "other")

TOKEN_PRIORITY_HINTS = {
    "feature": {
        "fit", "comfort", "durability", "quality", "support", "stretch",
        "soft", "warm", "lightweight", "breathable", "waterproof", "versatile",
        "performance", "practical", "slim", "loose", "fleece", "mesh",
    },
    "material": {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"},
    "use_case": {"hiking", "running", "gym", "winter", "outdoor", "work", "travel", "wedding", "casual"},
    "budget": {"budget", "price", "cost", "cheap", "expensive", "affordable"},
    "color": {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"},
    "size": {"size", "sizing", "wide", "narrow", "inseam", "waist", "small", "medium", "large", "plus"},
    "style": {"style", "fit", "sleeve", "neck", "formal", "dressy", "sporty", "casual", "slim", "relaxed"},
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]), _text(product.get("title")),
                    _text(product.get("categories")), _text(product.get("features")),
                    _text(product.get("details")), _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
