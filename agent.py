from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
OVERRIDE_RE = re.compile(r"\b(actually|ignore|instead|rather than|change of plans)\b", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "need", "not", "no", "preference", "additional", "still", "yet",
    "actually", "ignore", "earlier", "what", "matters", "those", "options",
    "quite", "right", "ask", "about", "one", "specific", "for",
}

QUESTION_TEXT = {
    "material": "What material would feel best for this?",
    "color": "Do you have a color in mind?",
    "size": "Is there a sizing or fit requirement I should prioritize?",
    "use_case": "What activity or situation will you use it for?",
    "budget": "Do you have a budget range in mind?",
    "style": "What style or cut are you after?",
    "feature": "What is the most important product feature for you?",
    "other": "What is the single most important detail I should use to narrow this down?",
}
QUESTION_ORDER = ("material", "color", "size", "use_case", "budget", "style", "feature")


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


def _attribute_in(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", lowered):
        return "material"
    if re.search(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", lowered):
        return "color"
    if re.search(r"\b(size|sizing|wide|narrow|inseam|waist|small|medium|large)\b", lowered):
        return "size"
    if re.search(r"(?:\$|\bunder\s+\d|\bbudget\b|\bprice\b)", lowered):
        return "budget"
    if re.search(r"\b(hiking|running|gym|winter|outdoor|work|travel|wedding|casual)\b", lowered):
        return "use_case"
    if re.search(r"\b(style|fit|sleeve|neck|formal|dressy|sporty)\b", lowered):
        return "style"
    return None


def _fts_expression(terms: list[str], operator: str) -> str:
    return f" {operator} ".join(f'"{term}"' for term in terms)


def _searchable_message(message: str) -> str:
    """Remove simulator-style conversational wrappers while retaining product evidence."""
    if ":" in message:
        _, detail = message.split(":", maxsplit=1)
        if detail.strip():
            return detail.strip()
    return message


class Agent:
    """Offline, stateful hybrid retriever for the TechJam agent contract."""

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
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile,
            "category_context": "",
            "evidence": [],
            "asked": [],
            "override_pending": False,
        }

    def _search(self, expression: str, limit: int = 80) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 7.0, 4.0, 3.0, 3.0, 2.0, 1.5) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _recommend(self, session: dict, top_k: int) -> list[dict]:
        all_terms = list(dict.fromkeys(_terms(" ".join(session["evidence"]))))[:45]
        recent_terms = list(dict.fromkeys(_terms(session["evidence"][-1] if session["evidence"] else "")))[:25]
        if not all_terms:
            return []

        rankings: list[tuple[float, list[str]]] = []
        if len(recent_terms) >= 2:
            rankings.append((3.0, self._search(_fts_expression(recent_terms, "AND"))))
        if len(all_terms) >= 2:
            rankings.append((2.0, self._search(_fts_expression(all_terms, "AND"))))
        rankings.append((1.0, self._search(_fts_expression(all_terms, "OR"))))
        if recent_terms and recent_terms != all_terms:
            rankings.append((1.5, self._search(_fts_expression(recent_terms, "OR"))))

        fused: defaultdict[str, float] = defaultdict(float)
        for weight, ranking in rankings:
            for rank, parent_asin in enumerate(ranking, start=1):
                fused[parent_asin] += weight / (40 + rank)
        ordered = sorted(fused, key=lambda asin: (-fused[asin], asin))[:top_k]
        return [{"parent_asin": parent_asin, "score": round(fused[parent_asin], 8)} for parent_asin in ordered]

    def _next_question(self, session: dict, user_message: str) -> str | None:
        asked: list[str] = session["asked"]
        if session["override_pending"]:
            session["override_pending"] = False
            asked.append("other")
            return "other"
        observed = {_attribute_in(text) for text in session["evidence"]}
        for attribute in QUESTION_ORDER:
            if attribute not in observed and attribute not in asked:
                asked.append(attribute)
                return attribute
        if "other" not in asked:
            asked.append("other")
            return "other"
        return None

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        session = self._sessions[session_id]
        if not session["category_context"]:
            session["category_context"] = user_message.split(".", maxsplit=1)[0]
        if OVERRIDE_RE.search(user_message):
            session["evidence"] = [session["category_context"], _searchable_message(user_message)]
            session["asked"] = []
            session["override_pending"] = True
        else:
            session["evidence"].append(_searchable_message(user_message))

        recommendations = self._recommend(session, min(top_k, 10))
        ask_attribute = self._next_question(session, user_message)
        message = QUESTION_TEXT[ask_attribute] if ask_attribute else "These are the closest matches based on what you shared."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
