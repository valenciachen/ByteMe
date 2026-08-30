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


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _attribute_in(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", lowered):
        return "material"
    if re.search(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", lowered):
        return "color"
    if re.search(r"\b(size|sizing|wide|narrow|inseam|waist|small|medium|large|plus)\b", lowered):
        return "size"
    if re.search(r"(?:\$|\bunder\s+\d|\bbudget\b|\bprice\b|\bcost\b)", lowered):
        return "budget"
    if re.search(r"\b(hiking|running|gym|winter|outdoor|work|travel|wedding|casual)\b", lowered):
        return "use_case"
    if re.search(r"\b(style|fit|sleeve|neck|formal|dressy|sporty|slim|relaxed)\b", lowered):
        return "style"
    return "feature"


def _clean_signal(text: str) -> str:
    if not text or NO_PREF_RE.search(text):
        return ""
    lowered = text.lower()
    if ":" in text or "what matters is" in lowered or "what i need is" in lowered:
        text = text.split(":", maxsplit=1)[-1]
    return re.sub(r"\s+", " ", text).strip(" -;,.\t\n")


def _fts_expression(terms: list[str], operator: str) -> str:
    return f" {operator} ".join(f'"{term}"' for term in terms) if terms else ""


def _profile_terms(user_profile: dict) -> list[str]:
    pieces = [str(tag) for tag in user_profile.get("preference_tags") or []]
    if summary := user_profile.get("summary"):
        pieces.append(str(summary))
    return _dedupe(_terms(" ".join(pieces)))


class Agent:
    """Stateful offline retriever for the public TechJam evaluator."""

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
        self._sessions[session_id] = {
            "profile": user_profile,
            "profile_terms": _profile_terms(user_profile),
            "category_context": "",
            "category_terms": [],
            "evidence": [],
            "asked": [],
            "negative_attrs": set(),
            "observed_attrs": set(),
            "last_ask_attribute": None,
        }

    def _search(self, expression: str, limit: int = 80) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.25) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _add_signal(self, session: dict, text: str) -> None:
        cleaned = _clean_signal(text)
        if not cleaned:
            return
        session["evidence"].append(cleaned)
        session["observed_attrs"].add(_attribute_in(cleaned))

    def _query_terms(self, session: dict) -> list[str]:
        return _dedupe(_terms(" ".join([
            session["category_context"], " ".join(session["profile_terms"]), *session["evidence"],
        ])))

    def _query_plan(self, session: dict) -> list[tuple[float, str]]:
        plans: list[tuple[float, str]] = []
        category_terms = session["category_terms"][:6]
        profile_terms = session["profile_terms"][:10]
        evidence_terms = [_terms(text) for text in session["evidence"] if text]
        all_terms = self._query_terms(session)[:40]
        recent_terms = evidence_terms[-1][:12] if evidence_terms else []
        prior_terms = evidence_terms[-2][:12] if len(evidence_terms) >= 2 else []

        if recent_terms:
            if len(recent_terms) >= 2:
                plans.append((4.0, _fts_expression(recent_terms[:6], "AND")))
                plans.append((2.5, _fts_expression(recent_terms[:6], "OR")))
                if len(recent_terms) >= 3:
                    plans.append((5.0, f'"{" ".join(recent_terms[:5])}"'))
            else:
                plans.append((2.0, _fts_expression(recent_terms, "OR")))
        if prior_terms:
            plans.append((2.0, _fts_expression(prior_terms, "AND")))
        if category_terms and recent_terms:
            plans.append((3.5, _fts_expression(_dedupe([*category_terms, *recent_terms[:5]]), "AND")))
        if profile_terms:
            plans.append((1.8, _fts_expression(profile_terms, "AND")))
            plans.append((1.2, _fts_expression(profile_terms, "OR")))
            if recent_terms:
                plans.append((2.0, _fts_expression(_dedupe([*profile_terms[:6], *recent_terms[:4]]), "AND")))
        if category_terms and profile_terms:
            plans.append((1.6, _fts_expression(_dedupe([*category_terms, *profile_terms[:6]]), "AND")))
        if all_terms:
            plans.append((2.4, _fts_expression(all_terms[:10], "AND")))
            plans.append((1.0, _fts_expression(all_terms[:20], "OR")))
        if category_terms and all_terms:
            plans.append((2.8, _fts_expression(_dedupe([*category_terms, *all_terms[:8]]), "AND")))

        cleaned: list[tuple[float, str]] = []
        seen: set[str] = set()
        for weight, expression in plans:
            if expression and expression not in seen:
                seen.add(expression)
                cleaned.append((weight, expression))
        return cleaned

    def _recommend(self, session: dict, top_k: int) -> list[dict]:
        fused: defaultdict[str, float] = defaultdict(float)
        for weight, expression in self._query_plan(session):
            for rank, parent_asin in enumerate(self._search(expression), start=1):
                fused[parent_asin] += weight / (30 + rank)
        ordered = sorted(fused, key=lambda asin: (-fused[asin], asin))[:top_k]
        return [{"parent_asin": asin, "score": round(fused[asin], 8)} for asin in ordered]

    def _next_question(self, session: dict) -> str | None:
        asked = set(session["asked"])
        negative = set(session["negative_attrs"])
        observed = set(session["observed_attrs"])
        if session["last_ask_attribute"]:
            asked.add(session["last_ask_attribute"])

        attr_bonus = {attr: 0 for attr in QUESTION_PRIORITY}
        for token in set(session["profile_terms"]):
            for attr, keywords in TOKEN_PRIORITY_HINTS.items():
                if token in keywords:
                    attr_bonus[attr] += 2 if attr == "feature" else 1
        for attr in observed:
            attr_bonus[attr] += 2

        best_attr, best_score = None, -1
        for attr in QUESTION_PRIORITY:
            if attr in asked or attr in negative or attr in observed:
                continue
            if best_attr is None or attr_bonus[attr] > best_score:
                best_attr, best_score = attr, attr_bonus[attr]
        return best_attr

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        session = self._sessions[session_id]
        previous_question = session["last_ask_attribute"]
        if previous_question and previous_question not in session["asked"]:
            session["asked"].append(previous_question)

        if not session["category_context"]:
            session["category_context"] = _clean_signal(user_message) or user_message
            session["category_terms"] = _terms(session["category_context"])
        else:
            if NO_PREF_RE.search(user_message) and previous_question:
                session["negative_attrs"].add(previous_question)
            if OVERRIDE_RE.search(user_message):
                session["evidence"] = []
                session["observed_attrs"] = set()
            self._add_signal(session, user_message)

        recommendations = self._recommend(session, min(top_k, 10))
        ask_attribute = None if turn >= 10 else self._next_question(session)
        session["last_ask_attribute"] = ask_attribute
        return {
            "message": QUESTION_TEXT[ask_attribute] if ask_attribute else "These are the closest matches I can infer from what you shared.",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
