from __future__ import annotations

import json
import math
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
INITIAL_INTENT_RE = re.compile(
    r"^I'm looking for (?P<category>.*?)(?:\. A key requirement is: (?P<constraint>.*)|, but I'm still exploring\.?|\. (?P<preference>.*))$",
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

COLOR_ALIASES = {
    "black": "black", "white": "white", "blue": "blue", "red": "red", "pink": "pink",
    "green": "green", "brown": "brown", "gray": "gray", "grey": "gray", "purple": "purple",
    "yellow": "yellow", "orange": "orange", "beige": "beige", "navy": "navy", "gold": "gold",
    "silver": "silver",
}
MATERIAL_ALIASES = {
    "cotton": "cotton", "polyester": "polyester", "nylon": "nylon", "leather": "leather",
    "wool": "wool", "spandex": "spandex", "silk": "silk", "rayon": "rayon", "fabric": "fabric",
    "canvas": "canvas", "denim": "denim", "suede": "suede",
}
STYLE_ALIASES = {
    "casual": "casual", "formal": "formal", "dressy": "dressy", "sporty": "sporty", "slim": "slim",
    "relaxed": "relaxed", "minimalist": "minimalist", "classic": "classic", "modern": "modern",
    "outdoor": "outdoor", "athletic": "athletic", "running": "running", "trail": "trail",
    "hiking": "hiking", "rain": "rain", "winter": "winter", "travel": "travel",
}
SYNONYM_MAP = {
    "jacket": {"jacket", "coat", "blazer", "shacket", "parka"},
    "shoe": {"shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers", "boot", "boots"},
    "hoodie": {"hoodie", "hooded", "sweatshirt"},
    "sweater": {"sweater", "pullover"},
    "pants": {"pants", "trouser", "trousers", "slacks"},
    "handbag": {"handbag", "purse", "bag", "clutch"},
    "gray": {"gray", "grey"},
    "wallet": {"wallet", "cardcase"},
    "raincoat": {"raincoat", "rain jacket"},
    "tee": {"tee", "tshirt", "t-shirt", "shirt"},
}
TOKEN_PRIORITY_HINTS = {
    "feature": {"fit", "comfort", "durability", "quality", "support", "stretch", "soft", "warm",
                "lightweight", "breathable", "waterproof", "versatile", "performance", "practical",
                "slim", "loose", "fleece", "mesh", "packable"},
    "material": {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"},
    "use_case": {"hiking", "running", "gym", "winter", "outdoor", "work", "travel", "wedding", "casual"},
    "budget": {"budget", "price", "cost", "cheap", "expensive", "affordable"},
    "color": {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"},
    "size": {"size", "sizing", "wide", "narrow", "inseam", "waist", "small", "medium", "large", "plus"},
    "style": {"style", "fit", "sleeve", "neck", "formal", "dressy", "sporty", "casual", "slim", "relaxed"},
    "brand": {"brand", "store", "shop", "manufacturer"},
}


def canonicalize_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text).lower()
    value = value.replace("&", " and ")
    value = value.replace("/", " ")
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9\s$]", " ", value)
    for canonical, aliases in sorted(SYNONYM_MAP.items(), key=lambda pair: len(pair[0]), reverse=True):
        for alias in sorted(aliases, key=len, reverse=True):
            value = re.sub(rf"\b{re.escape(alias)}\b", canonical, value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    normalized = canonicalize_text(text)
    terms = []
    for token in TOKEN_RE.findall(normalized):
        token = token.lower()
        if len(token) > 1 and token not in STOPWORDS:
            terms.append(token)
    return terms


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _attribute_in(text: str) -> str | None:
    lowered = canonicalize_text(text)
    tokens = set(lowered.split())
    if tokens & {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}:
        return "material"
    if tokens & {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "yellow", "orange", "beige", "navy"}:
        return "color"
    if tokens & {"size", "sizing", "wide", "narrow", "inseam", "waist", "small", "medium", "large", "plus", "xl", "xs", "xxl"}:
        return "size"
    if tokens & {"budget", "price", "cost", "cheap", "expensive", "affordable", "under", "below", "around"}:
        return "budget"
    if tokens & {"hiking", "running", "gym", "winter", "outdoor", "work", "travel", "wedding", "casual", "beach", "swim", "office"}:
        return "use_case"
    if tokens & {"style", "fit", "sleeve", "neck", "formal", "dressy", "sporty", "slim", "relaxed", "minimalist", "classic"}:
        return "style"
    if tokens & {"brand", "store", "manufacturer"}:
        return "brand"
    return "feature"


def parse_budget_hint(text: str) -> tuple[float, float] | None:
    lowered = canonicalize_text(text)
    if "affordable" in lowered or "inexpensive" in lowered:
        return (0.0, 50.0)
    match = re.search(r"(?:under|below|less than|up to)\s*(?:\$\s*)?(\d+(?:\.\d+)?)", lowered)
    if match:
        return (0.0, float(match.group(1)))
    match = re.search(r"(?:around|about|approximately|near|roughly)\s*(?:\$\s*)?(\d+(?:\.\d+)?)", lowered)
    if match:
        target = float(match.group(1))
        return (max(0.0, target * 0.8), target * 1.2)
    match = re.search(r"(?:budget(?:\s+of)?|price(?:\s+of)?|cost(?:\s+of)?)\s*(?:\$\s*)?(\d+(?:\.\d+)?)\s*(?:dollars?|usd)?", lowered)
    if match:
        return (0.0, float(match.group(1)))
    match = re.search(r"(?:\$\s*\d+(?:\.\d+)?\s*(?:to|-)\s*\$?\d+(?:\.\d+)?)", lowered)
    if match:
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", match.group(0))]
        if len(nums) >= 2:
            return (min(nums), max(nums))
    return None


def _clean_signal(text: str) -> str:
    if not text or NO_PREF_RE.search(text):
        return ""
    lowered = text.lower()
    if ":" in text or "what matters is" in lowered or "what i need is" in lowered:
        text = text.split(":", maxsplit=1)[-1]
    return re.sub(r"\s+", " ", text).strip(" -;,.\t\n")


def _initial_signals(user_message: str) -> tuple[str, list[str]]:
    match = INITIAL_INTENT_RE.match(user_message.strip())
    if not match:
        cleaned = _clean_signal(user_message) or user_message
        return cleaned, [cleaned] if cleaned else []
    category = _clean_signal(match.group("category"))
    signals: list[str] = []
    for key in ("constraint", "preference"):
        value = match.group(key)
        cleaned = _clean_signal(value) if value else ""
        if cleaned:
            signals.append(cleaned)
    return category or user_message, signals


def _constraint_parts(text: str) -> list[str]:
    cleaned = _clean_signal(text)
    if not cleaned:
        return []
    parts = [cleaned]
    lowered = cleaned.lower()
    if "what matters is:" in lowered:
        _, tail = cleaned.split(":", maxsplit=1)
        parts.extend(item.strip() for item in tail.split(";"))
    elif ";" in cleaned:
        parts.extend(item.strip() for item in cleaned.split(";"))
    return _dedupe([_clean_signal(part) for part in parts])


def _fts_expression(terms: list[str], operator: str) -> str:
    return f" {operator} ".join(f'"{term}"' for term in terms) if terms else ""


def _profile_terms(user_profile: dict) -> list[str]:
    pieces = [str(tag) for tag in user_profile.get("preference_tags") or []]
    if summary := user_profile.get("summary"):
        pieces.append(str(summary))
    return _dedupe(_terms(" ".join(pieces)))


class Agent:
    """Stateful offline retriever tuned to the product-attribute patterns in the public shopping sessions."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._catalog: dict[str, dict] = {}
        self._documents: dict[str, str] = {}
        self._sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, description, store, "
            "price, average_rating, rating_number, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self._catalog[parent_asin] = product
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                description = _text(product.get("description"))
                store = _text(product.get("store"))
                price = str(product.get("price") or "")
                avg_rating = str(product.get("average_rating") or "")
                rating_count = str(product.get("rating_number") or "")
                searchable = " ".join(
                    segment for segment in (title, categories, features, details, description, store, price, avg_rating, rating_count)
                    if segment
                )
                self._documents[parent_asin] = canonicalize_text(searchable)
                batch.append((parent_asin, title, categories, features, details, description, store, price, avg_rating, rating_count))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile,
            "profile_terms": _profile_terms(user_profile),
            "category_context": "",
            "category_terms": [],
            "constraints": {},
            "evidence": [],
            "asked": [],
            "negative_attrs": set(),
            "observed_attrs": set(),
            "last_ask_attribute": None,
        }

    def _search(self, expression: str, limit: int = 200) -> list[str]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? ORDER BY bm25(products) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _merge_constraint(self, session: dict, attr: str, value: str) -> None:
        if not attr or not value:
            return
        values = session["constraints"].setdefault(attr, [])
        if value not in values:
            values.append(value)

    def _extract_structured_constraints(self, text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        normalized = canonicalize_text(text)
        for attr, mapping in (("material", MATERIAL_ALIASES), ("color", COLOR_ALIASES), ("style", STYLE_ALIASES)):
            for key, value in mapping.items():
                if key in normalized:
                    result[attr] = value
                    break
        if "gray" in normalized or "grey" in normalized:
            result["color"] = "gray"
        if "women" in normalized or "womens" in normalized:
            result["gender"] = "women"
        elif "men" in normalized or "mens" in normalized:
            result["gender"] = "men"
        budget = parse_budget_hint(text)
        if budget:
            result["budget"] = f"{budget[0]}:{budget[1]}"
        return result

    def _add_signal(self, session: dict, text: str) -> None:
        for cleaned in _constraint_parts(text):
            if not cleaned:
                continue
            session["evidence"].append(cleaned)
            attr = _attribute_in(cleaned)
            if attr:
                session["observed_attrs"].add(attr)
            if OVERRIDE_RE.search(cleaned):
                session["constraints"].clear()
            for key, value in self._extract_structured_constraints(cleaned).items():
                self._merge_constraint(session, key, value)

    def _category_terms(self, session: dict) -> list[str]:
        category = session["category_context"]
        if not category:
            return []
        terms = [token for token in _terms(category) if token not in {"clothing"}]
        return _dedupe(terms)

    def _query_plan(self, session: dict) -> list[tuple[float, str]]:
        category_terms = self._category_terms(session)
        profile_terms = session["profile_terms"][:12]
        evidence_terms = [_terms(text) for text in session["evidence"] if text]
        recent = evidence_terms[-1][:12] if evidence_terms else []
        prior = evidence_terms[-2][:12] if len(evidence_terms) >= 2 else []
        all_terms = _dedupe(_terms(" ".join([session["category_context"], *session["evidence"], *profile_terms])))

        plan: list[tuple[float, str]] = []
        if recent:
            plan.append((5.0, _fts_expression(recent[:6], "AND")))
            plan.append((3.0, _fts_expression(recent[:10], "OR")))
        if prior:
            plan.append((2.0, _fts_expression(prior[:6], "AND")))
        if category_terms:
            plan.append((4.0, _fts_expression(category_terms[:6], "AND")))
            if recent:
                plan.append((3.5, _fts_expression(_dedupe([*category_terms[:6], *recent[:5]]), "AND")))
        if profile_terms:
            plan.append((2.0, _fts_expression(profile_terms[:8], "AND")))
        if all_terms:
            plan.append((2.5, _fts_expression(all_terms[:10], "AND")))

        cleaned: list[tuple[float, str]] = []
        seen: set[str] = set()
        for weight, expression in plan:
            if expression and expression not in seen:
                cleaned.append((weight, expression))
                seen.add(expression)
        return cleaned

    def _product_doc(self, product: dict) -> str:
        if not product:
            return ""
        fields = [
            product.get("title"),
            product.get("categories"),
            product.get("features"),
            product.get("details"),
            product.get("description"),
            product.get("store"),
            f"price {product.get('price')}",
            f"rating {product.get('average_rating')}",
            f"reviews {product.get('rating_number')}",
        ]
        return canonicalize_text(_text(fields))

    def _attribute_score(self, value: str, product: dict) -> float:
        if not value:
            return 0.0
        target = canonicalize_text(value)
        doc = self._product_doc(product)
        if not target:
            return 0.0
        if target in doc:
            return 32.0
        terms = _terms(target)
        if not terms:
            return 0.0
        overlap = sum(1 for term in terms if term in doc)
        if overlap:
            return 12.0 * (overlap / len(terms))
        return 0.0

    def _score_candidate(self, session: dict, parent_asin: str) -> float:
        product = self._catalog.get(parent_asin, {})
        doc = self._product_doc(product)
        score = 0.0

        for attr, values in session["constraints"].items():
            for value in values:
                if attr == "budget" and value and ":" in value:
                    lo, hi = map(float, value.split(":", 1))
                    price = product.get("price")
                    try:
                        price_value = float(price)
                    except (TypeError, ValueError):
                        continue
                    if lo <= price_value <= hi:
                        score += 60.0
                    else:
                        mid = (lo + hi) / 2.0
                        score += max(0.0, 18.0 - abs(price_value - mid) / 10.0)
                else:
                    score += self._attribute_score(value, product)

        for term in self._category_terms(session):
            if term in doc:
                score += 16.0
        for term in session["profile_terms"]:
            if term in doc:
                score += 6.0
        for expr in session["evidence"]:
            if expr and canonicalize_text(expr) in doc:
                score += 10.0

        title = canonicalize_text(product.get("title") or "")
        if title and any(term in title for term in self._category_terms(session)):
            score += 10.0

        try:
            avg = product.get("average_rating")
            if avg is not None:
                score += float(avg) * 4.0
            reviews = product.get("rating_number")
            if reviews is not None:
                score += min(float(reviews) / 100.0, 8.0)
        except (TypeError, ValueError):
            pass

        return score

    def _recommend(self, session: dict, top_k: int) -> list[dict]:
        fused: defaultdict[str, float] = defaultdict(float)
        for weight, expression in self._query_plan(session):
            for rank, asin in enumerate(self._search(expression, limit=200), start=1):
                fused[asin] += weight / (rank + 4.0)

        if not fused:
            candidates = list(self._catalog)
        else:
            candidates = list(fused)

        ordered = sorted(
            candidates,
            key=lambda asin: (-self._score_candidate(session, asin), -fused.get(asin, 0.0), asin),
        )[:top_k]
        return [{"parent_asin": asin, "score": round(self._score_candidate(session, asin), 6)} for asin in ordered]

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
            if attr == "budget" and session["constraints"].get("budget"):
                continue
            if attr_bonus[attr] > best_score:
                best_attr, best_score = attr, attr_bonus[attr]
        return best_attr

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        session = self._sessions[session_id]
        previous_question = session["last_ask_attribute"]
        if previous_question and previous_question not in session["asked"]:
            session["asked"].append(previous_question)

        if not session["category_context"]:
            category_context, signals = _initial_signals(user_message)
            session["category_context"] = category_context
            session["category_terms"] = self._category_terms(session)
            for signal in signals:
                self._add_signal(session, signal)
        else:
            if NO_PREF_RE.search(user_message) and previous_question:
                session["negative_attrs"].add(previous_question)
            if OVERRIDE_RE.search(user_message):
                session["constraints"].clear()
            self._add_signal(session, user_message)

        recommendations = self._recommend(session, min(top_k, 10))
        ask_attribute = None if turn >= 10 else self._next_question(session)
        session["last_ask_attribute"] = ask_attribute
        return {
            "message": QUESTION_TEXT.get(ask_attribute, "These are the closest matches I can infer from what you shared.") if ask_attribute else "These are the closest matches I can infer from what you shared.",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
