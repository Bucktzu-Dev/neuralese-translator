"""Heuristic (and optional LLM) English glosses for symbol classes."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Set

from neuralese.contracts import Observation

STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "this",
    "that",
    "with",
}

UNGLOSSED = "[unglossed]"
_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def learn_definition(
    observations: Sequence[Observation],
    *,
    llm_client: Any = None,
    include_private: bool = False,
) -> Dict[str, Any]:
    texts = [obs.text.strip() for obs in observations if obs.text and obs.text.strip()]
    tokens: List[str] = []
    for text in texts:
        tokens.extend(_TOKEN.findall(text.lower()))
    counted = Counter(t for t in tokens if t not in STOP)
    keywords = [w for w, _ in counted.most_common(8)]
    examples = texts[:8] if include_private else []
    if not include_private:
        keywords = _public_keywords(keywords, texts)

    if llm_client is not None and (texts if include_private else keywords):
        if include_private:
            prompt = (
                "Write one short English sentence defining the shared meaning of these examples. "
                "Do not add facts that are not in the examples.\n"
                + "\n".join(f"- {t}" for t in texts[:12])
            )
        else:
            prompt = (
                "Write one short English sentence defining a symbol whose keywords are: "
                + ", ".join(keywords)
                + ". Do not quote private examples or add facts that are not in the keywords."
            )
        try:
            definition = str(llm_client.generate(prompt, max_tokens=80)).strip()
        except TypeError:
            definition = str(llm_client.generate(prompt)).strip()
        except Exception:
            definition = _heuristic_definition(keywords, examples)
        if not include_private and _contains_raw_observation(definition, texts):
            definition = _heuristic_definition(keywords, [])
    else:
        definition = _heuristic_definition(keywords, examples)

    if not include_private:
        if not (definition or "").strip() or _contains_raw_observation(definition, texts):
            definition = _heuristic_definition(keywords, [])
            # Keyword glosses may share short tokens with observations; 8-char
            # windows apply to LLM echoes, not to the heuristic template.
            if not (definition or "").strip() or _contains_raw_observation(
                definition, texts, windows=False
            ):
                definition = UNGLOSSED

    if not (definition or "").strip():
        definition = UNGLOSSED

    observation_count = len(observations)
    text_count = len(texts)
    confidence = 0.0
    if observation_count and (definition or "").strip() != UNGLOSSED:
        confidence = min(1.0, 0.25 + 0.15 * text_count + 0.05 * len(keywords))
    return {
        "definition": definition,
        "examples": examples[:10],
        "keywords": keywords,
        "confidence": confidence,
        "observation_count": observation_count,
    }


def _normalized_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _public_keywords(keywords: Sequence[str], texts: Sequence[str]) -> List[str]:
    blocked = _raw_observation_forms(texts)
    lowered = [_normalized_text(t) for t in texts if t and t.strip()]
    safe: List[str] = []
    for word in keywords:
        token = word.lower()
        if token in blocked:
            continue
        if len(token) >= 8 and any(token in text for text in lowered):
            continue
        safe.append(word)
    return safe


def _raw_observation_forms(texts: Sequence[str]) -> Set[str]:
    """Full observations and single-token payloads that would reproduce raw text."""
    blocked: Set[str] = set()
    for text in texts:
        snippet = _normalized_text(text)
        if not snippet:
            continue
        blocked.add(snippet)
        words = _TOKEN.findall(snippet)
        if len(words) == 1:
            blocked.add(words[0])
        alnum = re.findall(r"[a-z0-9]+", snippet)
        if len(alnum) == 1:
            blocked.add(alnum[0])
    return blocked


def _contains_raw_observation(
    definition: str, texts: Sequence[str], *, windows: bool = True
) -> bool:
    blob = _normalized_text(definition or "")
    if not blob:
        return False
    for form in _raw_observation_forms(texts):
        if not form:
            continue
        if len(form) >= 3:
            if form in blob:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", blob):
            return True
    if not windows:
        return False
    for text in texts:
        snippet = _normalized_text(text)
        if len(snippet) < 8:
            continue
        for i in range(len(snippet) - 7):
            if snippet[i : i + 8] in blob:
                return True
    return False


def _heuristic_definition(keywords: List[str], examples: List[str]) -> str:
    if keywords:
        head = ", ".join(keywords[:4])
        return f"Symbol for {head}."
    if examples:
        snippet = examples[0]
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return f"Symbol attested by: {snippet}"
    return ""
