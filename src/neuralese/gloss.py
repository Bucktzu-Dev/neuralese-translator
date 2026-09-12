"""Heuristic (and optional LLM) English glosses for symbol classes."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

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


def learn_definition(
    observations: Sequence[Observation],
    *,
    llm_client: Any = None,
    include_private: bool = False,
) -> Dict[str, Any]:
    texts = [obs.text.strip() for obs in observations if obs.text and obs.text.strip()]
    tokens: List[str] = []
    for text in texts:
        tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower()))
    counted = Counter(t for t in tokens if t not in STOP)
    keywords = [w for w, _ in counted.most_common(8)]
    examples = texts[:8] if include_private else []

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

    observation_count = len(observations)
    text_count = len(texts)
    confidence = 0.0
    if observation_count:
        confidence = min(1.0, 0.25 + 0.15 * text_count + 0.05 * len(keywords))
    return {
        "definition": definition,
        "examples": examples[:10],
        "keywords": keywords,
        "confidence": confidence,
        "observation_count": observation_count,
    }


def _contains_raw_observation(definition: str, texts: Sequence[str]) -> bool:
    blob = (definition or "").lower()
    if not blob:
        return False
    for text in texts:
        snippet = text.strip()
        if len(snippet) >= 8 and snippet.lower() in blob:
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
