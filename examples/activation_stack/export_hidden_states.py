"""Write a deterministic hf_layers dump from the prose corpus.

The matrix is hashed text, not a model activation. Nothing here downloads
weights, reads a clock, or places orthogonal centers by hand.

Layer 0 is the character trigram hash ``learn`` already uses (dim 32, n 3).
Layer 1 is a hashed word-unigram vector of the same width.
Layer 2 is the per-row L2-normalized sum of those two.

Shape is ``(layers, n, d) = (3, 320, 32)``, which ``neuralese adapt`` reads
as layout ``hf_layers``.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from neuralese.adapters import hashed_ngram_vector, load_observations_jsonl

DIM = 32
NGRAM = 3
LAYOUT = "hf_layers"
LAYER_NAMES = ("char_trigram", "word_unigram", "sum_l2")
# Zip epoch. The npz must not contain the wall clock.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

ROOT = Path(__file__).resolve().parent
PROSE = ROOT.parent / "prose_corpus" / "observations.jsonl"
OUTPUT = ROOT / "hidden_states.npz"


def hashed_word_unigram(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic hashed word-unigram vector. One signed bin per word."""
    vec = np.zeros(dim, dtype=np.float64)
    words = text.lower().split()
    if not words:
        words = ["_"]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def stack_from_texts(texts: list[str]) -> np.ndarray:
    """Return hidden states with shape (3, n, 32) for these row texts."""
    if not texts:
        raise ValueError("activation export needs at least one text row")
    trigram = np.asarray(
        [hashed_ngram_vector(text, dim=DIM, n=NGRAM) for text in texts],
        dtype=np.float64,
    )
    words = np.stack([hashed_word_unigram(text, dim=DIM) for text in texts])
    mixed = trigram + words
    norms = np.linalg.norm(mixed, axis=1, keepdims=True)
    mixed = np.divide(mixed, norms, out=np.zeros_like(mixed), where=norms > 0.0)
    hidden = np.stack([trigram, words, mixed]).astype("<f8", copy=False)
    if hidden.shape != (len(LAYER_NAMES), len(texts), DIM):
        raise ValueError(f"unexpected hidden shape {hidden.shape}")
    if not np.isfinite(hidden).all():
        raise ValueError("hidden states contain non-finite values")
    return np.ascontiguousarray(hidden)


def build_arrays() -> dict[str, np.ndarray]:
    observations = load_observations_jsonl(PROSE)
    texts = [obs.text or "" for obs in observations]
    if any(not text for text in texts):
        raise ValueError("prose rows must have text")
    ids = [obs.observation_id for obs in observations]
    return {
        "hidden_states": stack_from_texts(texts),
        "texts": np.asarray(texts),
        "observation_ids": np.asarray(ids),
        "layout": np.asarray(LAYOUT),
        "layer_names": np.asarray(list(LAYER_NAMES)),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def save_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Zip npy members with a fixed timestamp so the file does not embed a clock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            payload = _npy_bytes(arrays[name])
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.flag_bits = 0
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compresslevel=9)


def main() -> None:
    arrays = build_arrays()
    save_deterministic_npz(OUTPUT, arrays)
    hidden = arrays["hidden_states"]
    print(
        f"wrote {OUTPUT.name} layout={LAYOUT} shape={tuple(int(x) for x in hidden.shape)} "
        f"layers={','.join(LAYER_NAMES)}"
    )


if __name__ == "__main__":
    main()
