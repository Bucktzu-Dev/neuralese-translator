"""Activation-boundary distinguishability. Research only.

The final-layer state in the parent capture is the final layer norm, and every
prompt there ends on tokenizer id 15. This module asks whether that boundary
and that shared terminal token hide a controlled slot distinction from the
existing farthest-first k-means. It does not download weights, does not run a
model, and does not change ``word_sentence_svd``, ``learn_pack``,
``clustering.kmeans``, ``certify``, the taus, or the package version.

Admission constants above ``run_activation_boundary`` were fixed before this
trial's held-out scores were computed. Probe labels are not an input to
``kmeans``. A gloss is not an input to the verdict.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from neuralese.activation_source import (
    EXPECTED_HIDDEN_SIZE,
    FIXTURE_META as PARENT_META,
    FIXTURE_NPZ as PARENT_NPZ,
    HELDOUT_PROMPTS as PARENT_HELDOUT_PROMPTS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_URL,
    TRAIN_PROMPTS as PARENT_TRAIN_PROMPTS,
    WEIGHT_SHA256,
    assign_squared_euclidean,
    assess_fixture as assess_parent_fixture,
    float32_sha256,
    int32_sha256,
    prompt_fingerprint as parent_prompt_fingerprint,
)
from neuralese.clustering import kmeans
from neuralese.contracts import canonical_dumps

# Status of this trial. Not a production admission.
STATUS = "PLAN_ONLY"
PARENT_COMMIT = "852b681"
BASELINE_COMMIT = "c6321d5"
ENTERING_VERDICT = "UNRESOLVED"
ENTERING_RECOMMENDATION = "NONE"
RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"

# Frozen stack. The capture script checks these before the forward.
EXPECTED_TORCH = "2.14.1+cpu"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_TUPLE_LENGTH = 7
FORWARD_DTYPE = "float32"
STORED_DTYPE = "float32"

# Layer and token coordinates. Chosen before scoring.
# hidden_states index 6 is the final layer norm (parent boundary).
# hidden_states index 3 is the tensor appended before GPTNeoX block 3
# (0-based), the output of block 2. It is not the embedding and not the
# final layer norm. Condition 3 reads that index at the final unpadded
# position of the condition-2 prompts.
FINAL_LAYER_INDEX = 6
INTERMEDIATE_LAYER_INDEX = 3
CONDITION_3_TOKEN_COORDINATE = "condition_2_final_unpadded_position"
DISTANCE = "squared_euclidean"
DISPERSION = "sample_standard_deviation_ddof_1"

# Controlled property. A probe only. It is not an embedding feature, not a
# cluster feature, and not a model-selection target.
CONTROLLED_PROPERTY = (
    "binary identity of one subject-slot word in a fixed frame; "
    "probe 0 is the string cat; probe 1 is the string dog; "
    "each string is one token at index 1; "
    "pair members share the frame and the terminal token"
)
SLOT_WORDS: Tuple[str, ...] = ("cat", "dog")
SLOT_TOKEN_IDS: Tuple[int, ...] = (5798, 4370)
SLOT_INDEX = 1
SHARED_TERMINAL = "."
SHARED_TERMINAL_ID = 15
VARIED_TERMINALS: Tuple[str, ...] = ("!", "?", ";", ":")
VARIED_TERMINAL_IDS: Tuple[int, ...] = (2, 32, 28, 27)
TIE_BREAK = "smaller_probe_id"

# Codebook width matches the library default so the comparison is the
# boundary, not a new k. Production ``kmeans`` is called as-is.
N_SYMBOLS = 8
LEARN_SEED = 0
SHUFFLE_ASSIGNMENT_SEED = 0

# Admission gates. Fixed before scoring. Do not edit these to fit a result.
DISTINCTION_MARGIN = 0.10
DISTINCTION_GATE = (
    "heldout_probe_accuracy_at_least_margin_above_train_mean_"
    "shuffled_assignment_token_identity_and_nearest_train_row"
)
GEOMETRY_NN_FLOOR = 0.75
GEOMETRY_SEPARATION_GATE = "nearest_train_row_probe_accuracy_at_least_floor"
TERMINAL_CONCENTRATION_DROP = 0.25
TERMINAL_CLAIM_GATE = (
    "parent_heldout_max_centroid_fraction_drops_by_at_least_"
    "the_predeclared_amount_when_only_the_final_token_changes"
)
# Logits of the clustered hidden state are a linear readout of that state.
# This trial does not substitute them, a centroid cosine, or a reconstruction
# error for a behavioral target.
INDEPENDENT_TARGET_STATUS = "INDEPENDENT_TARGET_UNAVAILABLE"
INDEPENDENT_TARGET_REASON = (
    "no behavioral target that is not a function of the clustered vector "
    "was constructed; next-token logits of the clustered hidden state are disallowed"
)

# Parent k-means geometry, published before this trial. Reproduction must hit
# these figures. They are not thresholds for the new prompts.
PARENT_TRAIN_COUNTS = (1, 2, 5, 1, 3, 3, 16, 1)
PARENT_HELDOUT_COUNTS = (0, 0, 4, 0, 0, 1, 11, 0)
PARENT_TRAIN_RESIDUAL = 0.22755261731526205
PARENT_HELDOUT_PROTOTYPE_RESIDUAL = 0.29343442149783966
PARENT_HELDOUT_TRAIN_MEAN_RESIDUAL = 0.30512149677079187
PARENT_HELDOUT_SHUFFLED_RESIDUAL = 0.33508908512206886
REPRODUCTION_ABS = 1e-12

VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_RESEARCH_FAIL = "RESEARCH_FAIL"
VERDICT_GEOMETRY_ONLY = "GEOMETRY_ONLY"
VERDICT_DISTINCTION = "DISTINCTION_WITHOUT_INDEPENDENT_TARGET"
VERDICT_DISTINCTION_AND_TARGET = "DISTINCTION_WITH_INDEPENDENT_TARGET"
RECOMMENDATION_NONE = "NONE"
RECOMMENDATION_RESEARCH_NOTE = "RESEARCH_NOTE"
VERDICTS = (
    VERDICT_UNRESOLVED,
    VERDICT_RESEARCH_FAIL,
    VERDICT_GEOMETRY_ONLY,
    VERDICT_DISTINCTION,
    VERDICT_DISTINCTION_AND_TARGET,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "examples" / "activation_boundary"
FIXTURE_NPZ = FIXTURE_DIR / "activations.npz"
FIXTURE_META = FIXTURE_DIR / "capture_meta.json"

# Context-matched frames. The slot is ``{w}``. The terminal is appended later.
# Train and held-out frames were listed before the forward.
TRAIN_FRAMES: Tuple[str, ...] = (
    "The {w} rested beside the gate",
    "The {w} waited under the awning",
    "The {w} crossed the empty square",
    "The {w} slept near the window",
    "The {w} stood behind the counter",
    "The {w} paused along the fence",
    "The {w} watched from the doorway",
    "The {w} settled onto the bench",
    "The {w} followed the gravel path",
    "The {w} remained inside the barn",
    "The {w} approached the wooden bridge",
    "The {w} circled the stone fountain",
    "The {w} stayed near the mailbox",
    "The {w} climbed onto the wagon",
    "The {w} wandered through the orchard",
    "The {w} returned before the bell",
)
HELDOUT_FRAMES: Tuple[str, ...] = (
    "The {w} listened from the hallway",
    "The {w} gathered by the river",
    "The {w} hurried across the courtyard",
    "The {w} lingered near the library",
    "The {w} arrived after the rain",
    "The {w} departed before sunrise",
    "The {w} noticed the open window",
    "The {w} guarded the side door",
)


def design_fingerprint() -> str:
    payload = {
        "heldout_frames": list(HELDOUT_FRAMES),
        "intermediate_layer_index": INTERMEDIATE_LAYER_INDEX,
        "parent_fingerprint": parent_prompt_fingerprint(),
        "shared_terminal": SHARED_TERMINAL,
        "slot_words": list(SLOT_WORDS),
        "train_frames": list(TRAIN_FRAMES),
        "varied_terminals": list(VARIED_TERMINALS),
    }
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def build_pair_rows(frames: Sequence[str], terminals: Sequence[str]) -> List[dict]:
    """Row order is frame, then terminal, then slot word.

    Adjacent rows ``(2i, 2i+1)`` are one context-matched pair.
    """
    rows: List[dict] = []
    for frame_index, frame in enumerate(frames):
        for terminal in terminals:
            for probe, word in enumerate(SLOT_WORDS):
                rows.append(
                    {
                        "text": f"{frame.format(w=word)}{terminal}",
                        "probe": probe,
                        "frame_index": frame_index,
                        "terminal": terminal,
                        "word": word,
                    }
                )
    return rows


def pair_texts(frames: Sequence[str], terminals: Sequence[str]) -> List[str]:
    return [row["text"] for row in build_pair_rows(frames, terminals)]


def swap_texts(prompts: Sequence[str]) -> List[str]:
    """Replace only the final period, cycling the varied terminals by row."""
    out: List[str] = []
    width = len(VARIED_TERMINALS)
    for index, text in enumerate(prompts):
        if not text.endswith(SHARED_TERMINAL):
            raise ValueError("parent prompt does not end with the shared terminal")
        out.append(text[:-1] + VARIED_TERMINALS[index % width])
    return out


def validate_design() -> None:
    """String-level freeze. Token ids are checked again from the fixture."""
    if len(SLOT_WORDS) != 2 or SLOT_WORDS[0] == SLOT_WORDS[1]:
        raise ValueError("controlled property must be two distinct slot strings")
    if len(SLOT_TOKEN_IDS) != 2 or SLOT_TOKEN_IDS[0] == SLOT_TOKEN_IDS[1]:
        raise ValueError("slot token ids must be two distinct ids")
    if SHARED_TERMINAL_ID in VARIED_TERMINAL_IDS:
        raise ValueError("shared terminal id is in the varied set")
    if len(set(VARIED_TERMINAL_IDS)) != len(VARIED_TERMINAL_IDS):
        raise ValueError("varied terminal ids are not distinct")
    if len(VARIED_TERMINALS) != len(VARIED_TERMINAL_IDS):
        raise ValueError("varied terminal strings and ids differ in length")
    if INTERMEDIATE_LAYER_INDEX == FINAL_LAYER_INDEX:
        raise ValueError("intermediate layer index equals the final layer")
    if INTERMEDIATE_LAYER_INDEX < 0 or INTERMEDIATE_LAYER_INDEX >= EXPECTED_TUPLE_LENGTH:
        raise ValueError("intermediate layer index is outside the hidden-state tuple")
    if FINAL_LAYER_INDEX != EXPECTED_TUPLE_LENGTH - 1:
        raise ValueError("final layer index is not the last hidden state")
    if len(TRAIN_FRAMES) != 16 or len(HELDOUT_FRAMES) != 8:
        raise ValueError("frame split changed")
    if len(set(TRAIN_FRAMES)) != len(TRAIN_FRAMES) or len(set(HELDOUT_FRAMES)) != len(HELDOUT_FRAMES):
        raise ValueError("duplicate frame")
    if set(TRAIN_FRAMES) & set(HELDOUT_FRAMES):
        raise ValueError("train and held-out frames intersect")
    for frame in TRAIN_FRAMES + HELDOUT_FRAMES:
        if frame.count("{w}") != 1 or frame.replace("{w}", "").count("{") or "}" in frame.replace("{w}", ""):
            raise ValueError(f"frame does not have one slot: {frame!r}")
        if frame != frame.strip() or frame.endswith(SHARED_TERMINAL):
            raise ValueError(f"frame is not a bare prefix: {frame!r}")
        if SLOT_WORDS[0] in frame or SLOT_WORDS[1] in frame:
            raise ValueError(f"frame already contains a slot word: {frame!r}")
    c2_train = pair_texts(TRAIN_FRAMES, VARIED_TERMINALS)
    c2_held = pair_texts(HELDOUT_FRAMES, VARIED_TERMINALS)
    c4_train = pair_texts(TRAIN_FRAMES, (SHARED_TERMINAL,))
    c4_held = pair_texts(HELDOUT_FRAMES, (SHARED_TERMINAL,))
    swap_train = swap_texts(PARENT_TRAIN_PROMPTS)
    swap_held = swap_texts(PARENT_HELDOUT_PROMPTS)
    parent = list(PARENT_TRAIN_PROMPTS) + list(PARENT_HELDOUT_PROMPTS)
    for label, left, right in (
        ("pair split", c2_train, c2_held),
        ("shared split", c4_train, c4_held),
        ("pairs vs parent", c2_train + c2_held + c4_train + c4_held, parent),
        ("swap vs pairs", swap_train + swap_held, c2_train + c2_held + c4_train + c4_held),
    ):
        _assert_text_disjoint(left, right, label)
    if len(set(c2_train + c2_held + c4_train + c4_held)) != len(c2_train + c2_held + c4_train + c4_held):
        raise ValueError("duplicate rendered prompt")
    if swap_train == list(PARENT_TRAIN_PROMPTS) or swap_held == list(PARENT_HELDOUT_PROMPTS):
        raise ValueError("terminal swap did not change the parent prompts")
    for original, swapped in zip(list(PARENT_TRAIN_PROMPTS) + list(PARENT_HELDOUT_PROMPTS), swap_train + swap_held):
        if original[:-1] != swapped[:-1] or original[-1] == swapped[-1]:
            raise ValueError("swap changed more than the final character")


def _assert_text_disjoint(left: Sequence[str], right: Sequence[str], label: str) -> None:
    if set(left) & set(right):
        raise ValueError(f"{label}: sets intersect")
    for item in left:
        for other in right:
            if item in other or other in item:
                raise ValueError(f"{label}: containment between {item!r} and {other!r}")


def decide_verdict(
    *,
    genuine: bool,
    condition_1_reproduced: bool,
    distinction_gate: bool,
    geometry_separates: bool,
    independent_target_satisfied: bool,
) -> Tuple[str, str, str]:
    """Return verdict, recommendation, and the first failing reason.

    Authority stays ``RUNTIME_NOT_AUTHORIZED`` on every branch. The runner
    passes ``independent_target_satisfied=False`` because this trial has no
    behavioral target that is not a function of the clustered vector.
    """
    if not genuine:
        return (
            VERDICT_UNRESOLVED,
            RECOMMENDATION_NONE,
            "capture_not_genuine_or_not_reproducible",
        )
    if not condition_1_reproduced:
        return (
            VERDICT_UNRESOLVED,
            RECOMMENDATION_NONE,
            "condition_1_parent_boundary_not_reproduced",
        )
    if distinction_gate and independent_target_satisfied:
        return (
            VERDICT_DISTINCTION_AND_TARGET,
            RECOMMENDATION_RESEARCH_NOTE,
            "distinction_gate_and_independent_target",
        )
    if distinction_gate:
        return (
            VERDICT_DISTINCTION,
            RECOMMENDATION_NONE,
            "distinction_gate_passed_without_independent_target",
        )
    if geometry_separates:
        return (
            VERDICT_GEOMETRY_ONLY,
            RECOMMENDATION_NONE,
            "geometry_separates_symbols_miss_margin",
        )
    return (
        VERDICT_RESEARCH_FAIL,
        RECOMMENDATION_NONE,
        "symbols_and_geometry_missed_controlled_distinction",
    )


def _normalized_frobenius(matrix: np.ndarray, hat: np.ndarray) -> float:
    denom = float(np.linalg.norm(matrix))
    if denom <= 1e-12:
        return 0.0
    return float(np.linalg.norm(matrix - hat) / denom)


def _mean_std(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vals.size < 2:
        raise ValueError("dispersion needs at least two distances")
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "n": int(vals.size),
        "distance": DISTANCE,
        "dispersion": DISPERSION,
    }


def _sqeuclid_matrix(matrix: np.ndarray) -> np.ndarray:
    """Squared Euclidean matrix, the same expansion ``kmeans`` uses for assignment."""
    data = np.asarray(matrix, dtype=np.float64)
    dist = ((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2)
    np.maximum(dist, 0.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def _row_sqeuclid(left: np.ndarray, right: np.ndarray) -> dict:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return _mean_std(np.sum(diff * diff, axis=1))


def _within_condition(matrix: np.ndarray) -> dict:
    dist = _sqeuclid_matrix(matrix)
    return _mean_std(dist[np.triu_indices(dist.shape[0], k=1)])


def _pair_geometry(matrix: np.ndarray) -> dict:
    dist = _sqeuclid_matrix(matrix)
    n = dist.shape[0]
    if n % 2 != 0:
        raise ValueError("pair geometry expects an even row count")
    pair_vals = np.asarray([dist[i, i + 1] for i in range(0, n, 2)], dtype=np.float64)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    for index in range(0, n, 2):
        mask[index, index + 1] = False
    return {
        "within_condition": _mean_std(dist[np.triu_indices(n, k=1)]),
        "within_pair": _mean_std(pair_vals),
        "nonpair": _mean_std(dist[mask]),
    }


def _aligned_terminal_distances(shared: np.ndarray, varied: np.ndarray) -> dict:
    """Each shared-terminal row against the varied-terminal rows of the same frame and word."""
    words = len(SLOT_WORDS)
    n_terminals = len(VARIED_TERMINALS)
    per_frame = n_terminals * words
    if shared.shape[0] % words != 0 or varied.shape[0] != shared.shape[0] * n_terminals:
        raise ValueError("shared and varied matrices are not aligned")
    if shared.shape[1] != varied.shape[1]:
        raise ValueError("aligned distance width mismatch")
    n_frames = shared.shape[0] // words
    values: List[float] = []
    shared64 = np.asarray(shared, dtype=np.float64)
    varied64 = np.asarray(varied, dtype=np.float64)
    for frame in range(n_frames):
        for word in range(words):
            origin = shared64[frame * words + word]
            for terminal in range(n_terminals):
                other = varied64[frame * per_frame + terminal * words + word]
                values.append(float(np.sum((origin - other) ** 2)))
    return _mean_std(np.asarray(values, dtype=np.float64))


def _concentration(labels: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=N_SYMBOLS)
    n = int(labels.shape[0])
    occupied_counts = counts[counts > 0]
    probs = occupied_counts.astype(np.float64) / n
    entropy = float(-(probs * np.log2(probs)).sum())
    if entropy == 0.0:
        entropy = 0.0
    max_count = int(counts.max())
    return {
        "counts": [int(value) for value in counts],
        "n": n,
        "max_count": max_count,
        "max_fraction": max_count / n,
        "occupied": int(occupied_counts.shape[0]),
        "entropy_bits": entropy,
        "effective_symbols": float(2.0 ** entropy),
        "empty": int(np.sum(counts == 0)),
    }


def _token_concentration(token_ids: np.ndarray) -> dict:
    values, counts = np.unique(np.asarray(token_ids, dtype=np.int64), return_counts=True)
    n = int(token_ids.shape[0])
    probs = counts.astype(np.float64) / n
    entropy = float(-(probs * np.log2(probs)).sum())
    if entropy == 0.0:
        entropy = 0.0
    max_count = int(counts.max())
    order = np.argsort(values)
    return {
        "ids": [int(values[index]) for index in order],
        "counts": [int(counts[index]) for index in order],
        "n": n,
        "max_count": max_count,
        "max_fraction": max_count / n,
        "occupied": int(values.shape[0]),
        "entropy_bits": entropy,
    }


def _fit_symbols(train_activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Production farthest-first k-means on training activations only.

    The ``rng`` argument is passed and ignored by ``_farthest_first``. Probe
    labels and token ids are not arguments. This call does not edit
    ``clustering.py``.
    """
    matrix = np.asarray(train_activations, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_HIDDEN_SIZE:
        raise ValueError("symbol fit expects the activation width only")
    labels, centroids = kmeans(matrix, N_SYMBOLS, rng=np.random.default_rng(LEARN_SEED))
    return np.asarray(labels, dtype=np.int64), np.asarray(centroids, dtype=np.float64)


def _majority_label(labels: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    if values.size == 0:
        return 0
    winners = values[counts == counts.max()]
    return int(np.min(winners))


def _fit_map(symbols: np.ndarray, probes: np.ndarray) -> Tuple[Dict[int, int], int]:
    """Train-only symbol to probe map. Ties use the smaller probe id.

    A symbol with no training row maps to the global training majority.
    """
    default = _majority_label(probes)
    mapping: Dict[int, int] = {}
    for symbol in np.unique(symbols):
        members = probes[symbols == symbol]
        mapping[int(symbol)] = _majority_label(members) if members.size else default
    return mapping, default


def _predict(mapping: Dict[int, int], default: int, symbols: np.ndarray) -> Tuple[np.ndarray, int]:
    missing = 0
    pred = np.empty(symbols.shape[0], dtype=np.int64)
    for index, symbol in enumerate(symbols.tolist()):
        key = int(symbol)
        if key not in mapping:
            pred[index] = default
            missing += 1
        else:
            pred[index] = mapping[key]
    return pred, missing


def _accuracy_block(pred: np.ndarray, probes: np.ndarray, unseen: int) -> dict:
    n = int(probes.shape[0])
    n_correct = int(np.sum(pred == probes))
    return {
        "n_correct": n_correct,
        "n": n,
        "accuracy": n_correct / n,
        "unseen_symbol_rows": unseen,
    }


def _pair_split_block(labels: np.ndarray) -> dict:
    n_pairs = int(labels.shape[0] // 2)
    n_split = 0
    for index in range(n_pairs):
        if int(labels[2 * index]) != int(labels[2 * index + 1]):
            n_split += 1
    return {
        "n_split": n_split,
        "n_pairs": n_pairs,
        "rate": n_split / n_pairs,
    }


def _mapped_from_symbols(
    train_symbols: np.ndarray,
    held_symbols: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> dict:
    mapping, default = _fit_map(train_symbols, train_y)
    pred, unseen = _predict(mapping, default, held_symbols)
    block = _accuracy_block(pred, held_y, unseen)
    block["pair_split"] = _pair_split_block(held_symbols)
    return block


def _train_mean_baseline(
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> dict:
    mean = np.asarray(train_x, dtype=np.float64).mean(axis=0, keepdims=True)
    train_symbols = assign_squared_euclidean(np.asarray(train_x, dtype=np.float64), mean)
    held_symbols = assign_squared_euclidean(np.asarray(held_x, dtype=np.float64), mean)
    block = _mapped_from_symbols(train_symbols, held_symbols, train_y, held_y)
    block["n_centroids"] = 1
    return block


def _shuffled_baseline(
    train_symbols: np.ndarray,
    held_symbols: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> dict:
    rng = np.random.default_rng(SHUFFLE_ASSIGNMENT_SEED)
    shuffled = held_symbols[rng.permutation(held_symbols.shape[0])]
    return _mapped_from_symbols(train_symbols, shuffled, train_y, held_y)


def _token_baseline(
    train_token: np.ndarray,
    held_token: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> dict:
    return _mapped_from_symbols(train_token, held_token, train_y, held_y)


def _nearest_train_baseline(
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> dict:
    """One nearest training row under squared Euclidean distance. Ties take the smallest index."""
    train64 = np.asarray(train_x, dtype=np.float64)
    held64 = np.asarray(held_x, dtype=np.float64)
    dist = ((held64[:, None, :] - train64[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(dist, axis=1)
    pred = np.asarray(train_y, dtype=np.int64)[nearest]
    block = _accuracy_block(pred, held_y, 0)
    # Pair disagreement of the inherited probe, not a codebook symbol.
    n_pairs = int(held_y.shape[0] // 2)
    n_split = sum(int(pred[2 * i] != pred[2 * i + 1]) for i in range(n_pairs))
    block["pair_prediction_disagreement"] = {
        "n_split": n_split,
        "n_pairs": n_pairs,
        "rate": n_split / n_pairs,
    }
    return block


def _beats_margin(accuracy: float, baseline: float) -> bool:
    if not np.isfinite(accuracy) or not np.isfinite(baseline):
        return False
    return accuracy >= baseline + DISTINCTION_MARGIN


def _timed_fit(train_activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    labels, centroids = _fit_symbols(train_activations)
    return labels, centroids, time.perf_counter() - start


def _learn_and_assign(
    train_x: np.ndarray,
    held_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    labels, centroids, learn_seconds = _timed_fit(train_x)
    start = time.perf_counter()
    held_labels = assign_squared_euclidean(np.asarray(held_x, dtype=np.float64), centroids)
    return labels, centroids, held_labels, learn_seconds, time.perf_counter() - start


def _as_f64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def _as_i64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.int64)


def _checksums(arrays: Dict[str, np.ndarray]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in sorted(arrays):
        array = arrays[key]
        if np.issubdtype(array.dtype, np.floating):
            out[key] = float32_sha256(array)
        elif np.issubdtype(array.dtype, np.integer):
            out[key] = int32_sha256(array)
        else:
            raise ValueError(f"unsupported dtype for {key}: {array.dtype}")
    return out


def _load_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _token_rows(ids: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
    rows: List[np.ndarray] = []
    for index, length in enumerate(lengths.tolist()):
        if length < 1 or length > ids.shape[1]:
            raise ValueError("token length is outside the padded row")
        rows.append(np.asarray(ids[index, : int(length)], dtype=np.int64))
    return rows


def _pair_token_failures(prefix: str, arrays: Dict[str, np.ndarray], terminals: Sequence[int]) -> List[str]:
    failures: List[str] = []
    for split in ("train", "heldout"):
        ids = arrays.get(f"{prefix}_{split}_token_ids")
        lengths = arrays.get(f"{prefix}_{split}_token_lengths")
        last = arrays.get(f"{prefix}_{split}_last_token_id")
        probes = arrays.get(f"{prefix}_{split}_probe")
        slots = arrays.get(f"{prefix}_{split}_slot_token_id")
        if any(item is None for item in (ids, lengths, last, probes, slots)):
            failures.append(f"tokens_{prefix}_{split}")
            continue
        if ids.dtype != np.int32 or lengths.dtype != np.int32 or last.dtype != np.int32:
            failures.append(f"token_dtype_{prefix}_{split}")
            continue
        if probes.dtype != np.int32 or slots.dtype != np.int32:
            failures.append(f"probe_dtype_{prefix}_{split}")
            continue
        n = int(last.shape[0])
        if n % 2 != 0 or probes.shape[0] != n or slots.shape[0] != n or ids.shape[0] != n:
            failures.append(f"pair_shape_{prefix}_{split}")
            continue
        expected_last = np.asarray(
            [terminals[(index // 2) % len(terminals)] for index in range(n)],
            dtype=np.int64,
        )
        # Frame-major, terminal-minor, word-fastest: each terminal is repeated twice.
        if not np.array_equal(_as_i64(last), expected_last):
            failures.append(f"terminal_pattern_{prefix}_{split}")
        if not np.array_equal(_as_i64(probes), np.tile(np.arange(2), n // 2)):
            failures.append(f"probe_pattern_{prefix}_{split}")
        try:
            sequences = _token_rows(ids, lengths)
        except ValueError:
            failures.append(f"token_length_{prefix}_{split}")
            continue
        for index in range(0, n, 2):
            left, right = sequences[index], sequences[index + 1]
            if left.shape != right.shape or int(left[-1]) != int(right[-1]):
                failures.append(f"pair_terminal_{prefix}_{split}")
                break
            diff = np.flatnonzero(left != right)
            if diff.tolist() != [SLOT_INDEX]:
                failures.append(f"pair_diff_{prefix}_{split}")
                break
            if int(left[SLOT_INDEX]) != SLOT_TOKEN_IDS[0] or int(right[SLOT_INDEX]) != SLOT_TOKEN_IDS[1]:
                failures.append(f"slot_id_{prefix}_{split}")
                break
            if int(slots[index]) != SLOT_TOKEN_IDS[0] or int(slots[index + 1]) != SLOT_TOKEN_IDS[1]:
                failures.append(f"slot_column_{prefix}_{split}")
                break
        unique, counts = np.unique(last, return_counts=True)
        if split == "train" or split == "heldout":
            if not np.all(counts == counts[0]):
                failures.append(f"terminal_imbalance_{prefix}_{split}")
        if set(int(item) for item in unique) != set(terminals):
            failures.append(f"terminal_set_{prefix}_{split}")
    return failures


def _activation_failures(prefix: str, arrays: Dict[str, np.ndarray], n_train: int, n_held: int) -> List[str]:
    failures: List[str] = []
    for split, n in (("train", n_train), ("heldout", n_held)):
        array = arrays.get(f"{prefix}_{split}_activations")
        if array is None or tuple(array.shape) != (n, EXPECTED_HIDDEN_SIZE) or array.dtype != np.float32:
            failures.append(f"shape_{prefix}_{split}")
        elif not np.isfinite(array).all():
            failures.append(f"nonfinite_{prefix}_{split}")
    return failures


def assess_boundary(
    npz_path: Path = FIXTURE_NPZ,
    meta_path: Path = FIXTURE_META,
    parent_npz: Path = PARENT_NPZ,
    parent_meta: Path = PARENT_META,
) -> dict:
    """Check the committed capture. A failed check is not scored as genuine."""
    failures: List[str] = []
    if not npz_path.is_file() or not meta_path.is_file():
        return {
            "genuine": False,
            "blocker": "fixture_missing",
            "failures": ["fixture_missing"],
        }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = _load_arrays(npz_path)
    parent = assess_parent_fixture(parent_npz, parent_meta)
    if not parent["genuine"]:
        failures.append("parent_fixture")
    expected_meta = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "url": MODEL_URL,
        "weight_sha256": WEIGHT_SHA256,
        "forward_dtype": FORWARD_DTYPE,
        "stored_dtype": STORED_DTYPE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "final_layer_index": FINAL_LAYER_INDEX,
        "condition_3_layer_index": INTERMEDIATE_LAYER_INDEX,
        "condition_3_token_coordinate": CONDITION_3_TOKEN_COORDINATE,
        "hidden_states_tuple_length": EXPECTED_TUPLE_LENGTH,
        "torch_version": EXPECTED_TORCH,
        "transformers_version": EXPECTED_TRANSFORMERS,
        "design_fingerprint": design_fingerprint(),
        "parent_prompt_fingerprint": parent_prompt_fingerprint(),
        "controlled_property": CONTROLLED_PROPERTY,
        "download_seconds": 0.0,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            failures.append(f"meta_{key}")
    for key, expected in (
        ("c2_train_prompts", pair_texts(TRAIN_FRAMES, VARIED_TERMINALS)),
        ("c2_heldout_prompts", pair_texts(HELDOUT_FRAMES, VARIED_TERMINALS)),
        ("c4_train_prompts", pair_texts(TRAIN_FRAMES, (SHARED_TERMINAL,))),
        ("c4_heldout_prompts", pair_texts(HELDOUT_FRAMES, (SHARED_TERMINAL,))),
        ("swap_train_prompts", swap_texts(PARENT_TRAIN_PROMPTS)),
        ("swap_heldout_prompts", swap_texts(PARENT_HELDOUT_PROMPTS)),
    ):
        if list(meta.get(key) or []) != expected:
            failures.append(key)
    recomputed = _checksums(arrays)
    if recomputed != dict(meta.get("checksums") or {}):
        failures.append("checksum_mismatch")
    n2_train = len(TRAIN_FRAMES) * len(VARIED_TERMINALS) * len(SLOT_WORDS)
    n2_held = len(HELDOUT_FRAMES) * len(VARIED_TERMINALS) * len(SLOT_WORDS)
    n4_train = len(TRAIN_FRAMES) * len(SLOT_WORDS)
    n4_held = len(HELDOUT_FRAMES) * len(SLOT_WORDS)
    failures.extend(_activation_failures("c2", arrays, n2_train, n2_held))
    failures.extend(_activation_failures("c3", arrays, n2_train, n2_held))
    failures.extend(_activation_failures("c4", arrays, n4_train, n4_held))
    failures.extend(_activation_failures("swap", arrays, len(PARENT_TRAIN_PROMPTS), len(PARENT_HELDOUT_PROMPTS)))
    for name, expected in (
        ("c2_layer_index", FINAL_LAYER_INDEX),
        ("c3_layer_index", INTERMEDIATE_LAYER_INDEX),
        ("c4_layer_index", FINAL_LAYER_INDEX),
        ("swap_layer_index", FINAL_LAYER_INDEX),
    ):
        array = arrays.get(name)
        if array is None or int(np.asarray(array).size) != 1 or int(np.asarray(array).reshape(-1)[0]) != expected:
            failures.append(name)
    failures.extend(_pair_token_failures("c2", arrays, VARIED_TERMINAL_IDS))
    failures.extend(_pair_token_failures("c4", arrays, (SHARED_TERMINAL_ID,)))
    if parent["genuine"]:
        failures.extend(_swap_failures(arrays, parent["arrays"]))
    if "c2_train_activations" in arrays and "c3_train_activations" in arrays:
        if np.array_equal(arrays["c2_train_activations"], arrays["c3_train_activations"]):
            failures.append("condition_3_activations_identical_to_condition_2")
        if np.array_equal(arrays["c2_heldout_activations"], arrays["c3_heldout_activations"]):
            failures.append("condition_3_activations_identical_to_condition_2")
    for key in (
        "load_seconds",
        "capture_condition_2_forward_seconds",
        "capture_condition_2_slice_seconds",
        "capture_condition_3_slice_seconds",
        "capture_condition_4_seconds",
        "capture_parent_swap_seconds",
    ):
        value = meta.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            failures.append(key)
    genuine = not failures
    return {
        "genuine": genuine,
        "blocker": None if genuine else failures[0],
        "failures": failures,
        "meta": meta if genuine else None,
        "arrays": arrays if genuine else None,
        "parent_arrays": parent["arrays"] if parent["genuine"] else None,
        "checksums": recomputed,
    }


def _swap_failures(arrays: Dict[str, np.ndarray], parent_arrays: Dict[str, np.ndarray]) -> List[str]:
    failures: List[str] = []
    for split, n in (("train", len(PARENT_TRAIN_PROMPTS)), ("heldout", len(PARENT_HELDOUT_PROMPTS))):
        ids = arrays.get(f"swap_{split}_token_ids")
        lengths = arrays.get(f"swap_{split}_token_lengths")
        last = arrays.get(f"swap_{split}_last_token_id")
        parent_ids = parent_arrays.get(f"{split}_token_ids")
        parent_lengths = parent_arrays.get(f"{split}_token_lengths")
        parent_last = parent_arrays.get(f"{split}_last_token_id")
        if any(item is None for item in (ids, lengths, last, parent_ids, parent_lengths, parent_last)):
            failures.append(f"swap_tokens_{split}")
            continue
        if int(last.shape[0]) != n or not np.array_equal(lengths, parent_lengths):
            failures.append(f"swap_length_{split}")
            continue
        expected = np.asarray(
            [VARIED_TERMINAL_IDS[index % len(VARIED_TERMINAL_IDS)] for index in range(n)],
            dtype=np.int64,
        )
        if not np.array_equal(_as_i64(last), expected):
            failures.append(f"swap_terminal_pattern_{split}")
        if set(int(item) for item in parent_last.tolist()) != {SHARED_TERMINAL_ID}:
            failures.append(f"parent_terminal_{split}")
        try:
            swapped = _token_rows(ids, lengths)
            original = _token_rows(parent_ids, parent_lengths)
        except ValueError:
            failures.append(f"swap_token_length_{split}")
            continue
        for index, (left, right) in enumerate(zip(original, swapped)):
            if left.shape != right.shape or not np.array_equal(left[:-1], right[:-1]):
                failures.append(f"swap_prefix_{split}")
                break
            if int(left[-1]) != SHARED_TERMINAL_ID or int(right[-1]) != int(expected[index]):
                failures.append(f"swap_last_{split}")
                break
    return failures


def _cost_block(capture: float, learn: float, translate: float, n_capture: int, n_train: int, n_held: int) -> dict:
    return {
        "capture_seconds": capture,
        "learn_seconds": learn,
        "translate_seconds": translate,
        "per_observation": {
            "capture": capture / n_capture,
            "learn": learn / n_train,
            "translate": translate / n_held,
        },
        "n_capture": n_capture,
        "n_train": n_train,
        "n_held": n_held,
    }


def _distinction_report(
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_labels: np.ndarray,
    held_labels: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
    train_token: np.ndarray,
    held_token: np.ndarray,
) -> dict:
    symbol = _mapped_from_symbols(train_labels, held_labels, train_y, held_y)
    baselines = {
        "train_mean": _train_mean_baseline(train_x, held_x, train_y, held_y),
        "shuffled_assignment": _shuffled_baseline(train_labels, held_labels, train_y, held_y),
        "token_identity": _token_baseline(train_token, held_token, train_y, held_y),
        "nearest_train_row": _nearest_train_baseline(train_x, held_x, train_y, held_y),
    }
    margins = {
        name: symbol["accuracy"] - block["accuracy"] for name, block in baselines.items()
    }
    beats = all(_beats_margin(symbol["accuracy"], block["accuracy"]) for block in baselines.values())
    nn_accuracy = baselines["nearest_train_row"]["accuracy"]
    return {
        "symbol": symbol,
        "baselines": baselines,
        "margin_over_baseline": margins,
        "required_margin": DISTINCTION_MARGIN,
        "beats_gate": beats,
        "nearest_train_clears_floor": nn_accuracy >= GEOMETRY_NN_FLOOR,
        "geometry_floor": GEOMETRY_NN_FLOOR,
        "gate": DISTINCTION_GATE,
    }


def _heldout_on_empty(train_labels: np.ndarray, held_labels: np.ndarray) -> int:
    counts = np.bincount(train_labels, minlength=N_SYMBOLS)
    empty = set(int(index) for index in np.flatnonzero(counts == 0))
    return int(sum(int(label) in empty for label in held_labels))


def run_activation_boundary(
    npz_path: Path = FIXTURE_NPZ,
    meta_path: Path = FIXTURE_META,
    parent_npz: Path = PARENT_NPZ,
    parent_meta: Path = PARENT_META,
) -> dict:
    """Score the frozen capture. Probe columns are not passed to ``kmeans``."""
    validate_design()
    if INDEPENDENT_TARGET_STATUS != "INDEPENDENT_TARGET_UNAVAILABLE":
        raise RuntimeError("this trial has no independent behavioral target")
    assessment = assess_boundary(npz_path, meta_path, parent_npz, parent_meta)
    base = {
        "status": STATUS,
        "authority": RUNTIME_AUTHORITY,
        "parent_commit": PARENT_COMMIT,
        "baseline_commit": BASELINE_COMMIT,
        "entering_verdict": ENTERING_VERDICT,
        "entering_recommendation": ENTERING_RECOMMENDATION,
        "probe_used_in_fit": False,
        "gloss_used_as_evidence": False,
        "independent_target": {
            "status": INDEPENDENT_TARGET_STATUS,
            "reason": INDEPENDENT_TARGET_REASON,
            "satisfied": False,
        },
    }
    if not assessment["genuine"]:
        verdict, recommendation, reason = decide_verdict(
            genuine=False,
            condition_1_reproduced=False,
            distinction_gate=False,
            geometry_separates=False,
            independent_target_satisfied=False,
        )
        base.update(
            {
                "genuine": False,
                "blocker": assessment["blocker"],
                "verdict": verdict,
                "recommendation": recommendation,
                "reason": reason,
                "explicit_failures": list(assessment["failures"]),
                "condition_1_reproduced": False,
            }
        )
        return base

    meta = assessment["meta"]
    arrays = assessment["arrays"]
    parent_arrays = assessment["parent_arrays"]
    parent_meta_obj = json.loads(Path(parent_meta).read_text(encoding="utf-8"))
    parent_capture_seconds = float(parent_meta_obj["capture_seconds"])
    c1_train = _as_f64(parent_arrays["train_activations"])
    c1_held = _as_f64(parent_arrays["heldout_activations"])
    # Warm one fit so the timed calls are the clustering itself.
    _fit_symbols(c1_train)

    def _activation_bundle(prefix: str) -> dict:
        train_y = arrays.get(f"{prefix}_train_probe")
        held_y = arrays.get(f"{prefix}_heldout_probe")
        return {
            "train_x": _as_f64(arrays[f"{prefix}_train_activations"]),
            "held_x": _as_f64(arrays[f"{prefix}_heldout_activations"]),
            "train_token": _as_i64(arrays[f"{prefix}_train_last_token_id"]),
            "held_token": _as_i64(arrays[f"{prefix}_heldout_last_token_id"]),
            "train_y": None if train_y is None else _as_i64(train_y),
            "held_y": None if held_y is None else _as_i64(held_y),
        }

    # Activation matrices only. Probe vectors stay in *_y and are not fit features.
    c1 = {
        "train_x": c1_train,
        "held_x": c1_held,
        "train_token": _as_i64(parent_arrays["train_last_token_id"]),
        "held_token": _as_i64(parent_arrays["heldout_last_token_id"]),
        "train_y": None,
        "held_y": None,
    }
    c2 = _activation_bundle("c2")
    c4 = _activation_bundle("c4")
    swap = _activation_bundle("swap")
    # Condition 3 uses condition 2's token ids and probe labels at the same rows.
    c3 = {
        "train_x": _as_f64(arrays["c3_train_activations"]),
        "held_x": _as_f64(arrays["c3_heldout_activations"]),
        "train_token": c2["train_token"],
        "held_token": c2["held_token"],
        "train_y": c2["train_y"],
        "held_y": c2["held_y"],
    }

    learned = {}
    for name, bundle in (("condition_1", c1), ("condition_2", c2), ("condition_3", c3), ("condition_4", c4), ("parent_terminal_swap", swap)):
        labels, centroids, held_labels, learn_seconds, translate_seconds = _learn_and_assign(
            bundle["train_x"], bundle["held_x"]
        )
        learned[name] = {
            "train_labels": labels,
            "centroids": centroids,
            "held_labels": held_labels,
            "learn_seconds": learn_seconds,
            "translate_seconds": translate_seconds,
        }

    c1_train_labels = learned["condition_1"]["train_labels"]
    c1_held_labels = learned["condition_1"]["held_labels"]
    c1_centroids = learned["condition_1"]["centroids"]
    train_counts = np.bincount(c1_train_labels, minlength=N_SYMBOLS)
    held_counts = np.bincount(c1_held_labels, minlength=N_SYMBOLS)
    train_residual = _normalized_frobenius(c1_train, c1_centroids[c1_train_labels])
    prototype_residual = _normalized_frobenius(c1_held, c1_centroids[c1_held_labels])
    mean_hat = np.repeat(c1_train.mean(axis=0)[None, :], c1_held.shape[0], axis=0)
    mean_residual = _normalized_frobenius(c1_held, mean_hat)
    shuffle_rng = np.random.default_rng(SHUFFLE_ASSIGNMENT_SEED)
    shuffled_labels = c1_held_labels[shuffle_rng.permutation(c1_held_labels.shape[0])]
    shuffled_residual = _normalized_frobenius(c1_held, c1_centroids[shuffled_labels])
    reproduced = (
        train_counts.tolist() == list(PARENT_TRAIN_COUNTS)
        and held_counts.tolist() == list(PARENT_HELDOUT_COUNTS)
        and abs(train_residual - PARENT_TRAIN_RESIDUAL) <= REPRODUCTION_ABS
        and abs(prototype_residual - PARENT_HELDOUT_PROTOTYPE_RESIDUAL) <= REPRODUCTION_ABS
        and abs(mean_residual - PARENT_HELDOUT_TRAIN_MEAN_RESIDUAL) <= REPRODUCTION_ABS
        and abs(shuffled_residual - PARENT_HELDOUT_SHUFFLED_RESIDUAL) <= REPRODUCTION_ABS
    )
    explicit: List[str] = []
    if not reproduced:
        explicit.append("condition_1_parent_boundary_not_reproduced")

    def _condition_body(name: str, bundle: dict, capture_seconds: float, n_capture: int) -> dict:
        item = learned[name]
        train_x = bundle["train_x"]
        held_x = bundle["held_x"]
        body = {
            "train_concentration": _concentration(item["train_labels"]),
            "heldout_concentration": _concentration(item["held_labels"]),
            "heldout_on_empty_train_clusters": _heldout_on_empty(item["train_labels"], item["held_labels"]),
            "token_train": _token_concentration(bundle["train_token"]),
            "token_heldout": _token_concentration(bundle["held_token"]),
            "within_train": _within_condition(train_x),
            "within_heldout": _within_condition(held_x),
            "cost": _cost_block(
                capture_seconds,
                item["learn_seconds"],
                item["translate_seconds"],
                n_capture,
                int(train_x.shape[0]),
                int(held_x.shape[0]),
            ),
        }
        if bundle["train_y"] is not None:
            body["pair_geometry_train"] = _pair_geometry(train_x)
            body["pair_geometry_heldout"] = _pair_geometry(held_x)
            body["distinction"] = _distinction_report(
                train_x,
                held_x,
                item["train_labels"],
                item["held_labels"],
                bundle["train_y"],
                bundle["held_y"],
                bundle["train_token"],
                bundle["held_token"],
            )
        return body

    n2 = c2["train_x"].shape[0] + c2["held_x"].shape[0]
    n4 = c4["train_x"].shape[0] + c4["held_x"].shape[0]
    n_swap = swap["train_x"].shape[0] + swap["held_x"].shape[0]
    n1 = c1_train.shape[0] + c1_held.shape[0]
    # Condition 1 capture cost is the parent fixture's tokenize-plus-forward time.
    # It is not recomputed here. Conditions 2 and 3 share one forward; condition 3's
    # capture figure is only the extra slice of hidden_states index 3.
    condition_1 = _condition_body("condition_1", c1, parent_capture_seconds, n1)
    condition_1["reproduction"] = {
        "matched": reproduced,
        "train_counts": [int(value) for value in train_counts],
        "heldout_counts": [int(value) for value in held_counts],
        "train_residual": train_residual,
        "heldout_prototype_residual": prototype_residual,
        "heldout_train_mean_residual": mean_residual,
        "heldout_shuffled_assignment_residual": shuffled_residual,
        "expected_train_counts": list(PARENT_TRAIN_COUNTS),
        "expected_heldout_counts": list(PARENT_HELDOUT_COUNTS),
    }
    condition_2 = _condition_body(
        "condition_2",
        c2,
        float(meta["capture_condition_2_forward_seconds"]) + float(meta["capture_condition_2_slice_seconds"]),
        n2,
    )
    condition_3 = _condition_body(
        "condition_3",
        c3,
        float(meta["capture_condition_3_slice_seconds"]),
        n2,
    )
    condition_4 = _condition_body("condition_4", c4, float(meta["capture_condition_4_seconds"]), n4)
    swap_body = _condition_body("parent_terminal_swap", swap, float(meta["capture_parent_swap_seconds"]), n_swap)

    between = {
        "condition_2_vs_condition_3_train": _row_sqeuclid(c2["train_x"], c3["train_x"]),
        "condition_2_vs_condition_3_heldout": _row_sqeuclid(c2["held_x"], c3["held_x"]),
        "condition_4_vs_condition_2_train": _aligned_terminal_distances(c4["train_x"], c2["train_x"]),
        "condition_4_vs_condition_2_heldout": _aligned_terminal_distances(c4["held_x"], c2["held_x"]),
        "condition_1_vs_parent_terminal_swap_train": _row_sqeuclid(c1["train_x"], swap["train_x"]),
        "condition_1_vs_parent_terminal_swap_heldout": _row_sqeuclid(c1["held_x"], swap["held_x"]),
    }
    c1_fraction = condition_1["heldout_concentration"]["max_fraction"]
    swap_fraction = swap_body["heldout_concentration"]["max_fraction"]
    drop = c1_fraction - swap_fraction
    prefixes_match = "swap_prefix_train" not in assessment["failures"] and "swap_prefix_heldout" not in assessment["failures"]
    c1_token_n = condition_1["token_heldout"]["occupied"]
    swap_token_n = swap_body["token_heldout"]["occupied"]
    claim_allowed = bool(
        prefixes_match
        and c1_token_n == 1
        and swap_token_n >= len(VARIED_TERMINAL_IDS)
        and drop >= TERMINAL_CONCENTRATION_DROP
    )
    distinction_gate = any(
        item["distinction"]["beats_gate"] for item in (condition_2, condition_3, condition_4)
    )
    geometry_separates = any(
        item["distinction"]["nearest_train_clears_floor"] for item in (condition_2, condition_3, condition_4)
    )
    # Hard-coded. No score in this function can flip the independent target on.
    independent_target_satisfied = False
    verdict, recommendation, reason = decide_verdict(
        genuine=True,
        condition_1_reproduced=reproduced,
        distinction_gate=distinction_gate,
        geometry_separates=geometry_separates,
        independent_target_satisfied=independent_target_satisfied,
    )
    return {
        **base,
        "genuine": True,
        "blocker": None,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "design_fingerprint": design_fingerprint(),
        "controlled_property": CONTROLLED_PROPERTY,
        "distance": DISTANCE,
        "dispersion": DISPERSION,
        "final_layer_index": FINAL_LAYER_INDEX,
        "intermediate_layer_index": INTERMEDIATE_LAYER_INDEX,
        "condition_3_token_coordinate": CONDITION_3_TOKEN_COORDINATE,
        "n_symbols": N_SYMBOLS,
        "distinction_margin": DISTINCTION_MARGIN,
        "geometry_nn_floor": GEOMETRY_NN_FLOOR,
        "terminal_concentration_drop": TERMINAL_CONCENTRATION_DROP,
        "checksums": assessment["checksums"],
        "condition_1": condition_1,
        "condition_2": condition_2,
        "condition_3": condition_3,
        "condition_4": condition_4,
        "parent_terminal_swap": swap_body,
        "between_condition": between,
        "terminal_claim": {
            "allowed": claim_allowed,
            "gate": TERMINAL_CLAIM_GATE,
            "required_drop": TERMINAL_CONCENTRATION_DROP,
            "heldout_max_fraction_condition_1": c1_fraction,
            "heldout_max_fraction_swap": swap_fraction,
            "heldout_fraction_drop": drop,
            "condition_1_terminal_occupied": c1_token_n,
            "swap_terminal_occupied": swap_token_n,
            "prefixes_differ_only_at_final_token": prefixes_match,
        },
        "distinction_gate": distinction_gate,
        "geometry_separates": geometry_separates,
        "condition_1_reproduced": reproduced,
        "verdict": verdict,
        "recommendation": recommendation,
        "reason": reason,
        "explicit_failures": explicit,
        "capture_stack": {
            "torch": meta.get("torch_version"),
            "transformers": meta.get("transformers_version"),
            "download_seconds": meta.get("download_seconds"),
            "load_seconds": meta.get("load_seconds"),
            "shared_forward_seconds": meta.get("capture_condition_2_forward_seconds"),
        },
        "peak_memory": {
            "capture_rss_kb": meta.get("capture_peak_rss_kb"),
            "capture_tracemalloc_bytes": meta.get("capture_tracemalloc_peak_bytes"),
        },
    }
