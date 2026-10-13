"""Symbol-plus-residual distinguishability. Research only.

An inexpensive residual is attached to a production k-means symbol. The
question is whether that residual preserves a controlled slot distinction
that centroid assignment collapses. This module does not download weights,
does not run a model, and does not change the production encoder, k-means,
structural certificate, taus, CLI, or package version.

Admission constants above ``run_symbol_residual`` were fixed before any
held-out slot score in this trial was computed. Probe labels are not an
input to k-means, to the residual scale, or to the quantizer. A gloss is
not an input to the verdict. Nearest-row and full-vector accuracies are
references. They do not admit the residual.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neuralese.activation_source import (
    EXPECTED_HIDDEN_SIZE,
    assign_squared_euclidean,
    float32_sha256,
)
from neuralese.clustering import kmeans

# Status of this trial. Not a production admission.
STATUS = "RESEARCH_ONLY"
PLAN_STATUS = "PLAN_ONLY"
PARENT_COMMIT = "81077ec"
PARENT_PR = 5
ENTERING_VERDICT = "GEOMETRY_ONLY"
ENTERING_RECOMMENDATION = "NONE"
RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"

MODEL_ID = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
WEIGHT_SHA256 = "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"
FINAL_LAYER_INDEX = 6
FORWARD_DTYPE = "float32"
STORED_DTYPE = "float32"

# Controlled property. A probe only. It is not an embedding feature, not a
# cluster feature, not a residual feature, and not a model-selection target.
CONTROLLED_PROPERTY = (
    "binary identity of one subject-slot word in a fixed frame; "
    "probe 0 is the string cat; probe 1 is the string dog; "
    "labels are a train-only majority map applied after assignment"
)
TIE_BREAK = "smaller_probe_id"
N_SYMBOLS = 8
LEARN_SEED = 0
HIDDEN_SIZE = EXPECTED_HIDDEN_SIZE

# Frozen codec. Chosen before held-out scores. Do not edit these to fit a result.
#
# Precision is signed 4-bit, codes in [-7, 7], one shared train-only scale
# per coordinate. Eight-bit residuals plus a symbol byte are 129 bytes.
# One quarter of the float32 vector is 128 bytes, so 8-bit misses the
# predeclared budget. Four-bit is the widest signed packing that lands on
# a nibble, keeps a separate symbol byte, and still satisfies that budget.
# The scale is the per-coordinate maximum absolute train residual. A zero
# train coordinate uses scale 1 so the code is 0. Held-out rows reuse this
# scale and the train centroids. Nothing in the codec is refit on held-out
# rows. No hash arm, no PCA, no SVD.
RESIDUAL_BITS = 4
RESIDUAL_QMAX = 7
RESIDUAL_DEFINITION = "activation_minus_assigned_centroid"
SCALE_RULE = "per_coordinate_max_abs_train_residual_zero_maps_to_one"
QUANTIZE_RULE = "rint_then_clip_to_signed_qmax_train_scale_only"
SYMBOL_ID_BYTES = 1
RESIDUAL_PAYLOAD_BYTES = (HIDDEN_SIZE * RESIDUAL_BITS) // 8
BYTES_PER_ROW = SYMBOL_ID_BYTES + RESIDUAL_PAYLOAD_BYTES
FULL_VECTOR_BYTES = HIDDEN_SIZE * 4
FLOAT32_BYTES = 4
STORAGE_RATIO_MAX = 0.25
SHARED_FLOAT_BYTES = 8
SHARED_CENTROID_BYTES = N_SYMBOLS * HIDDEN_SIZE * SHARED_FLOAT_BYTES
SHARED_SCALE_BYTES = HIDDEN_SIZE * SHARED_FLOAT_BYTES

PACKING_SYMBOL = "uint8 symbol id; 3 information bits stored in 8 bits; no residual payload"
PACKING_SYMBOL_RESIDUAL = (
    "uint8 symbol id, then 64 bytes of signed int4; "
    "dimension 2i in the low nibble and 2i+1 in the high nibble; "
    "nibble is two's complement; frozen codes occupy [-7, 7]"
)
PACKING_FULL_VECTOR = "float32, 128 dimensions, 4 bytes each"
PACKING_PAYLOAD_ONLY = "64 bytes of signed int4 residuals; the symbol byte is omitted"
PACKING_NEAREST = (
    "not a compressed code; the query retained for the comparison is the float32 vector"
)

# Admission. Continuity with the activation-boundary margin of 0.10.
# Not a retune. Storage is the predeclared advantage. Wall-clock time is
# recorded and is not a threshold. Shared centroids and scales are reported
# and are not added into the per-row payload the gate compares.
ACCURACY_MARGIN = 0.10
ACCURACY_RULE = (
    "symbol-plus-residual held-out slot accuracy is at least "
    "ACCURACY_MARGIN above the symbol-only accuracy on the same condition"
)
SHUFFLE_FALL = 0.10
SHUFFLE_FALL_RULE = (
    "within-class residual-shuffle held-out slot accuracy is at least "
    "SHUFFLE_FALL below the intact symbol-plus-residual accuracy "
    "on the same condition, as an absolute difference"
)
STORAGE_RULE = (
    "stored bytes per row of the symbol-plus-residual payload are at most "
    "STORAGE_RATIO_MAX times FULL_VECTOR_BYTES"
)
ADVANTAGE_KIND = "bytes_per_row"
SHUFFLE_ASSIGNMENT_SEED = 0
WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED = 1
WITHIN_CLASS_SHUFFLE_RULE = (
    "permute held-out residual payloads among rows that share an assigned "
    "symbol; symbol bytes stay on the row; symbols are visited in ascending "
    "id; a class of size 0 or 1 does not draw from the generator; "
    "seed WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED"
)
RESIDUAL_WITHOUT_SYMBOL_RULE = (
    "drop the symbol byte and assign the residual payload to the nearest "
    "training payload under squared Euclidean distance on the dequantized "
    "codes; the train-only majority probe of exact distance ties is the "
    "prediction; the symbol is used only to form the residual at encode time"
)
RESIDUAL_WITHOUT_SYMBOL_CARRIES_RULE = (
    "reported, not an admission input: the payload-only arm carries the slot "
    "when its held-out accuracy is at least ACCURACY_MARGIN above symbol-only"
)
# The slot readout for a symbol-attached residual is the nearest training
# payload inside the assigned symbol. Exact distance ties take the majority
# probe, and a tie of probes takes the smaller probe id. An empty train
# symbol falls back to the global training majority. Probes are read only
# after that assignment. The full-vector probe readout is the nearest
# training row of the raw activation. That reference is not a compressed code.
DECODER_RULE = "within_symbol_nearest_dequantized_residual_then_majority"
FULL_VECTOR_READOUT = "nearest_training_row"
DISTANCE = "squared_euclidean"
FIDELITY_DISTANCE = "mean_per_row_squared_euclidean"
FIDELITY_NORMALIZED = "frobenius_norm_of_error_over_frobenius_norm_of_activation"
COST_UNIT = (
    "assignment_coordinate_ops and decode_coordinate_ops each count one "
    "coordinate touch in a squared-Euclidean comparison or in a quantize; "
    "a majority-map lookup is decode_row_ops and is not a coordinate op; "
    "capture time is not included"
)

# Parent boundary slot counts. Reproduction targets, not thresholds to lower.
PARENT_SLOT = {
    "condition_2": {
        "n": 64,
        "symbol": 38,
        "nearest_train_row": 59,
        "train_mean": 32,
        "shuffled_assignment": 38,
    },
    "condition_4": {
        "n": 16,
        "symbol": 12,
        "nearest_train_row": 14,
        "train_mean": 8,
        "shuffled_assignment": 8,
    },
}

ACTIVATION_SHA256 = {
    "c2_train_activations": "f9140300c4308ea67da83fae727b67ce93cd26fb55c396d9f57139fcb3ede999",
    "c2_heldout_activations": "33fe6c348437a28c6496fc52c5cb6e0a6d8dad3c67ab885dfa1253f2f607d9c4",
    "c4_train_activations": "dc17fbdffd8deb25cdb393239fba9fe26d6b27c0cf5280522a98406ffc629d64",
    "c4_heldout_activations": "fb23bebcfda1407a299c11ba48ebe04ebac68b319000f0b03e61a4ca3ce6f91c",
}
ACTIVATION_SHAPE = {
    "c2_train_activations": (128, HIDDEN_SIZE),
    "c2_heldout_activations": (64, HIDDEN_SIZE),
    "c4_train_activations": (32, HIDDEN_SIZE),
    "c4_heldout_activations": (16, HIDDEN_SIZE),
}
PROBE_KEYS = {
    "condition_2": ("c2_train_probe", "c2_heldout_probe"),
    "condition_4": ("c4_train_probe", "c4_heldout_probe"),
}

INDEPENDENT_TARGET_STATUS = "INDEPENDENT_TARGET_UNAVAILABLE"
INDEPENDENT_TARGET_REASON = (
    "no behavioral target that is not a function of the activation vector "
    "was constructed; logits, centroid cosine, and reconstruction error "
    "are not substitutes"
)

VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_RESEARCH_FAIL = "RESEARCH_FAIL"
VERDICT_GEOMETRY_ONLY = "GEOMETRY_ONLY"
VERDICT_RESIDUAL = "RESIDUAL_DISTINCTION"
RECOMMENDATION_NONE = "NONE"
VERDICTS = (
    VERDICT_UNRESOLVED,
    VERDICT_RESEARCH_FAIL,
    VERDICT_GEOMETRY_ONLY,
    VERDICT_RESIDUAL,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_NPZ = ROOT / "examples" / "activation_boundary" / "activations.npz"

_CONDITION_KEYS = {
    "condition_2": ("c2_train_activations", "c2_heldout_activations"),
    "condition_4": ("c4_train_activations", "c4_heldout_activations"),
}


def pack_int4(codes: np.ndarray) -> np.ndarray:
    """Pack signed codes into uint8 nibbles. Even dimensions occupy the low nibble."""
    width = int(codes.shape[1])
    if codes.ndim != 2 or width % 2 != 0:
        raise ValueError("int4 packing expects an even number of coordinates")
    if np.any(codes < -8) or np.any(codes > 7):
        raise ValueError("int4 code is outside the signed nibble")
    even = np.asarray(codes[:, 0::2], dtype=np.int16) & 0x0F
    odd = np.asarray(codes[:, 1::2], dtype=np.int16) & 0x0F
    return (even | (odd << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Inverse of ``pack_int4``."""
    low = (np.asarray(packed, dtype=np.uint8) & np.uint8(0x0F)).astype(np.int16)
    high = ((np.asarray(packed, dtype=np.uint8) >> np.uint8(4)) & np.uint8(0x0F)).astype(np.int16)
    low = np.where(low >= 8, low - 16, low)
    high = np.where(high >= 8, high - 16, high)
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    codes[:, 0::2] = low.astype(np.int8)
    codes[:, 1::2] = high.astype(np.int8)
    return codes


def pack_rows(symbols: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """One stored row: symbol byte, then the packed residual payload."""
    payload = pack_int4(codes)
    if payload.shape[1] != RESIDUAL_PAYLOAD_BYTES or codes.shape[1] != HIDDEN_SIZE:
        raise ValueError("residual payload does not match the frozen width")
    rows = np.empty((symbols.shape[0], BYTES_PER_ROW), dtype=np.uint8)
    rows[:, 0] = np.asarray(symbols, dtype=np.uint8)
    rows[:, 1:] = payload
    return rows


def fit_scale(train_residual: np.ndarray) -> np.ndarray:
    """Per-coordinate max absolute value of the training residual only."""
    if train_residual.ndim != 2:
        raise ValueError("scale is fit on a train residual matrix")
    magnitude = np.max(np.abs(np.asarray(train_residual, dtype=np.float64)), axis=0)
    return np.where(magnitude > 0.0, magnitude, 1.0).astype(np.float64)


def quantize(residual: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Round to the frozen signed grid. ``scale`` is an argument, not refit here."""
    unit = np.asarray(residual, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    codes = np.rint(unit * RESIDUAL_QMAX)
    return np.clip(codes, -RESIDUAL_QMAX, RESIDUAL_QMAX).astype(np.int8)


def dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    step = np.asarray(scale, dtype=np.float64) / RESIDUAL_QMAX
    return np.asarray(codes, dtype=np.float64) * step


def majority_probe(probes: np.ndarray) -> int:
    """Train-only majority. A tie uses the smaller probe id."""
    values, counts = np.unique(np.asarray(probes, dtype=np.int64), return_counts=True)
    if values.size == 0:
        raise ValueError("majority of an empty probe set")
    winners = values[counts == counts.max()]
    return int(np.min(winners))


def _condition_clause(accuracy_gain: bool, shuffle_fell: bool, storage_ok: bool) -> str:
    """Name the clause for one condition. Accuracy gain is required before the other misses."""
    if accuracy_gain and shuffle_fell and storage_ok:
        return "admission_gate_met"
    if not accuracy_gain:
        return "no_accuracy_gain_over_symbol_only"
    clauses: List[str] = []
    if not storage_ok:
        clauses.append("accuracy_gain_without_storage_advantage")
    if not shuffle_fell:
        clauses.append("accuracy_gain_survives_within_class_residual_shuffle")
    return "+".join(clauses)


def decide_verdict(
    *,
    integrity_failure: Optional[str],
    parent_reproduced: bool,
    conditions: Sequence[Dict[str, bool]],
) -> Tuple[str, str, str]:
    """Return verdict, recommendation, and the clause that fired.

    Recommendation is ``NONE`` on every branch. Authority is applied by the
    runner and is ``RUNTIME_NOT_AUTHORIZED``. ``conditions`` carries only
    the three admission booleans. It does not carry accuracies, fidelities,
    or byte counts. The runner computes those booleans from the frozen rules.
    """
    if integrity_failure:
        return VERDICT_UNRESOLVED, RECOMMENDATION_NONE, integrity_failure
    if not parent_reproduced:
        return (
            VERDICT_UNRESOLVED,
            RECOMMENDATION_NONE,
            "parent_symbol_boundary_not_reproduced",
        )
    admitted = False
    storage_miss = False
    shuffle_survives = False
    accuracy_gain = False
    for item in conditions:
        gain = bool(item["accuracy_gain"])
        fell = bool(item["shuffle_fell"])
        storage_ok = bool(item["storage_ok"])
        accuracy_gain = accuracy_gain or gain
        if gain and fell and storage_ok:
            admitted = True
        if gain and not storage_ok:
            storage_miss = True
        if gain and not fell:
            shuffle_survives = True
    if admitted:
        return (
            VERDICT_RESIDUAL,
            RECOMMENDATION_NONE,
            "admission_gate_met_without_independent_target",
        )
    if accuracy_gain and (storage_miss or shuffle_survives):
        clauses: List[str] = []
        if storage_miss:
            clauses.append("accuracy_gain_without_storage_advantage")
        if shuffle_survives:
            clauses.append("accuracy_gain_survives_within_class_residual_shuffle")
        return VERDICT_GEOMETRY_ONLY, RECOMMENDATION_NONE, "+".join(clauses)
    return (
        VERDICT_RESEARCH_FAIL,
        RECOMMENDATION_NONE,
        "no_accuracy_gain_over_symbol_only",
    )


def _accuracy(pred: np.ndarray, probes: np.ndarray) -> Dict[str, float]:
    n = int(probes.shape[0])
    n_correct = int(np.sum(np.asarray(pred, dtype=np.int64) == np.asarray(probes, dtype=np.int64)))
    return {
        "n_correct": n_correct,
        "n": n,
        "accuracy": n_correct / n,
    }


def _fit_map(symbols: np.ndarray, probes: np.ndarray) -> Tuple[Dict[int, int], int]:
    default = majority_probe(probes)
    mapping: Dict[int, int] = {}
    for symbol in np.unique(symbols):
        members = probes[symbols == symbol]
        mapping[int(symbol)] = majority_probe(members)
    return mapping, default


def _predict_map(
    mapping: Dict[int, int],
    default: int,
    symbols: np.ndarray,
) -> Tuple[np.ndarray, int]:
    missing = 0
    pred = np.empty(symbols.shape[0], dtype=np.int64)
    for index, symbol in enumerate(np.asarray(symbols, dtype=np.int64).tolist()):
        key = int(symbol)
        if key not in mapping:
            pred[index] = default
            missing += 1
        else:
            pred[index] = mapping[key]
    return pred, missing


def _symbol_accuracy(
    train_symbols: np.ndarray,
    held_symbols: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    mapping, default = _fit_map(train_symbols, train_y)
    pred, missing = _predict_map(mapping, default, held_symbols)
    block = _accuracy(pred, held_y)
    block["unseen_symbol_rows"] = missing
    return block, pred


def _nearest_train_rows(train_x: np.ndarray, held_x: np.ndarray) -> np.ndarray:
    """Smallest-index squared-Euclidean nearest training row."""
    dist = ((held_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist, axis=1).astype(np.int64)


def _fidelity(original: np.ndarray, hat: np.ndarray) -> dict:
    diff = np.asarray(original, dtype=np.float64) - np.asarray(hat, dtype=np.float64)
    per_row = np.sum(diff * diff, axis=1)
    denom = float(np.linalg.norm(original))
    if denom <= 1e-12:
        normalized = 0.0
    else:
        normalized = float(np.linalg.norm(diff) / denom)
    return {
        "mean_squared_euclidean": float(per_row.mean()),
        "normalized_frobenius": normalized,
        "n": int(original.shape[0]),
        "distance": FIDELITY_DISTANCE,
        "normalized": FIDELITY_NORMALIZED,
    }


def _identity_fidelity(original: np.ndarray) -> dict:
    return {
        "mean_squared_euclidean": 0.0,
        "normalized_frobenius": 0.0,
        "n": int(original.shape[0]),
        "distance": FIDELITY_DISTANCE,
        "normalized": FIDELITY_NORMALIZED,
    }


def _groups(symbols: np.ndarray) -> List[np.ndarray]:
    labels = np.asarray(symbols, dtype=np.int64)
    return [np.flatnonzero(labels == symbol) for symbol in range(N_SYMBOLS)]


def _decode_payloads(
    query: np.ndarray,
    library: np.ndarray,
    library_symbols: np.ndarray,
    library_probes: np.ndarray,
    query_symbols: Optional[np.ndarray],
    global_majority: int,
) -> Tuple[np.ndarray, int, int]:
    """Nearest library payload, then the majority probe of exact distance ties.

    ``query_symbols is None`` searches every training row. Otherwise the
    search stays inside the named symbol. An empty symbol uses
    ``global_majority``.
    """
    groups = _groups(library_symbols)
    pred = np.empty(query.shape[0], dtype=np.int64)
    empty = 0
    n_ties = 0
    for index in range(query.shape[0]):
        if query_symbols is None:
            members = np.arange(library.shape[0], dtype=np.int64)
        else:
            members = groups[int(query_symbols[index])]
        if members.size == 0:
            pred[index] = global_majority
            empty += 1
            continue
        dist = np.sum((library[members] - query[index]) ** 2, axis=1)
        best = float(dist.min())
        tied = members[dist == best]
        if tied.size > 1:
            n_ties += 1
        pred[index] = majority_probe(library_probes[tied])
    return pred, empty, n_ties


def _shuffle_payloads(symbols: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Permute residual bytes within a symbol. Symbol bytes stay put."""
    out = np.array(packed, copy=True)
    rng = np.random.default_rng(WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED)
    labels = np.asarray(symbols, dtype=np.int64)
    for symbol in range(N_SYMBOLS):
        idx = np.flatnonzero(labels == symbol)
        if idx.size <= 1:
            continue
        order = rng.permutation(idx.size)
        out[idx, 1:] = packed[idx[order], 1:]
    return out


def _occupancy(symbols: np.ndarray) -> List[int]:
    counts = np.bincount(np.asarray(symbols, dtype=np.int64), minlength=N_SYMBOLS)
    return [int(value) for value in counts]


def _probe_mix(symbols: np.ndarray, probes: np.ndarray) -> List[List[int]]:
    mix: List[List[int]] = []
    labels = np.asarray(symbols, dtype=np.int64)
    targets = np.asarray(probes, dtype=np.int64)
    for symbol in range(N_SYMBOLS):
        members = targets[labels == symbol]
        mix.append([int(np.sum(members == 0)), int(np.sum(members == 1))])
    return mix


def _arm(
    *,
    accuracy: dict,
    bytes_per_row: int,
    packing: str,
    role: str,
    assignment_coordinate_ops: int,
    decode_coordinate_ops: int,
    decode_row_ops: int,
    assignment_seconds: float,
    decode_seconds: float,
    fidelity: Optional[dict],
    fidelity_note: str,
    extra: Optional[dict] = None,
) -> dict:
    body = {
        "n_correct": accuracy["n_correct"],
        "n": accuracy["n"],
        "accuracy": accuracy["accuracy"],
        "bytes_per_row": bytes_per_row,
        "packing": packing,
        "role": role,
        "assignment_coordinate_ops": assignment_coordinate_ops,
        "decode_coordinate_ops": decode_coordinate_ops,
        "decode_row_ops": decode_row_ops,
        "assignment_seconds": assignment_seconds,
        "decode_seconds": decode_seconds,
        "seconds_are_a_gate": False,
        "fidelity": fidelity,
        "fidelity_note": fidelity_note,
        "admits_residual": False,
    }
    if extra:
        body.update(extra)
    return body


def _score_condition(
    name: str,
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
) -> dict:
    """Score one condition. Probes are not arguments of the fit or the quantizer."""
    n_train = int(train_x.shape[0])
    n_held = int(held_x.shape[0])
    dim = int(train_x.shape[1])
    if dim != HIDDEN_SIZE or held_x.shape[1] != HIDDEN_SIZE:
        raise ValueError("activation width is not the frozen hidden size")

    learn_start = time.perf_counter()
    train_symbols, centroids = kmeans(
        train_x,
        N_SYMBOLS,
        rng=np.random.default_rng(LEARN_SEED),
    )
    learn_seconds = time.perf_counter() - learn_start
    train_symbols = np.asarray(train_symbols, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float64)

    assign_start = time.perf_counter()
    held_symbols = assign_squared_euclidean(held_x, centroids)
    assign_seconds = time.perf_counter() - assign_start
    held_symbols = np.asarray(held_symbols, dtype=np.int64)

    symbol_block, symbol_pred = _symbol_accuracy(train_symbols, held_symbols, train_y, held_y)
    symbol_assignment_ops = n_held * N_SYMBOLS * dim
    symbol_hat = centroids[held_symbols]
    symbol_fidelity = _fidelity(held_x, symbol_hat)

    train_residual = train_x - centroids[train_symbols]
    held_residual = held_x - centroids[held_symbols]
    scale = fit_scale(train_residual)
    train_codes = quantize(train_residual, scale)
    held_codes = quantize(held_residual, scale)
    train_packed = pack_rows(train_symbols, train_codes)
    held_packed = pack_rows(held_symbols, held_codes)
    if train_packed.shape[1] != BYTES_PER_ROW or held_packed.shape[1] != BYTES_PER_ROW:
        raise RuntimeError("packer drifted from the frozen byte budget")
    train_unpacked = unpack_int4(train_packed[:, 1:])
    held_unpacked = unpack_int4(held_packed[:, 1:])
    if not np.array_equal(train_unpacked, train_codes) or not np.array_equal(held_unpacked, held_codes):
        raise RuntimeError("int4 pack did not round-trip")
    if not np.array_equal(held_packed[:, 0], held_symbols.astype(np.uint8)):
        raise RuntimeError("symbol byte does not match the assignment")

    train_q = dequantize(train_unpacked, scale)
    held_q = dequantize(held_unpacked, scale)
    global_majority = majority_probe(train_y)
    decode_start = time.perf_counter()
    residual_pred, empty_rows, n_ties = _decode_payloads(
        held_q,
        train_q,
        train_symbols,
        train_y,
        held_symbols,
        global_majority,
    )
    decode_seconds = time.perf_counter() - decode_start
    residual_block = _accuracy(residual_pred, held_y)
    disagreement = {
        "n_predictions_differ_from_symbol": int(np.sum(residual_pred != symbol_pred)),
        "n_only_residual_correct": int(np.sum((residual_pred == held_y) & (symbol_pred != held_y))),
        "n_only_symbol_correct": int(np.sum((residual_pred != held_y) & (symbol_pred == held_y))),
        "n_both_correct": int(np.sum((residual_pred == held_y) & (symbol_pred == held_y))),
        "n_both_wrong": int(np.sum((residual_pred != held_y) & (symbol_pred != held_y))),
    }
    search_ops = 0
    train_groups = _groups(train_symbols)
    for symbol in held_symbols.tolist():
        search_ops += int(train_groups[int(symbol)].shape[0]) * dim
    quantize_ops = n_held * dim
    residual_decode_ops = quantize_ops + search_ops
    residual_hat = centroids[held_symbols] + held_q
    residual_fidelity = _fidelity(held_x, residual_hat)

    nearest_start = time.perf_counter()
    nearest_index = _nearest_train_rows(train_x, held_x)
    nearest_seconds = time.perf_counter() - nearest_start
    nearest_pred = train_y[nearest_index]
    nearest_block = _accuracy(nearest_pred, held_y)
    nearest_ops = n_held * n_train * dim
    nearest_hat = train_x[nearest_index]
    nearest_fidelity = _fidelity(held_x, nearest_hat)

    mean_start = time.perf_counter()
    train_mean = train_x.mean(axis=0, keepdims=True)
    mean_train_symbols = assign_squared_euclidean(train_x, train_mean)
    mean_held_symbols = assign_squared_euclidean(held_x, train_mean)
    mean_seconds = time.perf_counter() - mean_start
    mean_block, _mean_pred = _symbol_accuracy(mean_train_symbols, mean_held_symbols, train_y, held_y)
    mean_hat = np.repeat(train_mean, n_held, axis=0)
    mean_fidelity = _fidelity(held_x, mean_hat)

    shuffle_start = time.perf_counter()
    shuffle_rng = np.random.default_rng(SHUFFLE_ASSIGNMENT_SEED)
    shuffled_symbols = held_symbols[shuffle_rng.permutation(n_held)]
    shuffle_seconds = time.perf_counter() - shuffle_start
    shuffled_block, _shuffled_pred = _symbol_accuracy(train_symbols, shuffled_symbols, train_y, held_y)
    shuffled_hat = centroids[shuffled_symbols]
    shuffled_fidelity = _fidelity(held_x, shuffled_hat)

    within_start = time.perf_counter()
    shuffled_packed = _shuffle_payloads(held_symbols, held_packed)
    shuffled_codes = unpack_int4(shuffled_packed[:, 1:])
    shuffled_q = dequantize(shuffled_codes, scale)
    within_pred, within_empty, within_ties = _decode_payloads(
        shuffled_q,
        train_q,
        train_symbols,
        train_y,
        held_symbols,
        global_majority,
    )
    within_seconds = time.perf_counter() - within_start
    within_block = _accuracy(within_pred, held_y)
    within_hat = centroids[held_symbols] + shuffled_q
    within_fidelity = _fidelity(held_x, within_hat)

    alone_start = time.perf_counter()
    alone_pred, alone_empty, alone_ties = _decode_payloads(
        held_q,
        train_q,
        train_symbols,
        train_y,
        None,
        global_majority,
    )
    alone_seconds = time.perf_counter() - alone_start
    alone_block = _accuracy(alone_pred, held_y)
    alone_decode_ops = quantize_ops + n_held * n_train * dim

    margin = residual_block["accuracy"] - symbol_block["accuracy"]
    fall = residual_block["accuracy"] - within_block["accuracy"]
    storage_ratio = BYTES_PER_ROW / FULL_VECTOR_BYTES
    accuracy_gain = residual_block["accuracy"] >= symbol_block["accuracy"] + ACCURACY_MARGIN
    shuffle_fell = fall >= SHUFFLE_FALL
    storage_ok = BYTES_PER_ROW <= STORAGE_RATIO_MAX * FULL_VECTOR_BYTES
    admitted = bool(accuracy_gain and shuffle_fell and storage_ok)
    carries = alone_block["accuracy"] >= symbol_block["accuracy"] + ACCURACY_MARGIN

    arms = {
        "symbol_only": _arm(
            accuracy=symbol_block,
            bytes_per_row=SYMBOL_ID_BYTES,
            packing=PACKING_SYMBOL,
            role="baseline",
            assignment_coordinate_ops=symbol_assignment_ops,
            decode_coordinate_ops=0,
            decode_row_ops=n_held,
            assignment_seconds=assign_seconds,
            decode_seconds=0.0,
            fidelity=symbol_fidelity,
            fidelity_note="reconstruction is the assigned train centroid",
            extra={"unseen_symbol_rows": symbol_block["unseen_symbol_rows"]},
        ),
        "symbol_plus_residual": _arm(
            accuracy=residual_block,
            bytes_per_row=BYTES_PER_ROW,
            packing=PACKING_SYMBOL_RESIDUAL,
            role="candidate",
            assignment_coordinate_ops=symbol_assignment_ops,
            decode_coordinate_ops=residual_decode_ops,
            decode_row_ops=n_held,
            assignment_seconds=assign_seconds,
            decode_seconds=decode_seconds,
            fidelity=residual_fidelity,
            fidelity_note="reconstruction is the assigned centroid plus the dequantized residual",
            extra={
                "empty_symbol_rows": empty_rows,
                "queries_with_distance_ties": n_ties,
                "storage_ratio_vs_full_vector": storage_ratio,
            },
        ),
        "nearest_train_row": _arm(
            accuracy=nearest_block,
            bytes_per_row=FULL_VECTOR_BYTES,
            packing=PACKING_NEAREST,
            role="reference",
            assignment_coordinate_ops=nearest_ops,
            decode_coordinate_ops=0,
            decode_row_ops=n_held,
            assignment_seconds=nearest_seconds,
            decode_seconds=0.0,
            fidelity=nearest_fidelity,
            fidelity_note="distance from the held-out row to the chosen training row; not a code",
        ),
        "full_vector": _arm(
            accuracy=nearest_block,
            bytes_per_row=FULL_VECTOR_BYTES,
            packing=PACKING_FULL_VECTOR,
            role="reference",
            assignment_coordinate_ops=nearest_ops,
            decode_coordinate_ops=0,
            decode_row_ops=n_held,
            assignment_seconds=nearest_seconds,
            decode_seconds=0.0,
            fidelity=_identity_fidelity(held_x),
            fidelity_note="the stored object is the activation, so the reconstruction error is zero",
            extra={"readout": FULL_VECTOR_READOUT},
        ),
        "train_mean": _arm(
            accuracy=mean_block,
            bytes_per_row=SYMBOL_ID_BYTES,
            packing=PACKING_SYMBOL,
            role="negative_control",
            assignment_coordinate_ops=n_held * dim,
            decode_coordinate_ops=0,
            decode_row_ops=n_held,
            assignment_seconds=mean_seconds,
            decode_seconds=0.0,
            fidelity=mean_fidelity,
            fidelity_note="reconstruction is the training-activation mean",
        ),
        "shuffled_assignment": _arm(
            accuracy=shuffled_block,
            bytes_per_row=SYMBOL_ID_BYTES,
            packing=PACKING_SYMBOL,
            role="negative_control",
            assignment_coordinate_ops=0,
            decode_coordinate_ops=0,
            decode_row_ops=n_held,
            assignment_seconds=shuffle_seconds,
            decode_seconds=0.0,
            fidelity=shuffled_fidelity,
            fidelity_note="reconstruction uses the centroid of the permuted symbol id",
            extra={"permutation_row_ops": n_held, "seed": SHUFFLE_ASSIGNMENT_SEED},
        ),
        "within_class_residual_shuffle": _arm(
            accuracy=within_block,
            bytes_per_row=BYTES_PER_ROW,
            packing=PACKING_SYMBOL_RESIDUAL,
            role="negative_control",
            assignment_coordinate_ops=symbol_assignment_ops,
            decode_coordinate_ops=residual_decode_ops,
            decode_row_ops=n_held,
            assignment_seconds=assign_seconds,
            decode_seconds=within_seconds,
            fidelity=within_fidelity,
            fidelity_note="reconstruction uses the row symbol and a residual payload taken from another row of that symbol",
            extra={
                "empty_symbol_rows": within_empty,
                "queries_with_distance_ties": within_ties,
                "seed": WITHIN_CLASS_RESIDUAL_SHUFFLE_SEED,
            },
        ),
        "residual_without_symbol": _arm(
            accuracy=alone_block,
            bytes_per_row=RESIDUAL_PAYLOAD_BYTES,
            packing=PACKING_PAYLOAD_ONLY,
            role="control",
            assignment_coordinate_ops=symbol_assignment_ops,
            decode_coordinate_ops=alone_decode_ops,
            decode_row_ops=n_held,
            assignment_seconds=assign_seconds,
            decode_seconds=alone_seconds,
            fidelity=None,
            fidelity_note=(
                "the activation is not reconstructed; the symbol byte that names "
                "the centroid is omitted. Payload quantization error equals the "
                "intact symbol-plus-residual activation error"
            ),
            extra={
                "empty_symbol_rows": alone_empty,
                "queries_with_distance_ties": alone_ties,
                "carries_slot": carries,
                "carries_rule": RESIDUAL_WITHOUT_SYMBOL_CARRIES_RULE,
                "payload_quantization_equals_intact_activation_fidelity": True,
            },
        ),
    }
    return {
        "condition": name,
        "n_train": n_train,
        "n_held": n_held,
        "learn_seconds": learn_seconds,
        "learn_seconds_are_a_gate": False,
        "min_cluster_size_quarantine": False,
        "train_symbol_counts": _occupancy(train_symbols),
        "held_symbol_counts": _occupancy(held_symbols),
        "train_probe_counts_per_symbol": _probe_mix(train_symbols, train_y),
        "heldout_on_empty_train_symbols": empty_rows,
        "prediction_disagreement_vs_symbol": disagreement,
        "quantization": {
            "train_code_min": int(train_codes.min()),
            "train_code_max": int(train_codes.max()),
            "held_code_min": int(held_codes.min()),
            "held_code_max": int(held_codes.max()),
            "held_zero_code_fraction": float(np.mean(held_codes == 0)),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "fit_on": "train_residual_only",
        },
        "arms": arms,
        "admission": {
            "accuracy_gain": accuracy_gain,
            "margin_over_symbol": margin,
            "required_margin": ACCURACY_MARGIN,
            "shuffle_fall": fall,
            "required_shuffle_fall": SHUFFLE_FALL,
            "shuffle_fell": shuffle_fell,
            "storage_ok": storage_ok,
            "bytes_per_row": BYTES_PER_ROW,
            "full_vector_bytes": FULL_VECTOR_BYTES,
            "storage_ratio": storage_ratio,
            "storage_ratio_max": STORAGE_RATIO_MAX,
            "admitted": admitted,
            "clause": _condition_clause(accuracy_gain, shuffle_fell, storage_ok),
            "residual_without_symbol_carries_slot": carries,
            "nearest_row_beats_symbol_by_margin": (
                nearest_block["accuracy"] >= symbol_block["accuracy"] + ACCURACY_MARGIN
            ),
            "references_do_not_admit": True,
        },
        "shared_bytes": {
            "centroids_float64": SHARED_CENTROID_BYTES,
            "scales_float64": SHARED_SCALE_BYTES,
            "included_in_per_row_gate": False,
        },
    }


def _probe_failure(name: str, probes: np.ndarray, n: int) -> Optional[str]:
    if probes.dtype != np.int32 or probes.shape != (n,):
        return f"probe_integrity:{name}"
    expected = np.tile(np.arange(2, dtype=np.int32), n // 2)
    if n % 2 != 0 or not np.array_equal(probes, expected):
        return f"probe_integrity:{name}"
    return None


def _assess(path: Path) -> dict:
    if not path.is_file():
        return {"ok": False, "reason": "fixture_missing", "arrays": None}
    imported = _import_failure()
    if imported:
        return {"ok": False, "reason": imported, "arrays": None}
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    for key, digest in ACTIVATION_SHA256.items():
        array = arrays.get(key)
        if array is None:
            return {"ok": False, "reason": "checksum_mismatch", "arrays": None}
        if array.dtype != np.float32 or tuple(array.shape) != ACTIVATION_SHAPE[key]:
            return {"ok": False, "reason": "checksum_mismatch", "arrays": None}
        if not np.isfinite(array).all():
            return {"ok": False, "reason": "checksum_mismatch", "arrays": None}
        if float32_sha256(array) != digest:
            return {"ok": False, "reason": "checksum_mismatch", "arrays": None}
    for condition, (train_key, held_key) in PROBE_KEYS.items():
        train_probe = arrays.get(train_key)
        held_probe = arrays.get(held_key)
        if train_probe is None or held_probe is None:
            return {"ok": False, "reason": f"probe_integrity:{condition}", "arrays": None}
        train_n = ACTIVATION_SHAPE[_CONDITION_KEYS[condition][0]][0]
        held_n = ACTIVATION_SHAPE[_CONDITION_KEYS[condition][1]][0]
        failure = _probe_failure(train_key, train_probe, train_n)
        if failure:
            return {"ok": False, "reason": failure, "arrays": None}
        failure = _probe_failure(held_key, held_probe, held_n)
        if failure:
            return {"ok": False, "reason": failure, "arrays": None}
    return {"ok": True, "reason": None, "arrays": arrays}


def _import_failure() -> Optional[str]:
    """Fail when another library module imports this research file.

    The check looks for an import of this module. A substring search is
    wrong here: the activation-source report uses the key
    ``shuffled_symbol_residual`` for a logit quantity, and that key is not
    an import.
    """
    package = ROOT / "src" / "neuralese"
    module = package / "symbol_residual.py"
    if not module.is_file():
        return "production_import:symbol_residual.py"
    needles = (
        "import neuralese.symbol_residual",
        "from neuralese.symbol_residual",
        "from neuralese import symbol_residual",
    )
    for path in sorted(package.glob("*.py")):
        if path.name == "symbol_residual.py":
            continue
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            return f"production_import:{path.name}"
    return None


def _parent_mismatch(scored: Dict[str, dict]) -> bool:
    for name, expected in PARENT_SLOT.items():
        arms = scored[name]["arms"]
        if arms["symbol_only"]["n"] != expected["n"]:
            return True
        if arms["symbol_only"]["n_correct"] != expected["symbol"]:
            return True
        if arms["nearest_train_row"]["n_correct"] != expected["nearest_train_row"]:
            return True
        if arms["full_vector"]["n_correct"] != expected["nearest_train_row"]:
            return True
        if arms["train_mean"]["n_correct"] != expected["train_mean"]:
            return True
        if arms["shuffled_assignment"]["n_correct"] != expected["shuffled_assignment"]:
            return True
    return False


def _base(genuine: bool) -> dict:
    return {
        "status": STATUS,
        "plan_status": PLAN_STATUS,
        "authority": RUNTIME_AUTHORITY,
        "parent_commit": PARENT_COMMIT,
        "parent_pr": PARENT_PR,
        "entering_verdict": ENTERING_VERDICT,
        "entering_recommendation": ENTERING_RECOMMENDATION,
        "genuine": genuine,
        "probe_used_in_fit": False,
        "gloss_used_as_evidence": False,
        "runtime_integration": False,
        "production_admission": False,
        "min_cluster_size_quarantine": False,
        "heldout_refit": False,
        "condition_1": "omitted_no_controlled_slot",
        "condition_3": "omitted_not_in_this_trial",
        "independent_target": {
            "status": INDEPENDENT_TARGET_STATUS,
            "reason": INDEPENDENT_TARGET_REASON,
            "satisfied": False,
        },
        "semantic_interpretation": False,
    }


def run_symbol_residual(npz_path: Path = FIXTURE_NPZ) -> dict:
    """Score the frozen capture. Held-out rows do not fit the scale or the centroids."""
    if INDEPENDENT_TARGET_STATUS != "INDEPENDENT_TARGET_UNAVAILABLE":
        raise RuntimeError("this trial has no independent behavioral target")
    if RESIDUAL_BITS != 4 or RESIDUAL_QMAX != 7:
        raise RuntimeError("residual precision drifted")
    if BYTES_PER_ROW != 65 or FULL_VECTOR_BYTES != 512:
        raise RuntimeError("byte budget drifted")
    if ACCURACY_MARGIN != 0.10 or SHUFFLE_FALL != 0.10 or STORAGE_RATIO_MAX != 0.25:
        raise RuntimeError("admission threshold drifted")
    assessment = _assess(npz_path)
    if not assessment["ok"]:
        verdict, recommendation, reason = decide_verdict(
            integrity_failure=assessment["reason"],
            parent_reproduced=False,
            conditions=(),
        )
        report = _base(False)
        report.update(
            {
                "verdict": verdict,
                "recommendation": recommendation,
                "reason": reason,
                "explicit_failures": [reason],
                "parent_reproduced": False,
            }
        )
        return report

    arrays = assessment["arrays"]
    scored: Dict[str, dict] = {}
    for condition, (train_key, held_key) in _CONDITION_KEYS.items():
        probe_train, probe_held = PROBE_KEYS[condition]
        train_x = np.asarray(arrays[train_key], dtype=np.float64)
        held_x = np.asarray(arrays[held_key], dtype=np.float64)
        train_y = np.asarray(arrays[probe_train], dtype=np.int64)
        held_y = np.asarray(arrays[probe_held], dtype=np.int64)
        scored[condition] = _score_condition(condition, train_x, held_x, train_y, held_y)

    parent_reproduced = not _parent_mismatch(scored)
    flags = []
    explicit: List[str] = []
    for name in ("condition_2", "condition_4"):
        admission = scored[name]["admission"]
        flags.append(
            {
                "name": name,
                "accuracy_gain": admission["accuracy_gain"],
                "shuffle_fell": admission["shuffle_fell"],
                "storage_ok": admission["storage_ok"],
            }
        )
        if not admission["admitted"]:
            if admission["accuracy_gain"] and not admission["storage_ok"]:
                explicit.append(f"{name}:accuracy_gain_without_storage_advantage")
            if admission["accuracy_gain"] and not admission["shuffle_fell"]:
                explicit.append(f"{name}:accuracy_gain_survives_within_class_residual_shuffle")
            if not admission["accuracy_gain"]:
                explicit.append(f"{name}:no_accuracy_gain_over_symbol_only")
    if not parent_reproduced:
        explicit.insert(0, "parent_symbol_boundary_not_reproduced")
    verdict, recommendation, reason = decide_verdict(
        integrity_failure=None,
        parent_reproduced=parent_reproduced,
        conditions=flags,
    )
    if verdict != VERDICT_RESIDUAL and reason not in explicit:
        explicit.insert(0, reason)
    report = _base(True)
    report.update(
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_sha256": WEIGHT_SHA256,
            "final_layer_index": FINAL_LAYER_INDEX,
            "controlled_property": CONTROLLED_PROPERTY,
            "residual_definition": RESIDUAL_DEFINITION,
            "residual_bits": RESIDUAL_BITS,
            "residual_qmax": RESIDUAL_QMAX,
            "scale_rule": SCALE_RULE,
            "decoder_rule": DECODER_RULE,
            "checksums": dict(ACTIVATION_SHA256),
            "parent_reproduced": parent_reproduced,
            "condition_2": scored["condition_2"],
            "condition_4": scored["condition_4"],
            "bytes_per_row": {
                "symbol_only": SYMBOL_ID_BYTES,
                "symbol_plus_residual": BYTES_PER_ROW,
                "nearest_train_row": FULL_VECTOR_BYTES,
                "full_vector": FULL_VECTOR_BYTES,
                "train_mean": SYMBOL_ID_BYTES,
                "shuffled_assignment": SYMBOL_ID_BYTES,
                "within_class_residual_shuffle": BYTES_PER_ROW,
                "residual_without_symbol": RESIDUAL_PAYLOAD_BYTES,
                "full_vector_float32": FULL_VECTOR_BYTES,
                "storage_ratio_max": STORAGE_RATIO_MAX,
                "symbol_plus_residual_ratio": BYTES_PER_ROW / FULL_VECTOR_BYTES,
            },
            "verdict": verdict,
            "recommendation": recommendation,
            "reason": reason,
            "explicit_failures": explicit,
        }
    )
    return report


def _format_report(report: dict) -> str:
    lines = [
        f"verdict {report['verdict']}",
        f"recommendation {report['recommendation']}",
        f"authority {report['authority']}",
        f"reason {report.get('reason')}",
        f"parent_reproduced {report.get('parent_reproduced')}",
        f"explicit_failures {report.get('explicit_failures')}",
    ]
    for name in ("condition_2", "condition_4"):
        block = report.get(name)
        if not isinstance(block, dict):
            continue
        lines.append(name)
        admission = block["admission"]
        lines.append(
            "  admission "
            f"gain={admission['accuracy_gain']} "
            f"margin={admission['margin_over_symbol']} "
            f"fall={admission['shuffle_fall']} "
            f"storage={admission['storage_ok']} "
            f"admitted={admission['admitted']} "
            f"alone_carries={admission['residual_without_symbol_carries_slot']}"
        )
        for arm_name, arm in block["arms"].items():
            fidelity = arm["fidelity"]
            if fidelity is None:
                fidelity_text = "None"
            else:
                fidelity_text = (
                    f"{fidelity['mean_squared_euclidean']} "
                    f"{fidelity['normalized_frobenius']}"
                )
            lines.append(
                f"  {arm_name} {arm['n_correct']}/{arm['n']} "
                f"bytes={arm['bytes_per_row']} "
                f"assign_ops={arm['assignment_coordinate_ops']} "
                f"decode_ops={arm['decode_coordinate_ops']} "
                f"fidelity={fidelity_text}"
            )
        lines.append(f"  train_counts {block['train_symbol_counts']}")
        lines.append(f"  held_counts {block['held_symbol_counts']}")
        lines.append(f"  probe_mix {block['train_probe_counts_per_symbol']}")
        lines.append(
            "  ties "
            f"residual={block['arms']['symbol_plus_residual']['queries_with_distance_ties']} "
            f"shuffle={block['arms']['within_class_residual_shuffle']['queries_with_distance_ties']} "
            f"alone={block['arms']['residual_without_symbol']['queries_with_distance_ties']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(_format_report(run_symbol_residual()))
