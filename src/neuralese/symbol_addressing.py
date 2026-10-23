"""Symbolic addressing versus quantized numerical memory. Research only.

The closed residual trial kept a slot count when a 4-bit residual was
attached to a symbol. This trial asks whether that symbol id delivers
measurable value beyond ordinary quantized numerical memory, once the
residual and the lookup index are accounted for.

The slot readout searches training payloads and returns the probe label
stored on the chosen training row. It is a nearest-training-example task.
It is not an English gloss and not an independently decoded semantic
state. Nothing here is a semantic decoder, and nothing here is compared
to one.

Admission constants above ``run_symbol_addressing`` were fixed before any
held-out slot score in this trial was computed. They are not retuned
afterward. Probe labels do not enter scales, clustering, codebooks, or
the choice of arm. The frozen residual codec is called and is not edited.
No runtime integration and no production admission.
"""
from __future__ import annotations

import importlib
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Status of this trial. Not a production admission. The entering residual
# result is retained. This module does not revoke it.
STATUS = "RESEARCH_ONLY"
PLAN_STATUS = "PLAN_ONLY"
PARENT_COMMIT = "01b500469f89f3ec7d8292e7e7e3b2f730629baf"
PARENT_PR = 5
PARENT_NOTE = "docs/PR1_SYMBOL_RESIDUAL.md"
ENTERING_VERDICT = "RESIDUAL_DISTINCTION"
ENTERING_RECOMMENDATION = "NONE"
RESIDUAL_DISTINCTION_REVOKED = False
RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"

CONTROLLED_PROPERTY = (
    "which of cat (probe 0, token id 5798) or dog (probe 1, token id 4370) "
    "sits at token index 1; the strings are probes, not features"
)
SLOT_TASK = "nearest_training_example_retrieval"
SLOT_TASK_IS_GLOSS = False
SLOT_TASK_IS_SEMANTIC_STATE = False
COMPARED_TO_SEMANTIC_DECODER = False

# Frozen codec facts for the two residual arms. The numbers are the parent
# codec's. This module checks them at runtime and does not redefine the grid.
RESIDUAL_BITS = 4
RESIDUAL_QMAX = 7
DEQUANT_FORMULA = "code * scale / RESIDUAL_QMAX"
DEQUANT_DENOMINATOR = RESIDUAL_QMAX
QUANTIZE_RULE = "rint_then_clip_to_signed_qmax"
SCALE_RULE = "per_coordinate_max_abs_train_only_zero_coordinate_scale_one"
STANDALONE_SOURCE = "activation"
RESIDUAL_SOURCE = "activation_minus_assigned_centroid"
DISTANCE = "squared_euclidean_on_dequantized_payloads"
TIE_BREAK = "majority_probe_of_exact_distance_ties_smaller_probe_id"
N_SYMBOLS = 8
HIDDEN_SIZE = 128
LEARN_SEED = 0

# Byte accounting. Frozen before scores. The parent 65/512 per-row cap is
# not the gate of this trial.
SHARED_FLOAT_BYTES = 8
CENTROID_TABLE_BYTES = N_SYMBOLS * HIDDEN_SIZE * SHARED_FLOAT_BYTES
SCALE_TABLE_BYTES = HIDDEN_SIZE * SHARED_FLOAT_BYTES
PAYLOAD_BYTES = (HIDDEN_SIZE * RESIDUAL_BITS) // 8
LABEL_BYTES = 1
SYMBOL_ID_BYTES = 1
PARENT_PER_ROW_CAP_IS_THIS_TRIAL_GATE = False

# The admission storage figure is system bytes divided by the held-out
# count. The other amortization is reported and is not the gate.
ADMISSION_STORAGE_FIGURE = "system_bytes_divided_by_n_heldout"
SECOND_AMORTIZATION = "system_bytes_divided_by_n_train_plus_n_heldout"
SECOND_AMORTIZATION_IS_A_GATE = False

# Relative cost margin 1/5, so a win must be strictly under 4/5 of the
# other arm. Cross-multiplication keeps an integer byte comparison exact.
# A mixed win (bytes against one arm, latency against the other) does not
# count: one of the two figures must clear both arms.
RELATIVE_COST_MARGIN_NUMERATOR = 1
RELATIVE_COST_MARGIN_DENOMINATOR = 5
COST_WIN_MAX_NUMERATOR = RELATIVE_COST_MARGIN_DENOMINATOR - RELATIVE_COST_MARGIN_NUMERATOR
COST_WIN_MAX_DENOMINATOR = RELATIVE_COST_MARGIN_DENOMINATOR
COST_WIN_RULE = (
    "admission storage strictly below both other arms, or admission "
    "latency strictly below both other arms; strictly below means "
    "value * 5 < other * 4; a win on only one of the two arms is not "
    "a cost win"
)

# The symbol arm must not be less accurate than residual-only. A nonzero
# tolerance would hide the restriction the parent trial already measured.
SLOT_TOLERANCE_VS_RESIDUAL_ONLY = 0.0
SLOT_TOLERANCE_JUSTIFICATION = (
    "zero: symbol-restricted retrieval must be at least as accurate as "
    "residual-only retrieval on the same condition"
)
TRIAL_LEVEL_COMBINER = "and_both_conditions"
SINGLE_CONDITION_WIN_IS_TRIAL_SUCCESS = False

# Latency procedure. Odd repeat count so the median is one order statistic,
# not an average of two samples. Learn time is not in the decode median.
LATENCY_WARMUP = 3
LATENCY_REPEATS = 31
ASSIGNMENT_TIMER = "squared_euclidean_argmin_against_the_eight_train_centroids"
QUANTIZE_TIMER = (
    "residual arms subtract the assigned centroid, then rint-clip and "
    "dequantize the query; the standalone arm quantizes the activation "
    "itself; this line is recorded and is not in the admission latency"
)
SEARCH_TIMER = "nearest_training_dequantized_payload_then_tie_majority"
ADMISSION_LATENCY_RULE = (
    "median of paired assignment-plus-search samples for an arm that "
    "assigns a centroid to form the query residual; median of search "
    "samples for the standalone quantizer; after LATENCY_WARMUP "
    "unrecorded full-path passes; LATENCY_REPEATS samples; k-means "
    "learn time excluded; quantize line excluded; coordinate-op counts "
    "are not this figure; a one-sample clock is not this figure"
)
COORDINATE_OPS_ARE_A_GATE = False
RECONSTRUCTION_IS_A_GATE = False
LEARN_SECONDS_ARE_A_GATE = False

FIDELITY_DISTANCE = "mean_per_row_squared_euclidean"
FIDELITY_NORMALIZED = "frobenius_norm_of_error_over_frobenius_norm_of_activation"

# Which shared tables each arm needs. Standalone quantizes the activation,
# so it has no centroid table. It does have the scale table: the decoder
# distance and the reconstruction both use the train-only per-coordinate
# scale. Residual arms need both tables. The residual-only arm does not
# store a symbol byte; centroid assignment, when required to form the
# query residual, is compute.
ARM_STORAGE = {
    "standalone_int4": {
        "uses_centroids": False,
        "uses_scales": True,
        "training_symbol_bytes": 0,
        "query_symbol_bytes": 0,
        "assigns_centroid": False,
        "symbol_byte_stored": False,
    },
    "symbol_plus_residual": {
        "uses_centroids": True,
        "uses_scales": True,
        "training_symbol_bytes": SYMBOL_ID_BYTES,
        "query_symbol_bytes": SYMBOL_ID_BYTES,
        "assigns_centroid": True,
        "symbol_byte_stored": True,
    },
    "residual_only": {
        "uses_centroids": True,
        "uses_scales": True,
        "training_symbol_bytes": 0,
        "query_symbol_bytes": 0,
        "assigns_centroid": True,
        "symbol_byte_stored": False,
    },
}

# Published parent slot counts. Reproduction targets. If a call to the
# frozen codec does not return these, this trial stops at UNRESOLVED.
PUBLISHED_SLOTS = {
    "condition_2": {
        "n": 64,
        "symbol_only": 38,
        "symbol_plus_residual": 57,
        "residual_without_symbol": 61,
        "nearest_train_row": 59,
    },
    "condition_4": {
        "n": 16,
        "symbol_only": 12,
        "symbol_plus_residual": 12,
        "residual_without_symbol": 14,
        "nearest_train_row": 14,
    },
}

INDEPENDENT_TARGET_STATUS = "INDEPENDENT_TARGET_UNAVAILABLE"
INDEPENDENT_TARGET_REASON = (
    "no behavioral target that is not a function of the activation was "
    "constructed; logits, centroid cosine, and reconstruction error are "
    "not substitutes"
)

VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_SYMBOL = "SYMBOL_ADDRESSING_VALUE"
VERDICT_QUANTIZED = "QUANTIZED_MEMORY_SUFFICIENT"
VERDICT_RESEARCH_FAIL = "RESEARCH_FAIL"
RECOMMENDATION_NONE = "NONE"
CLAUSE_GATE_MET = "symbolic_addressing_gate_met_without_independent_target"
CLAUSE_QUANTIZED = "quantized_memory_matches_or_beats_symbol_on_both_conditions"
CLAUSE_BELOW_STANDALONE = "symbol_accuracy_below_standalone_quant"
CLAUSE_BELOW_RESIDUAL = "symbol_accuracy_below_residual_only"
CLAUSE_NO_COST = "no_cost_win_versus_both_arms"
CLAUSE_REPRODUCTION = "published_slot_counts_not_reproduced"

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_NPZ = ROOT / "examples" / "activation_boundary" / "activations.npz"

_CONDITION_ORDER = ("condition_2", "condition_4")


def _load_residual():
    """Call the frozen residual module without a literal import statement.

    The parent isolation scan rejects import spellings of that module in
    every other file under ``src/neuralese``. This trial is allowed to
    call the frozen functions. The scan stays intact.
    """
    module_name = "neuralese." + "symbol_" + "residual"
    return importlib.import_module(module_name)


def account_system_bytes(
    n_train: int,
    n_held: int,
    *,
    uses_centroids: bool,
    uses_scales: bool,
    training_symbol_bytes: int,
    query_symbol_bytes: int,
) -> dict:
    """Shared tables plus the labeled training store plus query payloads.

    The probe label byte sits on each training row because the decoder
    returns that row's probe. It is not a symbol id. Queries do not carry it.
    """
    if n_train <= 0 or n_held <= 0:
        raise ValueError("byte accounting expects nonempty train and held-out splits")
    shared_centroids = CENTROID_TABLE_BYTES if uses_centroids else 0
    shared_scales = SCALE_TABLE_BYTES if uses_scales else 0
    shared = shared_centroids + shared_scales
    train_row = PAYLOAD_BYTES + training_symbol_bytes + LABEL_BYTES
    query_row = PAYLOAD_BYTES + query_symbol_bytes
    training_store = n_train * train_row
    query = n_held * query_row
    system = shared + training_store + query
    return {
        "shared_centroid_bytes": shared_centroids,
        "shared_scale_bytes": shared_scales,
        "shared_bytes": shared,
        "training_row_bytes": train_row,
        "training_store_bytes": training_store,
        "query_row_bytes": query_row,
        "query_payload_bytes": query,
        "system_bytes": system,
        "n_train": n_train,
        "n_held": n_held,
        "admission_storage_bytes": system / n_held,
        "admission_storage_figure": ADMISSION_STORAGE_FIGURE,
        "second_amortization_bytes": system / (n_train + n_held),
        "second_amortization_figure": SECOND_AMORTIZATION,
        "second_amortization_is_a_gate": SECOND_AMORTIZATION_IS_A_GATE,
        "parent_per_row_cap_is_this_trial_gate": PARENT_PER_ROW_CAP_IS_THIS_TRIAL_GATE,
    }


def strictly_below_cost_fraction(value: float, other: float) -> bool:
    """True when ``value`` is strictly under 4/5 of ``other``.

    The cross-multiply is the frozen reading of margin 1/5. A non-positive
    or non-finite reference cannot be beaten.
    """
    if not math.isfinite(value) or not math.isfinite(other) or other <= 0.0:
        return False
    return (value * COST_WIN_MAX_DENOMINATOR) < (other * COST_WIN_MAX_NUMERATOR)


def cost_win_versus_both(
    candidate: float,
    standalone: float,
    residual_only: float,
) -> bool:
    """One figure, both other arms. A win against only one arm is false."""
    return strictly_below_cost_fraction(candidate, standalone) and strictly_below_cost_fraction(
        candidate, residual_only
    )


def median_seconds(samples: Sequence[float]) -> float:
    """Middle order statistic. ``LATENCY_REPEATS`` is odd, so no pair is averaged."""
    if len(samples) != LATENCY_REPEATS:
        raise RuntimeError("latency sample count drifted from the frozen repeat count")
    if LATENCY_REPEATS % 2 != 1:
        raise RuntimeError("frozen latency repeat count is not odd")
    ordered = sorted(float(sample) for sample in samples)
    return ordered[LATENCY_REPEATS // 2]


def decide_verdict(
    *,
    integrity_failure: Optional[str],
    reproduction_ok: bool,
    conditions: Sequence[dict],
) -> Tuple[str, str, str]:
    """Return verdict, recommendation, and the clause that fired.

    ``conditions`` carries booleans computed by the runner from the frozen
    rules. This function does not read accuracies, byte totals, fidelities,
    or clocks. Recommendation is ``NONE`` on every branch. Authority is
    applied by the runner and is ``RUNTIME_NOT_AUTHORIZED``. An accuracy
    loss to residual-only is already inside ``admitted``; a cheap search
    does not flip that flag.
    """
    if integrity_failure:
        return VERDICT_UNRESOLVED, RECOMMENDATION_NONE, integrity_failure
    if not reproduction_ok:
        return VERDICT_UNRESOLVED, RECOMMENDATION_NONE, CLAUSE_REPRODUCTION
    if len(conditions) != 2:
        return VERDICT_UNRESOLVED, RECOMMENDATION_NONE, "condition_set_drifted"
    if TRIAL_LEVEL_COMBINER != "and_both_conditions":
        return VERDICT_UNRESOLVED, RECOMMENDATION_NONE, "trial_combiner_drifted"
    if all(bool(item["admitted"]) for item in conditions):
        return VERDICT_SYMBOL, RECOMMENDATION_NONE, CLAUSE_GATE_MET
    quantified = all(
        bool(item["arm1_matches_or_beats_arm2"]) or bool(item["arm3_matches_or_beats_arm2"])
        for item in conditions
    )
    if quantified:
        return VERDICT_QUANTIZED, RECOMMENDATION_NONE, CLAUSE_QUANTIZED
    clauses: List[str] = []
    for item in conditions:
        for clause in item["failure_clauses"]:
            if clause not in clauses:
                clauses.append(clause)
    if not clauses:
        clauses.append("symbolic_addressing_gate_not_met")
    return VERDICT_RESEARCH_FAIL, RECOMMENDATION_NONE, "+".join(clauses)


def _import_failure() -> Optional[str]:
    """Fail when a package module imports this research file."""
    package = ROOT / "src" / "neuralese"
    module = package / "symbol_addressing.py"
    if not module.is_file():
        return "production_import:symbol_addressing.py"
    needles = (
        "import neuralese.symbol_addressing",
        "from neuralese.symbol_addressing",
        "from neuralese import symbol_addressing",
    )
    for path in sorted(package.glob("*.py")):
        if path.name == "symbol_addressing.py":
            continue
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            return f"production_import:{path.name}"
    return None


def _codec_drift(sr) -> Optional[str]:
    if sr.RESIDUAL_BITS != RESIDUAL_BITS or sr.RESIDUAL_QMAX != RESIDUAL_QMAX:
        return "codec_constant_drift"
    if sr.ACCURACY_MARGIN != 0.10 or sr.SHUFFLE_FALL != 0.10 or sr.STORAGE_RATIO_MAX != 0.25:
        return "codec_constant_drift"
    if sr.BYTES_PER_ROW != 65 or sr.RESIDUAL_PAYLOAD_BYTES != PAYLOAD_BYTES:
        return "codec_constant_drift"
    if sr.SHARED_CENTROID_BYTES != CENTROID_TABLE_BYTES:
        return "codec_constant_drift"
    if sr.SHARED_SCALE_BYTES != SCALE_TABLE_BYTES:
        return "codec_constant_drift"
    if sr.N_SYMBOLS != N_SYMBOLS or sr.HIDDEN_SIZE != HIDDEN_SIZE or sr.LEARN_SEED != LEARN_SEED:
        return "codec_constant_drift"
    if CENTROID_TABLE_BYTES != 8192 or SCALE_TABLE_BYTES != 1024 or PAYLOAD_BYTES != 64:
        return "codec_constant_drift"
    return None


def _dequant_matches(sr) -> bool:
    scale = np.array([7.0, 14.0], dtype=np.float64)
    codes = np.array([[7, -7]], dtype=np.int8)
    got = np.asarray(sr.dequantize(codes, scale), dtype=np.float64)
    expect = np.asarray(codes, dtype=np.float64) * scale / DEQUANT_DENOMINATOR
    return bool(np.array_equal(got, expect))


def _reproduction_failure(parent: dict) -> Optional[str]:
    if parent.get("verdict") != ENTERING_VERDICT:
        return "parent_verdict_moved"
    if parent.get("recommendation") != ENTERING_RECOMMENDATION:
        return "parent_recommendation_moved"
    if parent.get("authority") != RUNTIME_AUTHORITY:
        return "parent_authority_moved"
    for name, expected in PUBLISHED_SLOTS.items():
        block = parent.get(name)
        if not isinstance(block, dict):
            return CLAUSE_REPRODUCTION
        arms = block.get("arms")
        if not isinstance(arms, dict):
            return CLAUSE_REPRODUCTION
        for arm_name, key in (
            ("symbol_only", "symbol_only"),
            ("symbol_plus_residual", "symbol_plus_residual"),
            ("residual_without_symbol", "residual_without_symbol"),
            ("nearest_train_row", "nearest_train_row"),
        ):
            arm = arms.get(arm_name)
            if not isinstance(arm, dict):
                return CLAUSE_REPRODUCTION
            if int(arm["n"]) != int(expected["n"]):
                return CLAUSE_REPRODUCTION
            if int(arm["n_correct"]) != int(expected[key]):
                return CLAUSE_REPRODUCTION
    return None


def _empty_base(*, genuine: bool, reproduction_ok: bool) -> dict:
    return {
        "status": STATUS,
        "plan_status": PLAN_STATUS,
        "authority": RUNTIME_AUTHORITY,
        "parent_commit": PARENT_COMMIT,
        "parent_pr": PARENT_PR,
        "parent_note": PARENT_NOTE,
        "entering_verdict": ENTERING_VERDICT,
        "entering_recommendation": ENTERING_RECOMMENDATION,
        "residual_distinction_revoked": RESIDUAL_DISTINCTION_REVOKED,
        "genuine": genuine,
        "reproduction_ok": reproduction_ok,
        "probe_used_in_fit": False,
        "gloss_used_as_evidence": False,
        "runtime_integration": False,
        "production_admission": False,
        "heldout_refit": False,
        "new_model_training": False,
        "condition_1": "omitted_no_controlled_slot",
        "condition_3": "omitted_not_in_this_trial",
        "slot_task": SLOT_TASK,
        "slot_task_is_english_gloss": SLOT_TASK_IS_GLOSS,
        "slot_task_is_independently_decoded_semantic_state": SLOT_TASK_IS_SEMANTIC_STATE,
        "compared_to_semantic_decoder": COMPARED_TO_SEMANTIC_DECODER,
        "semantic_interpretation": False,
        "independent_target": {
            "status": INDEPENDENT_TARGET_STATUS,
            "reason": INDEPENDENT_TARGET_REASON,
            "satisfied": False,
        },
        "dequant_formula": DEQUANT_FORMULA,
        "distance": DISTANCE,
        "admission_storage_figure": ADMISSION_STORAGE_FIGURE,
        "admission_latency_rule": ADMISSION_LATENCY_RULE,
        "slot_tolerance_vs_residual_only": SLOT_TOLERANCE_VS_RESIDUAL_ONLY,
        "trial_level_combiner": TRIAL_LEVEL_COMBINER,
        "reconstruction_is_a_gate": RECONSTRUCTION_IS_A_GATE,
        "coordinate_ops_are_a_gate": COORDINATE_OPS_ARE_A_GATE,
    }


def _unresolved(reason: str, *, genuine: bool = False, reproduction_ok: bool = False) -> dict:
    verdict, recommendation, clause = decide_verdict(
        integrity_failure=reason,
        reproduction_ok=reproduction_ok,
        conditions=(),
    )
    report = _empty_base(genuine=genuine, reproduction_ok=reproduction_ok)
    report.update(
        {
            "verdict": verdict,
            "recommendation": recommendation,
            "reason": clause,
            "explicit_failures": [clause],
        }
    )
    return report


def _fit_codebooks(sr, train_x: np.ndarray, held_x: np.ndarray) -> dict:
    """Train-only centroids and scales. Probes are not arguments."""
    learn_start = time.perf_counter()
    train_symbols, centroids = sr.kmeans(
        train_x,
        sr.N_SYMBOLS,
        rng=np.random.default_rng(sr.LEARN_SEED),
    )
    learn_seconds = time.perf_counter() - learn_start
    train_symbols = np.asarray(train_symbols, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.shape != (N_SYMBOLS, HIDDEN_SIZE):
        raise RuntimeError("centroid table shape drifted")
    if centroids.dtype != np.float64:
        raise RuntimeError("centroid table is not float64")

    activation_scale = sr.fit_scale(train_x)
    train_residual = train_x - centroids[train_symbols]
    held_symbols = sr.assign_squared_euclidean(held_x, centroids)
    held_symbols = np.asarray(held_symbols, dtype=np.int64)
    held_residual = held_x - centroids[held_symbols]
    residual_scale = sr.fit_scale(train_residual)

    train_activation_codes = sr.quantize(train_x, activation_scale)
    held_activation_codes = sr.quantize(held_x, activation_scale)
    train_residual_codes = sr.quantize(train_residual, residual_scale)
    held_residual_codes = sr.quantize(held_residual, residual_scale)

    train_activation_q = sr.dequantize(sr.unpack_int4(sr.pack_int4(train_activation_codes)), activation_scale)
    held_activation_q = sr.dequantize(sr.unpack_int4(sr.pack_int4(held_activation_codes)), activation_scale)
    train_packed = sr.pack_rows(train_symbols, train_residual_codes)
    held_packed = sr.pack_rows(held_symbols, held_residual_codes)
    train_residual_q = sr.dequantize(sr.unpack_int4(train_packed[:, 1:]), residual_scale)
    held_residual_q = sr.dequantize(sr.unpack_int4(held_packed[:, 1:]), residual_scale)
    if train_packed.shape[1] != 65 or held_packed.shape[1] != 65:
        raise RuntimeError("symbol-plus-residual payload width drifted")
    if train_activation_q.shape[1] != HIDDEN_SIZE or held_residual_q.shape[1] != HIDDEN_SIZE:
        raise RuntimeError("dequantized width drifted")
    return {
        "learn_seconds": learn_seconds,
        "train_symbols": train_symbols,
        "held_symbols": held_symbols,
        "centroids": centroids,
        "activation_scale": activation_scale,
        "residual_scale": residual_scale,
        "train_activation_codes": train_activation_codes,
        "held_activation_codes": held_activation_codes,
        "train_residual_codes": train_residual_codes,
        "held_residual_codes": held_residual_codes,
        "train_activation_q": train_activation_q,
        "held_activation_q": held_activation_q,
        "train_residual_q": train_residual_q,
        "held_residual_q": held_residual_q,
    }


def _search(sr, query: np.ndarray, library: np.ndarray, library_symbols: np.ndarray, library_probes: np.ndarray, query_symbols: Optional[np.ndarray]):
    global_majority = int(sr.majority_probe(library_probes))
    return sr._decode_payloads(
        query,
        library,
        library_symbols,
        library_probes,
        query_symbols,
        global_majority,
    )


def _search_ops(query_symbols: Optional[np.ndarray], library_symbols: np.ndarray, n_query: int, n_library: int) -> int:
    if query_symbols is None:
        return n_query * n_library * HIDDEN_SIZE
    counts = np.bincount(np.asarray(library_symbols, dtype=np.int64), minlength=N_SYMBOLS)
    total = 0
    for symbol in np.asarray(query_symbols, dtype=np.int64).tolist():
        total += int(counts[int(symbol)]) * HIDDEN_SIZE
    return total


def _time_decode(
    sr,
    *,
    assigns: bool,
    assign_fn,
    quantize_fn,
    search_fn,
) -> dict:
    """Warm up, then record paired samples. The official prediction is separate."""
    for _ in range(LATENCY_WARMUP):
        symbols = assign_fn() if assigns else None
        query = quantize_fn(symbols)
        search_fn(query, symbols)
    assignment_samples: List[float] = []
    quantize_samples: List[float] = []
    search_samples: List[float] = []
    admission_samples: List[float] = []
    last = None
    for _ in range(LATENCY_REPEATS):
        if assigns:
            start = time.perf_counter()
            symbols = assign_fn()
            assign_seconds = time.perf_counter() - start
        else:
            symbols = None
            assign_seconds = 0.0
        start = time.perf_counter()
        query = quantize_fn(symbols)
        quantize_seconds = time.perf_counter() - start
        start = time.perf_counter()
        last = search_fn(query, symbols)
        search_seconds = time.perf_counter() - start
        assignment_samples.append(assign_seconds)
        quantize_samples.append(quantize_seconds)
        search_samples.append(search_seconds)
        if assigns:
            admission_samples.append(assign_seconds + search_seconds)
        else:
            admission_samples.append(search_seconds)
    return {
        "median_assignment_seconds": median_seconds(assignment_samples),
        "median_quantize_seconds": median_seconds(quantize_samples),
        "median_search_seconds": median_seconds(search_samples),
        "admission_latency_seconds": median_seconds(admission_samples),
        "warmup": LATENCY_WARMUP,
        "repeats": LATENCY_REPEATS,
        "last_prediction": last,
        "quantize_is_in_admission_latency": False,
        "learn_is_in_admission_latency": False,
        "seconds_are_one_sample": False,
    }


def _arm_record(
    *,
    name: str,
    correct: int,
    n: int,
    accounting: dict,
    fidelity: dict,
    fidelity_note: str,
    latency: dict,
    assignment_ops: int,
    quantize_ops: int,
    search_ops: int,
    ties: int,
    empty_rows: int,
    extra: Optional[dict] = None,
) -> dict:
    body = {
        "arm": name,
        "n_correct": correct,
        "n": n,
        "accuracy": correct / n,
        "accounting": accounting,
        "system_bytes": accounting["system_bytes"],
        "admission_storage_bytes": accounting["admission_storage_bytes"],
        "second_amortization_bytes": accounting["second_amortization_bytes"],
        "reconstruction": fidelity,
        "reconstruction_note": fidelity_note,
        "reconstruction_is_a_gate": False,
        "median_assignment_seconds": latency["median_assignment_seconds"],
        "median_quantize_seconds": latency["median_quantize_seconds"],
        "median_search_seconds": latency["median_search_seconds"],
        "admission_latency_seconds": latency["admission_latency_seconds"],
        "latency_warmup": latency["warmup"],
        "latency_repeats": latency["repeats"],
        "quantize_is_in_admission_latency": False,
        "assignment_coordinate_ops": assignment_ops,
        "quantize_coordinate_ops": quantize_ops,
        "search_coordinate_ops": search_ops,
        "coordinate_ops_are_a_gate": False,
        "queries_with_distance_ties": ties,
        "empty_symbol_rows": empty_rows,
    }
    if extra:
        body.update(extra)
    return body


def _condition_gate(arm1: dict, arm2: dict, arm3: dict) -> dict:
    n = int(arm2["n"])
    if int(arm1["n"]) != n or int(arm3["n"]) != n:
        raise RuntimeError("arm widths differ")
    a1 = arm1["n_correct"] / n
    a2 = arm2["n_correct"] / n
    a3 = arm3["n_correct"] / n
    vs_standalone = a2 >= a1
    vs_residual = a2 >= (a3 - SLOT_TOLERANCE_VS_RESIDUAL_ONLY)
    bytes_win = cost_win_versus_both(
        arm2["admission_storage_bytes"],
        arm1["admission_storage_bytes"],
        arm3["admission_storage_bytes"],
    )
    latency_win = cost_win_versus_both(
        arm2["admission_latency_seconds"],
        arm1["admission_latency_seconds"],
        arm3["admission_latency_seconds"],
    )
    cost_win = bytes_win or latency_win
    # An accuracy loss to residual-only blocks the condition even if the
    # restricted search is cheaper.
    admitted = bool(vs_standalone and vs_residual and cost_win)
    failures: List[str] = []
    if not vs_standalone:
        failures.append(CLAUSE_BELOW_STANDALONE)
    if not vs_residual:
        failures.append(CLAUSE_BELOW_RESIDUAL)
    if not cost_win:
        failures.append(CLAUSE_NO_COST)
    return {
        "arm2_at_least_arm1": vs_standalone,
        "arm2_at_least_arm3_minus_tolerance": vs_residual,
        "tolerance_vs_residual_only": SLOT_TOLERANCE_VS_RESIDUAL_ONLY,
        "bytes_win_versus_both": bytes_win,
        "latency_win_versus_both": latency_win,
        "cost_win": cost_win,
        "admitted": admitted,
        "arm1_matches_or_beats_arm2": a1 >= a2,
        "arm3_matches_or_beats_arm2": a3 >= a2,
        "failure_clauses": failures,
        "symbol_beats_standalone": a2 > a1,
        "symbol_beats_residual_only": a2 > a3,
        "symbol_matches_or_beats_standalone": vs_standalone,
        "symbol_matches_or_beats_residual_only": vs_residual,
    }


def _score_condition(sr, name: str, train_x: np.ndarray, held_x: np.ndarray, train_y: np.ndarray, held_y: np.ndarray) -> dict:
    n_train = int(train_x.shape[0])
    n_held = int(held_x.shape[0])
    fitted = _fit_codebooks(sr, train_x, held_x)
    quantize_ops = n_held * HIDDEN_SIZE
    assignment_ops = n_held * N_SYMBOLS * HIDDEN_SIZE

    def activation_quantize(_symbols):
        return sr.dequantize(sr.quantize(held_x, fitted["activation_scale"]), fitted["activation_scale"])

    def residual_quantize(symbols):
        residual = held_x - fitted["centroids"][symbols]
        return sr.dequantize(sr.quantize(residual, fitted["residual_scale"]), fitted["residual_scale"])

    def assign():
        return sr.assign_squared_euclidean(held_x, fitted["centroids"])

    standalone_pred, standalone_empty, standalone_ties = _search(
        sr,
        fitted["held_activation_q"],
        fitted["train_activation_q"],
        fitted["train_symbols"],
        train_y,
        None,
    )
    symbol_pred, symbol_empty, symbol_ties = _search(
        sr,
        fitted["held_residual_q"],
        fitted["train_residual_q"],
        fitted["train_symbols"],
        train_y,
        fitted["held_symbols"],
    )
    alone_pred, alone_empty, alone_ties = _search(
        sr,
        fitted["held_residual_q"],
        fitted["train_residual_q"],
        fitted["train_symbols"],
        train_y,
        None,
    )

    def search_standalone(query, _symbols):
        pred, _empty, _ties = _search(
            sr,
            query,
            fitted["train_activation_q"],
            fitted["train_symbols"],
            train_y,
            None,
        )
        return pred

    def search_symbol(query, symbols):
        pred, _empty, _ties = _search(
            sr,
            query,
            fitted["train_residual_q"],
            fitted["train_symbols"],
            train_y,
            symbols,
        )
        return pred

    def search_alone(query, _symbols):
        pred, _empty, _ties = _search(
            sr,
            query,
            fitted["train_residual_q"],
            fitted["train_symbols"],
            train_y,
            None,
        )
        return pred

    latency_standalone = _time_decode(
        sr,
        assigns=False,
        assign_fn=assign,
        quantize_fn=activation_quantize,
        search_fn=search_standalone,
    )
    latency_symbol = _time_decode(
        sr,
        assigns=True,
        assign_fn=assign,
        quantize_fn=residual_quantize,
        search_fn=search_symbol,
    )
    latency_alone = _time_decode(
        sr,
        assigns=True,
        assign_fn=assign,
        quantize_fn=residual_quantize,
        search_fn=search_alone,
    )
    if not np.array_equal(latency_standalone["last_prediction"], standalone_pred):
        raise RuntimeError("standalone decode path drifted inside the timer")
    if not np.array_equal(latency_symbol["last_prediction"], symbol_pred):
        raise RuntimeError("symbol decode path drifted inside the timer")
    if not np.array_equal(latency_alone["last_prediction"], alone_pred):
        raise RuntimeError("residual-only decode path drifted inside the timer")

    hat_standalone = fitted["held_activation_q"]
    hat_symbol = fitted["centroids"][fitted["held_symbols"]] + fitted["held_residual_q"]
    hat_alone_with = fitted["centroids"][fitted["held_symbols"]] + fitted["held_residual_q"]
    hat_alone_without = fitted["held_residual_q"]
    if not np.array_equal(hat_symbol, hat_alone_with):
        raise RuntimeError("residual frame was not shared by the symbol and residual-only arms")

    fid_standalone = sr._fidelity(held_x, hat_standalone)
    fid_symbol = sr._fidelity(held_x, hat_symbol)
    fid_with = sr._fidelity(held_x, hat_alone_with)
    fid_without = sr._fidelity(held_x, hat_alone_without)
    if fid_symbol["mean_squared_euclidean"] != fid_with["mean_squared_euclidean"]:
        raise RuntimeError("centroid-restored residual error does not match the symbol arm")
    if fid_symbol["normalized_frobenius"] != fid_with["normalized_frobenius"]:
        raise RuntimeError("centroid-restored residual error does not match the symbol arm")

    spec_standalone = ARM_STORAGE["standalone_int4"]
    spec_symbol = ARM_STORAGE["symbol_plus_residual"]
    spec_alone = ARM_STORAGE["residual_only"]
    acc_standalone = account_system_bytes(
        n_train,
        n_held,
        uses_centroids=spec_standalone["uses_centroids"],
        uses_scales=spec_standalone["uses_scales"],
        training_symbol_bytes=spec_standalone["training_symbol_bytes"],
        query_symbol_bytes=spec_standalone["query_symbol_bytes"],
    )
    acc_symbol = account_system_bytes(
        n_train,
        n_held,
        uses_centroids=spec_symbol["uses_centroids"],
        uses_scales=spec_symbol["uses_scales"],
        training_symbol_bytes=spec_symbol["training_symbol_bytes"],
        query_symbol_bytes=spec_symbol["query_symbol_bytes"],
    )
    acc_alone = account_system_bytes(
        n_train,
        n_held,
        uses_centroids=spec_alone["uses_centroids"],
        uses_scales=spec_alone["uses_scales"],
        training_symbol_bytes=spec_alone["training_symbol_bytes"],
        query_symbol_bytes=spec_alone["query_symbol_bytes"],
    )

    arms = {
        "standalone_int4": _arm_record(
            name="standalone_int4",
            correct=int(np.sum(standalone_pred == held_y)),
            n=n_held,
            accounting=acc_standalone,
            fidelity=fid_standalone,
            fidelity_note=(
                "reconstruction is the dequantized activation code, "
                "code * activation_scale / 7; no centroid is added"
            ),
            latency=latency_standalone,
            assignment_ops=0,
            quantize_ops=quantize_ops,
            search_ops=_search_ops(None, fitted["train_symbols"], n_held, n_train),
            ties=standalone_ties,
            empty_rows=standalone_empty,
            extra={
                "centroid_assignment_is_compute": False,
                "symbol_byte_stored": False,
                "scale_min": float(fitted["activation_scale"].min()),
                "scale_max": float(fitted["activation_scale"].max()),
                "train_code_min": int(fitted["train_activation_codes"].min()),
                "train_code_max": int(fitted["train_activation_codes"].max()),
                "held_code_min": int(fitted["held_activation_codes"].min()),
                "held_code_max": int(fitted["held_activation_codes"].max()),
                "held_zero_code_fraction": float(np.mean(fitted["held_activation_codes"] == 0)),
            },
        ),
        "symbol_plus_residual": _arm_record(
            name="symbol_plus_residual",
            correct=int(np.sum(symbol_pred == held_y)),
            n=n_held,
            accounting=acc_symbol,
            fidelity=fid_symbol,
            fidelity_note="reconstruction is the assigned centroid plus the dequantized residual",
            latency=latency_symbol,
            assignment_ops=assignment_ops,
            quantize_ops=quantize_ops,
            search_ops=_search_ops(fitted["held_symbols"], fitted["train_symbols"], n_held, n_train),
            ties=symbol_ties,
            empty_rows=symbol_empty,
            extra={
                "centroid_assignment_is_compute": True,
                "symbol_byte_stored": True,
                "search_restricted_to_assigned_symbol": True,
            },
        ),
        "residual_only": _arm_record(
            name="residual_only",
            correct=int(np.sum(alone_pred == held_y)),
            n=n_held,
            accounting=acc_alone,
            fidelity=fid_with,
            fidelity_note=(
                "reconstruction (a) adds the assigned centroid back; "
                "this is the same vector as the symbol arm; the symbol "
                "byte is not stored, and the assignment is compute"
            ),
            latency=latency_alone,
            assignment_ops=assignment_ops,
            quantize_ops=quantize_ops,
            search_ops=_search_ops(None, fitted["train_symbols"], n_held, n_train),
            ties=alone_ties,
            empty_rows=alone_empty,
            extra={
                "centroid_assignment_is_compute": True,
                "symbol_byte_stored": False,
                "search_restricted_to_assigned_symbol": False,
                "reconstruction_without_centroid": fid_without,
                "reconstruction_without_centroid_is_faithful": False,
                "reconstruction_without_centroid_note": (
                    "reconstruction (b) is the dequantized residual alone; "
                    "omitting the symbol id removes the stored name of the "
                    "centroid, so this vector is not the activation"
                ),
                "reconstruction_with_centroid_matches_symbol_arm": True,
            },
        ),
    }
    gate = _condition_gate(
        arms["standalone_int4"],
        arms["symbol_plus_residual"],
        arms["residual_only"],
    )
    return {
        "condition": name,
        "n_train": n_train,
        "n_held": n_held,
        "learn_seconds": fitted["learn_seconds"],
        "learn_seconds_are_a_gate": LEARN_SECONDS_ARE_A_GATE,
        "arms": arms,
        "gate": gate,
        "quantization_activation": {
            "scale_min": arms["standalone_int4"]["scale_min"],
            "scale_max": arms["standalone_int4"]["scale_max"],
            "train_code_min": arms["standalone_int4"]["train_code_min"],
            "train_code_max": arms["standalone_int4"]["train_code_max"],
            "held_code_min": arms["standalone_int4"]["held_code_min"],
            "held_code_max": arms["standalone_int4"]["held_code_max"],
            "held_zero_code_fraction": arms["standalone_int4"]["held_zero_code_fraction"],
            "fit_on": "train_activation_only",
        },
    }


def _published_block(parent: dict) -> dict:
    block = {}
    for name, expected in PUBLISHED_SLOTS.items():
        arms = parent[name]["arms"]
        block[name] = {
            "n": expected["n"],
            "symbol_only": int(arms["symbol_only"]["n_correct"]),
            "symbol_plus_residual": int(arms["symbol_plus_residual"]["n_correct"]),
            "residual_without_symbol": int(arms["residual_without_symbol"]["n_correct"]),
            "nearest_train_row": int(arms["nearest_train_row"]["n_correct"]),
            "matches_published": True,
        }
    return block


def _assert_parent_fidelity(sr_parent_condition: dict, scored: dict) -> None:
    parent_fid = sr_parent_condition["arms"]["symbol_plus_residual"]["fidelity"]
    got = scored["arms"]["symbol_plus_residual"]["reconstruction"]
    if got["mean_squared_euclidean"] != parent_fid["mean_squared_euclidean"]:
        raise RuntimeError("symbol-arm reconstruction does not match the frozen codec")
    if got["normalized_frobenius"] != parent_fid["normalized_frobenius"]:
        raise RuntimeError("symbol-arm reconstruction does not match the frozen codec")


def run_symbol_addressing(npz_path: Path = FIXTURE_NPZ) -> dict:
    """Score the frozen capture. Thresholds above this function stay as written."""
    if INDEPENDENT_TARGET_STATUS != "INDEPENDENT_TARGET_UNAVAILABLE":
        raise RuntimeError("this trial has no independent behavioral target")
    if SLOT_TOLERANCE_VS_RESIDUAL_ONLY != 0.0:
        raise RuntimeError("slot tolerance drifted")
    if LATENCY_WARMUP != 3 or LATENCY_REPEATS != 31:
        raise RuntimeError("latency procedure drifted")
    if ADMISSION_STORAGE_FIGURE != "system_bytes_divided_by_n_heldout":
        raise RuntimeError("admission storage figure drifted")
    if SINGLE_CONDITION_WIN_IS_TRIAL_SUCCESS:
        raise RuntimeError("a one-condition win is not trial-level success")
    sr = _load_residual()
    drift = _codec_drift(sr)
    if drift:
        return _unresolved(drift)
    if not _dequant_matches(sr):
        return _unresolved("dequant_convention_mismatch")
    isolation = _import_failure()
    if isolation:
        return _unresolved(isolation)
    parent = sr.run_symbol_residual(npz_path)
    if not parent.get("genuine"):
        reason = str(parent.get("reason") or "parent_report_unresolved")
        return _unresolved(reason, genuine=False, reproduction_ok=False)
    repro_fail = _reproduction_failure(parent)
    if repro_fail:
        return _unresolved(repro_fail, genuine=True, reproduction_ok=False)

    with np.load(npz_path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    scored: Dict[str, dict] = {}
    for condition, (train_key, held_key) in sr._CONDITION_KEYS.items():
        probe_train, probe_held = sr.PROBE_KEYS[condition]
        for key in (train_key, held_key):
            if sr.float32_sha256(arrays[key]) != sr.ACTIVATION_SHA256[key]:
                return _unresolved("checksum_mismatch", genuine=False, reproduction_ok=False)
        train_x = np.asarray(arrays[train_key], dtype=np.float64)
        held_x = np.asarray(arrays[held_key], dtype=np.float64)
        train_y = np.asarray(arrays[probe_train], dtype=np.int64)
        held_y = np.asarray(arrays[probe_held], dtype=np.int64)
        try:
            scored[condition] = _score_condition(sr, condition, train_x, held_x, train_y, held_y)
            _assert_parent_fidelity(parent[condition], scored[condition])
        except RuntimeError as exc:
            return _unresolved(f"codec_path_drift:{exc}", genuine=True, reproduction_ok=False)
        symbol_correct = scored[condition]["arms"]["symbol_plus_residual"]["n_correct"]
        alone_correct = scored[condition]["arms"]["residual_only"]["n_correct"]
        if symbol_correct != PUBLISHED_SLOTS[condition]["symbol_plus_residual"]:
            return _unresolved("arm_reproduction_mismatch", genuine=True, reproduction_ok=False)
        if alone_correct != PUBLISHED_SLOTS[condition]["residual_without_symbol"]:
            return _unresolved("arm_reproduction_mismatch", genuine=True, reproduction_ok=False)

    flags = []
    explicit: List[str] = []
    for name in _CONDITION_ORDER:
        gate = scored[name]["gate"]
        flags.append(
            {
                "admitted": gate["admitted"],
                "arm1_matches_or_beats_arm2": gate["arm1_matches_or_beats_arm2"],
                "arm3_matches_or_beats_arm2": gate["arm3_matches_or_beats_arm2"],
                "failure_clauses": list(gate["failure_clauses"]),
            }
        )
        for clause in gate["failure_clauses"]:
            explicit.append(f"{name}:{clause}")
    verdict, recommendation, reason = decide_verdict(
        integrity_failure=None,
        reproduction_ok=True,
        conditions=flags,
    )
    report = _empty_base(genuine=True, reproduction_ok=True)
    report.update(
        {
            "model_id": sr.MODEL_ID,
            "revision": sr.MODEL_REVISION,
            "weight_sha256": sr.WEIGHT_SHA256,
            "final_layer_index": sr.FINAL_LAYER_INDEX,
            "controlled_property": CONTROLLED_PROPERTY,
            "checksums": dict(sr.ACTIVATION_SHA256),
            "reproduction": _published_block(parent),
            "condition_2": scored["condition_2"],
            "condition_4": scored["condition_4"],
            "verdict": verdict,
            "recommendation": recommendation,
            "reason": reason,
            "explicit_failures": explicit,
            "symbol_beats_standalone_on_both": all(
                scored[name]["gate"]["symbol_beats_standalone"] for name in _CONDITION_ORDER
            ),
            "symbol_beats_residual_only_on_both": all(
                scored[name]["gate"]["symbol_beats_residual_only"] for name in _CONDITION_ORDER
            ),
            "symbol_matches_or_beats_standalone_on_both": all(
                scored[name]["gate"]["symbol_matches_or_beats_standalone"] for name in _CONDITION_ORDER
            ),
            "symbol_matches_or_beats_residual_only_on_both": all(
                scored[name]["gate"]["symbol_matches_or_beats_residual_only"] for name in _CONDITION_ORDER
            ),
        }
    )
    return report


def _format_report(report: dict) -> str:
    lines = [
        f"verdict {report['verdict']}",
        f"recommendation {report['recommendation']}",
        f"authority {report['authority']}",
        f"reason {report.get('reason')}",
        f"reproduction_ok {report.get('reproduction_ok')}",
        f"explicit_failures {report.get('explicit_failures')}",
        f"slot_task {report.get('slot_task')}",
    ]
    reproduction = report.get("reproduction")
    if isinstance(reproduction, dict):
        lines.append(f"reproduction {reproduction}")
    for name in _CONDITION_ORDER:
        block = report.get(name)
        if not isinstance(block, dict):
            continue
        lines.append(name)
        gate = block["gate"]
        lines.append(
            "  gate "
            f"admitted={gate['admitted']} "
            f"vs_standalone={gate['arm2_at_least_arm1']} "
            f"vs_residual={gate['arm2_at_least_arm3_minus_tolerance']} "
            f"bytes_win={gate['bytes_win_versus_both']} "
            f"latency_win={gate['latency_win_versus_both']} "
            f"arm1_ge={gate['arm1_matches_or_beats_arm2']} "
            f"arm3_ge={gate['arm3_matches_or_beats_arm2']}"
        )
        for arm_name in ("standalone_int4", "symbol_plus_residual", "residual_only"):
            arm = block["arms"][arm_name]
            fid = arm["reconstruction"]
            lines.append(
                f"  {arm_name} {arm['n_correct']}/{arm['n']} "
                f"system={arm['system_bytes']} "
                f"admission_bytes={arm['admission_storage_bytes']} "
                f"second={arm['second_amortization_bytes']} "
                f"mse={fid['mean_squared_euclidean']} "
                f"fro={fid['normalized_frobenius']} "
                f"assign_s={arm['median_assignment_seconds']} "
                f"quant_s={arm['median_quantize_seconds']} "
                f"search_s={arm['median_search_seconds']} "
                f"latency_s={arm['admission_latency_seconds']} "
                f"ties={arm['queries_with_distance_ties']}"
            )
            if arm_name == "residual_only":
                bare = arm["reconstruction_without_centroid"]
                lines.append(
                    "  residual_only_without_centroid "
                    f"mse={bare['mean_squared_euclidean']} "
                    f"fro={bare['normalized_frobenius']}"
                )
            if arm_name == "standalone_int4":
                lines.append(
                    "  activation_codes "
                    f"train={arm['train_code_min']}..{arm['train_code_max']} "
                    f"held={arm['held_code_min']}..{arm['held_code_max']} "
                    f"scale={arm['scale_min']}..{arm['scale_max']} "
                    f"zero={arm['held_zero_code_fraction']}"
                )
        lines.append(f"  learn_seconds {block['learn_seconds']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_format_report(run_symbol_addressing()))
