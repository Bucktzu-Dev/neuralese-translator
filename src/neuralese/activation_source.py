"""Feasibility of the existing pack on a frozen model's activations.

This module does not download weights, run a model, or replace
``word_sentence_svd``. ``learn_pack`` keeps its default encoder. Rows that
already carry an embedding stay on that vector. Clustering is production
``kmeans`` (farthest-first, squared Euclidean), fit on training activations
only. Held-out rows are assigned to those frozen centroids. ``certify`` keeps
the library taus.

The admission constants above ``run_activation_source`` were fixed before any
held-out residual was read. A gloss is not an input to the verdict.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuralese.adapters import hashed_ngram_vector
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.clustering import kmeans
from neuralese.contracts import Observation, canonical_dumps
from neuralese.encode import ENCODER_CHAR_TRIGRAM
from neuralese.factorization import reconstruction_error
from neuralese.translator import translate_stream

# Status of the trial this experiment starts from. Not this run's verdict.
STATUS = "RESEARCH_ONLY"
BASELINE_COMMIT = "c6321d5"
ENTERING_VERDICT = "UNRESOLVED"
ENTERING_RECOMMENDATION = "NONE"
RUNTIME_AUTHORITY = "RUNTIME_NOT_AUTHORIZED"

# Frozen checkpoint. Hashes were computed from the files before the forward pass.
MODEL_ID = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
MODEL_URL = (
    "https://huggingface.co/EleutherAI/pythia-14m/commit/"
    "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
)
WEIGHT_FILE = "model.safetensors"
WEIGHT_BYTES = 28143920
WEIGHT_SHA256 = "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"
CONFIG_SHA256 = "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17"
TOKENIZER_SHA256 = "870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2"
TOKENIZER_CONFIG_SHA256 = "eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809"
SPECIAL_TOKENS_SHA256 = "10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c"
GENERATION_CONFIG_SHA256 = "494cbba887cc299273f0c54d7612596918e74c7881a27cbd95e4e7e2464ea090"
EXPECTED_HIDDEN_SIZE = 128
EXPECTED_LAYERS = 6
EXPECTED_VOCAB = 50304
EXPECTED_ARCHITECTURE = "GPTNeoXForCausalLM"
CHECKPOINT_WEIGHT_DTYPE = "float16"
FORWARD_DTYPE = "float32"
STORED_DTYPE = "float32"

# Coordinate rule. The capture takes the last tensor in outputs.hidden_states
# at the final position of each unpadded prompt. add_special_tokens is false.
# No padding is applied, so that position is the last non-padding token.
LAYER_RULE = "last tensor in outputs.hidden_states"
TOKEN_RULE = (
    "final position of an unpadded prompt; add_special_tokens false; no padding"
)

# Admission gates. Fixed before any held-out residual was computed.
# Geometry has no numeric slack: the prototype residual must be strictly lower.
GEOMETRY_GATE = (
    "heldout_prototype_residual_strictly_below_train_mean_and_shuffled_assignment"
)
GLOBAL_MEAN_GATE = "heldout_prototype_residual_strictly_below_residual_to_train_mean"
SHUFFLED_ASSIGNMENT_GATE = (
    "heldout_prototype_residual_strictly_below_residual_after_permuting_"
    "heldout_assignments_to_frozen_train_centroids"
)
# Relative drop required of the held-out logit residual against the shuffled
# symbol-id residual. (shuffled - true) / shuffled >= margin, and shuffled > 0.
DOWNSTREAM_GATE = (
    "heldout_logit_residual_relative_drop_vs_shuffled_symbol_id_at_least_margin"
)
DOWNSTREAM_RELATIVE_MARGIN = 0.05

# Control seeds. Fixed before scoring. Production k-means ignores its rng.
SHUFFLE_ASSIGNMENT_SEED = 0
SHUFFLE_SYMBOL_SEED = 1
STABILITY_SEED = 1
SUBSAMPLE_SEED = 2
SUBSAMPLE_FRACTION = 0.8
LEARN_SEED = 0
UNKNOWN_CODE = 10**9

VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_GEOMETRY_ONLY = "GEOMETRY_ONLY"
VERDICT_STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"
VERDICT_PASS = "PASS_FOR_LATER_PR"
RECOMMENDATION_NONE = "NONE"
RECOMMENDATION_LATER_PR = "LATER_ISOLATED_PR"
VERDICTS = (
    VERDICT_UNRESOLVED,
    VERDICT_GEOMETRY_ONLY,
    VERDICT_STRUCTURAL_FAILURE,
    VERDICT_PASS,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "examples" / "activation_source"
FIXTURE_NPZ = FIXTURE_DIR / "activations.npz"
FIXTURE_META = FIXTURE_DIR / "capture_meta.json"

# Disjoint prompts. Fixed before the forward pass. Original sentences.
# Every row ends with a period so the final-token rule is the same shape on
# each string. The period is part of the prompt, not a filter applied later.
TRAIN_PROMPTS: Tuple[str, ...] = (
    "The baker scored the loaf before it went into the oven.",
    "A cedar fence surrounds the small vegetable plot.",
    "She tuned the violin until the open string stopped beating.",
    "Rain collected in the barrel beside the greenhouse door.",
    "The ferry sounded its horn before leaving the harbor.",
    "He labeled each jar of honey with the harvest date.",
    "The potter centered the clay and opened the walls of the bowl.",
    "A red kite lifted over the empty soccer field.",
    "The librarian wheeled the returns cart past the atlas stand.",
    "Steam fogged the window above the soup pot.",
    "They patched the bicycle tire under the maple tree.",
    "The night clerk counted the room keys twice.",
    "Moss grew along the shaded side of the stone wall.",
    "A child stacked smooth pebbles on the picnic table.",
    "The mechanic wiped oil from the wrench and closed the hood.",
    "Bread cooled on the rack while the kettle whistled.",
    "The gardener tied the tomato vines to bamboo stakes.",
    "Fog hid the far end of the wooden pier.",
    "She copied the recipe onto a card and filed it.",
    "The drummer tightened the snare before the rehearsal.",
    "Apples filled the crate beside the cider press.",
    "He sanded the chair leg until the grain felt even.",
    "The lantern threw a small circle of light on the map.",
    "A gull stood on the post at the end of the dock.",
    "The nurse wrote the pulse in the margin of the chart.",
    "Snow slid off the greenhouse roof in one sheet.",
    "They rinsed the paintbrushes in a jar of cloudy water.",
    "The clock in the bakery struck four.",
    "A narrow path led between the blackberry canes.",
    "He folded the letter and sealed it with wax.",
    "The kiln cooled overnight before the shelves were unloaded.",
    "Wind moved the laundry on the line behind the cottage.",
)
HELDOUT_PROMPTS: Tuple[str, ...] = (
    "The archivist boxed the brittle newspapers in acid-free folders.",
    "A mule carried water cans up the switchback trail.",
    "She calibrated the microscope and slid the sample into place.",
    "Thunder rolled across the dry lake bed after midnight.",
    "The cobbler replaced the worn heel and polished the toe.",
    "Ice formed along the rim of the cattle trough.",
    "He sketched the bridge cables from the riverbank.",
    "The choir held the last chord until the director closed a hand.",
    "Salt crusted the rope that tied the skiff to the ring.",
    "A sparrow hopped along the bakery windowsill.",
    "They measured the doorway before ordering the new stove.",
    "The printer jammed on the third page of the invoice.",
    "Embers settled in the fire pit behind the cabin.",
    "She wound the clock and set it back on the mantel.",
    "The surveyor planted a stake at the corner of the field.",
    "Orange peels dried on a screen in the shed.",
)


def prompt_fingerprint() -> str:
    payload = {"heldout": list(HELDOUT_PROMPTS), "train": list(TRAIN_PROMPTS)}
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def observation_id(split: str, index: int) -> str:
    """1-based id. ``index`` is the position in the frozen prompt tuple."""
    return f"{split}-{index:03d}"


def train_ids() -> List[str]:
    return [observation_id("train", i) for i in range(1, len(TRAIN_PROMPTS) + 1)]


def heldout_ids() -> List[str]:
    return [observation_id("heldout", i) for i in range(1, len(HELDOUT_PROMPTS) + 1)]


def validate_prompts() -> None:
    train = list(TRAIN_PROMPTS)
    held = list(HELDOUT_PROMPTS)
    if len(train) < 2 or len(held) < 2:
        raise ValueError("train and held-out prompts must both be non-trivial")
    if len(set(train)) != len(train) or len(set(held)) != len(held):
        raise ValueError("duplicate prompt inside a split")
    if set(train) & set(held):
        raise ValueError("train and held-out prompts intersect")
    for text in train + held:
        if not text or text != text.strip() or not text.endswith("."):
            raise ValueError(f"prompt is not a stripped sentence: {text!r}")
    for left in train:
        for right in held:
            if left in right or right in left:
                raise ValueError("a prompt contains the other split's prompt")


def float32_sha256(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
    return hashlib.sha256(raw).hexdigest()


def int32_sha256(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array, dtype=np.int32).tobytes()
    return hashlib.sha256(raw).hexdigest()


def assign_squared_euclidean(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared-Euclidean argmin, the assignment step inside production ``kmeans``.

    ``clustering.py`` is unchanged. Held-out rows are not passed to ``kmeans``.
    """
    if matrix.ndim != 2 or centroids.ndim != 2:
        raise ValueError("assignment expects two matrices")
    if matrix.shape[1] != centroids.shape[1]:
        raise ValueError("assignment width mismatch")
    dist = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1).astype(np.int64)


def geometry_beats(prototype: float, global_mean: float, shuffled: float) -> bool:
    """True only when the prototype residual is strictly below both baselines."""
    if not all(np.isfinite(v) for v in (prototype, global_mean, shuffled)):
        return False
    return prototype < global_mean and prototype < shuffled


def downstream_beats(true_residual: float, shuffled_residual: float) -> bool:
    """True when the true symbol id beats a shuffled id by the frozen margin."""
    if not np.isfinite(true_residual) or not np.isfinite(shuffled_residual):
        return False
    if shuffled_residual <= 0.0:
        return False
    relative = (shuffled_residual - true_residual) / shuffled_residual
    return relative >= DOWNSTREAM_RELATIVE_MARGIN and true_residual < shuffled_residual


def decide_verdict(
    *,
    genuine: bool,
    structural_pass: bool,
    geometry_ok: bool,
    downstream_present: bool,
    downstream_ok: bool,
    fixture_reproducible: bool,
) -> Tuple[str, str, str]:
    """Return verdict, recommendation, and the first failing reason.

    Authority stays ``RUNTIME_NOT_AUTHORIZED`` for every branch. A pass is a
    recommendation for a later isolated PR, not a production change.
    """
    if not genuine or not fixture_reproducible:
        return (
            VERDICT_UNRESOLVED,
            RECOMMENDATION_NONE,
            "capture_not_genuine_or_not_reproducible",
        )
    if not structural_pass:
        return (
            VERDICT_STRUCTURAL_FAILURE,
            RECOMMENDATION_NONE,
            "train_pack_failed_structural_certify",
        )
    if not geometry_ok:
        return (
            VERDICT_UNRESOLVED,
            RECOMMENDATION_NONE,
            "geometry_controls_did_not_separate",
        )
    if not downstream_present:
        return (
            VERDICT_GEOMETRY_ONLY,
            RECOMMENDATION_NONE,
            "geometry_without_downstream_quantity",
        )
    if not downstream_ok:
        return (
            VERDICT_UNRESOLVED,
            RECOMMENDATION_NONE,
            "downstream_control_did_not_separate",
        )
    return (
        VERDICT_PASS,
        RECOMMENDATION_LATER_PR,
        "all_admission_gates_passed",
    )


def load_meta(path: Path = FIXTURE_META) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_arrays(path: Path = FIXTURE_NPZ) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _checksums(arrays: Mapping[str, np.ndarray]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, array in arrays.items():
        if array.dtype == np.float32 or np.issubdtype(array.dtype, np.floating):
            out[key] = float32_sha256(array)
        elif array.dtype == np.int32 or np.issubdtype(array.dtype, np.integer):
            out[key] = int32_sha256(array)
        else:
            raise ValueError(f"unsupported fixture dtype for {key}: {array.dtype}")
    return out


def assess_fixture(
    npz_path: Path = FIXTURE_NPZ,
    meta_path: Path = FIXTURE_META,
) -> dict:
    """Check the committed capture against the frozen boundary.

    A missing file, a hash mismatch, or a prompt mismatch is not a model
    activation. The caller must not invent a replacement matrix.
    """
    failures: List[str] = []
    if not npz_path.is_file() or not meta_path.is_file():
        return {
            "genuine": False,
            "fixture_reproducible": False,
            "blocker": "fixture_missing",
            "failures": ["fixture_missing"],
        }
    meta = load_meta(meta_path)
    arrays = load_arrays(npz_path)
    expected_meta = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "url": MODEL_URL,
        "weight_file": WEIGHT_FILE,
        "weight_bytes": WEIGHT_BYTES,
        "weight_sha256": WEIGHT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "special_tokens_sha256": SPECIAL_TOKENS_SHA256,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "checkpoint_weight_dtype": CHECKPOINT_WEIGHT_DTYPE,
        "forward_dtype": FORWARD_DTYPE,
        "stored_dtype": STORED_DTYPE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": EXPECTED_LAYERS,
        "vocab_size": EXPECTED_VOCAB,
        "architecture": EXPECTED_ARCHITECTURE,
        "layer_rule": LAYER_RULE,
        "token_rule": TOKEN_RULE,
        "prompt_fingerprint": prompt_fingerprint(),
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            failures.append(f"meta_{key}")
    if list(meta.get("train_prompts") or []) != list(TRAIN_PROMPTS):
        failures.append("train_prompts")
    if list(meta.get("heldout_prompts") or []) != list(HELDOUT_PROMPTS):
        failures.append("heldout_prompts")
    hidden_index = meta.get("hidden_state_index")
    tuple_length = meta.get("hidden_states_tuple_length")
    if not isinstance(hidden_index, int) or not isinstance(tuple_length, int):
        failures.append("hidden_state_index")
    elif hidden_index != tuple_length - 1 or hidden_index < 0:
        failures.append("hidden_state_index_not_last")
    recomputed = _checksums(arrays)
    recorded = dict(meta.get("checksums") or {})
    if recomputed != recorded:
        failures.append("activation_checksum_mismatch")
    shapes = {
        "train_activations": (len(TRAIN_PROMPTS), EXPECTED_HIDDEN_SIZE),
        "heldout_activations": (len(HELDOUT_PROMPTS), EXPECTED_HIDDEN_SIZE),
        "train_logits": (len(TRAIN_PROMPTS), EXPECTED_VOCAB),
        "heldout_logits": (len(HELDOUT_PROMPTS), EXPECTED_VOCAB),
    }
    for key, shape in shapes.items():
        array = arrays.get(key)
        if array is None or tuple(array.shape) != shape or array.dtype != np.float32:
            failures.append(f"shape_{key}")
        elif not np.isfinite(array).all():
            failures.append(f"nonfinite_{key}")
    _check_token_coordinate(arrays, "train", failures)
    _check_token_coordinate(arrays, "heldout", failures)
    capture_seconds = meta.get("capture_seconds")
    if not isinstance(capture_seconds, (int, float)) or capture_seconds < 0:
        failures.append("capture_seconds")
    genuine = not failures
    return {
        "genuine": genuine,
        "fixture_reproducible": genuine,
        "blocker": None if genuine else failures[0],
        "failures": failures,
        "meta": meta if genuine else meta,
        "arrays": arrays if genuine else None,
        "checksums": recomputed,
    }


def _check_token_coordinate(
    arrays: Mapping[str, np.ndarray],
    split: str,
    failures: List[str],
) -> None:
    ids = arrays.get(f"{split}_token_ids")
    lengths = arrays.get(f"{split}_token_lengths")
    last = arrays.get(f"{split}_last_token_id")
    if ids is None or lengths is None or last is None:
        failures.append(f"tokens_{split}")
        return
    if ids.dtype != np.int32 or lengths.dtype != np.int32 or last.dtype != np.int32:
        failures.append(f"token_dtype_{split}")
        return
    if len(lengths) != len(last) or ids.shape[0] != len(lengths):
        failures.append(f"token_shape_{split}")
        return
    for row, length in enumerate(lengths.tolist()):
        if length < 1 or length > ids.shape[1]:
            failures.append(f"token_length_{split}")
            return
        if int(ids[row, length - 1]) != int(last[row]):
            failures.append(f"last_token_{split}")
            return
        if length < ids.shape[1] and np.any(ids[row, length:] != -1):
            failures.append(f"token_padding_{split}")
            return


def _observations(ids: Sequence[str], vectors: np.ndarray, texts: Sequence[str]) -> List[Observation]:
    rows: List[Observation] = []
    for obs_id, vector, text in zip(ids, vectors, texts):
        rows.append(
            Observation(
                observation_id=obs_id,
                embedding=np.asarray(vector, dtype=np.float64).tolist(),
                text=text,
            )
        )
    return rows


def _labels_from_pack(pack, ids: Sequence[str]) -> np.ndarray:
    index: Dict[str, int] = {}
    for symbol in pack.symbols:
        for obs_id in symbol.observation_ids:
            index[obs_id] = int(symbol.class_id)
    missing = [obs_id for obs_id in ids if obs_id not in index]
    if missing:
        raise ValueError(f"train id missing from pack: {missing[0]}")
    return np.asarray([index[obs_id] for obs_id in ids], dtype=np.int64)


def _matched_agreement(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Maximum fraction of rows agreeing after a one-to-one label remap."""
    if left.shape != right.shape or left.size == 0:
        return 0.0
    best = 0
    for perm in itertools.permutations(range(k)):
        mapped = np.fromiter((perm[int(label)] for label in right), dtype=np.int64, count=right.shape[0])
        agreed = int(np.sum(mapped == left))
        if agreed > best:
            best = agreed
    return float(best / left.shape[0])


def _symbol_logit_means(
    labels: np.ndarray,
    logits: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Train-only mean logit per symbol. Empty symbols fall back to the train mean.

    The fallback count is the number of symbol ids, not the number of rows.
    """
    width = logits.shape[1]
    means = np.zeros((k, width), dtype=np.float64)
    global_mean = logits.mean(axis=0)
    fallbacks = 0
    for class_id in range(k):
        members = logits[labels == class_id]
        if members.shape[0] == 0:
            means[class_id] = global_mean
            fallbacks += 1
        else:
            means[class_id] = members.mean(axis=0)
    return means, global_mean, fallbacks


def _predict_residual(logits: np.ndarray, symbol_ids: np.ndarray, means: np.ndarray) -> float:
    predicted = means[symbol_ids]
    return reconstruction_error(logits, predicted)


def _subsample_agreement(matrix: np.ndarray, labels: np.ndarray, k: int) -> float:
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    n = matrix.shape[0]
    take = max(k, int(round(SUBSAMPLE_FRACTION * n)))
    take = min(take, n)
    chosen = np.sort(rng.choice(n, size=take, replace=False))
    _sub_labels, sub_centroids = kmeans(
        matrix[chosen],
        k,
        rng=np.random.default_rng(LEARN_SEED),
    )
    assigned = assign_squared_euclidean(matrix, sub_centroids)
    return _matched_agreement(labels, assigned, k)


def run_activation_source(
    npz_path: Path = FIXTURE_NPZ,
    meta_path: Path = FIXTURE_META,
) -> dict:
    """Learn on train activations, assign held-out rows, and apply the gates.

    The text of a row is stored for the heuristic gloss only. It is not an
    embedding, and the verdict does not read the gloss.
    """
    validate_prompts()
    cfg = LearnConfig()
    if cfg.encoder != ENCODER_CHAR_TRIGRAM:
        raise RuntimeError("learn_pack default encoder changed")
    if cfg.tau_residual != 0.55 or cfg.tau_kappa != 0.35:
        raise RuntimeError("library taus changed")
    if cfg.n_symbols != 8 or cfg.seed != LEARN_SEED:
        raise RuntimeError("library learn defaults changed")

    assessment = assess_fixture(npz_path, meta_path)
    explicit: List[str] = list(assessment["failures"])
    if not assessment["genuine"]:
        verdict, recommendation, reason = decide_verdict(
            genuine=False,
            structural_pass=False,
            geometry_ok=False,
            downstream_present=False,
            downstream_ok=False,
            fixture_reproducible=False,
        )
        return {
            "status": STATUS,
            "authority": RUNTIME_AUTHORITY,
            "baseline_commit": BASELINE_COMMIT,
            "entering_verdict": ENTERING_VERDICT,
            "entering_recommendation": ENTERING_RECOMMENDATION,
            "genuine": False,
            "fixture_reproducible": False,
            "blocker": assessment["blocker"],
            "verdict": verdict,
            "recommendation": recommendation,
            "reason": reason,
            "explicit_failures": explicit,
            "gloss_used_as_evidence": False,
        }

    meta = assessment["meta"]
    arrays = assessment["arrays"]
    train = np.asarray(arrays["train_activations"], dtype=np.float64)
    held = np.asarray(arrays["heldout_activations"], dtype=np.float64)
    train_logits = np.asarray(arrays["train_logits"], dtype=np.float64)
    held_logits = np.asarray(arrays["heldout_logits"], dtype=np.float64)
    ids = train_ids()
    held_ids = heldout_ids()
    train_obs = _observations(ids, train, TRAIN_PROMPTS)

    # Warm one factorization so the recorded cluster time is the pack itself.
    learn_pack(train_obs, config=LearnConfig())
    cluster_start = time.perf_counter()
    pack = learn_pack(train_obs, config=LearnConfig())
    cluster_seconds = time.perf_counter() - cluster_start

    k = int(cfg.n_symbols)
    produced_labels, centroids = kmeans(train, k, rng=np.random.default_rng(LEARN_SEED))
    pack_labels = _labels_from_pack(pack, ids)
    if produced_labels.shape != pack_labels.shape or not np.array_equal(produced_labels, pack_labels):
        explicit.append("pack_labels_differ_from_production_kmeans")
    class_ids = sorted(int(symbol.class_id) for symbol in pack.symbols)
    if class_ids != list(range(centroids.shape[0])):
        explicit.append("class_ids_not_contiguous")
    for symbol in pack.symbols:
        if not symbol.observation_ids:
            continue
        proto = np.asarray(symbol.proto_embedding, dtype=np.float64)
        if proto.shape != centroids[symbol.class_id].shape or not np.allclose(
            proto, centroids[symbol.class_id]
        ):
            explicit.append(f"proto_mismatch_class_{symbol.class_id}")
            break

    reservoir_ok = True
    for obs_id, vector, text in zip(ids, train, TRAIN_PROMPTS):
        stored = pack.observations.get(obs_id)
        if stored is None or stored.text != text:
            reservoir_ok = False
            explicit.append("reservoir_row_missing")
            break
        stored_vec = np.asarray(stored.embedding, dtype=np.float64)
        if stored_vec.shape != vector.shape or not np.array_equal(stored_vec, vector):
            reservoir_ok = False
            explicit.append("text_encoder_replaced_embeddings")
            break
        hashed = np.asarray(hashed_ngram_vector(text), dtype=np.float64)
        if hashed.shape == stored_vec.shape and np.allclose(hashed, stored_vec):
            reservoir_ok = False
            explicit.append("embedding_equals_char_trigram")
            break
    if any(obs_id in pack.observations for obs_id in held_ids):
        reservoir_ok = False
        explicit.append("heldout_row_entered_the_pack")

    lookup_start = time.perf_counter()
    held_labels = assign_squared_euclidean(held, centroids)
    code_of_class = {int(symbol.class_id): int(symbol.code) for symbol in pack.symbols}
    held_codes = [code_of_class[int(label)] for label in held_labels]
    glosses = translate_stream(pack, held_codes)
    unknown = translate_stream(pack, [UNKNOWN_CODE])
    lookup_seconds = time.perf_counter() - lookup_start
    unknown_state = unknown[0].state if unknown else None
    if unknown_state != "unknown":
        explicit.append("unknown_code_not_fail_closed")

    certify_start = time.perf_counter()
    certificate = certify(pack)
    certify_seconds = time.perf_counter() - certify_start

    tracemalloc.start()
    try:
        timed_pack = learn_pack(train_obs, config=LearnConfig())
        _timed_labels = assign_squared_euclidean(held, centroids)
        translate_stream(timed_pack, held_codes)
        certify(timed_pack)
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    structural_pass = bool(
        certificate.passed
        and reservoir_ok
        and unknown_state == "unknown"
        and "pack_labels_differ_from_production_kmeans" not in explicit
        and "text_encoder_replaced_embeddings" not in explicit
        and "heldout_row_entered_the_pack" not in explicit
    )
    if not certificate.passed:
        explicit.extend(f"certify: {item}" for item in certificate.failures)

    cluster_residual = reconstruction_error(train, centroids[produced_labels])
    empty_clusters = int(sum(1 for class_id in range(centroids.shape[0]) if not np.any(produced_labels == class_id)))
    prototype_hat = centroids[held_labels]
    prototype_residual = reconstruction_error(held, prototype_hat)
    train_mean = train.mean(axis=0)
    mean_hat = np.repeat(train_mean[None, :], held.shape[0], axis=0)
    global_mean_residual = reconstruction_error(held, mean_hat)
    shuffle_rng = np.random.default_rng(SHUFFLE_ASSIGNMENT_SEED)
    shuffled_labels = held_labels[shuffle_rng.permutation(held_labels.shape[0])]
    shuffled_residual = reconstruction_error(held, centroids[shuffled_labels])
    geometry_ok = geometry_beats(prototype_residual, global_mean_residual, shuffled_residual)

    means, _logit_mean, fallbacks = _symbol_logit_means(produced_labels, train_logits, centroids.shape[0])
    true_logit_residual = _predict_residual(held_logits, held_labels, means)
    symbol_rng = np.random.default_rng(SHUFFLE_SYMBOL_SEED)
    shuffled_symbols = held_labels[symbol_rng.permutation(held_labels.shape[0])]
    shuffled_logit_residual = _predict_residual(held_logits, shuffled_symbols, means)
    downstream_present = True
    downstream_ok = downstream_beats(true_logit_residual, shuffled_logit_residual)
    if shuffled_logit_residual > 0.0:
        downstream_relative = (shuffled_logit_residual - true_logit_residual) / shuffled_logit_residual
    else:
        downstream_relative = None

    nearest_agreement = float(np.mean(assign_squared_euclidean(train, centroids) == produced_labels))
    seed_labels, _seed_centroids = kmeans(train, k, rng=np.random.default_rng(STABILITY_SEED))
    seed_agreement = _matched_agreement(produced_labels, seed_labels, k)
    subsample_agreement = _subsample_agreement(train, produced_labels, k)

    n_train = int(train.shape[0])
    n_held = int(held.shape[0])
    n_all = n_train + n_held
    capture_seconds = float(meta["capture_seconds"])
    verdict, recommendation, reason = decide_verdict(
        genuine=True,
        structural_pass=structural_pass,
        geometry_ok=geometry_ok,
        downstream_present=downstream_present,
        downstream_ok=downstream_ok,
        fixture_reproducible=True,
    )
    live = [symbol for symbol in pack.symbols if not symbol.quarantined]
    return {
        "status": STATUS,
        "authority": RUNTIME_AUTHORITY,
        "baseline_commit": BASELINE_COMMIT,
        "entering_verdict": ENTERING_VERDICT,
        "entering_recommendation": ENTERING_RECOMMENDATION,
        "genuine": True,
        "fixture_reproducible": True,
        "blocker": None,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "url": MODEL_URL,
        "layer_rule": LAYER_RULE,
        "hidden_state_index": int(meta["hidden_state_index"]),
        "hidden_states_tuple_length": int(meta["hidden_states_tuple_length"]),
        "num_hidden_layers": EXPECTED_LAYERS,
        "token_rule": TOKEN_RULE,
        "forward_dtype": FORWARD_DTYPE,
        "stored_dtype": STORED_DTYPE,
        "checkpoint_weight_dtype": CHECKPOINT_WEIGHT_DTYPE,
        "weight_sha256": WEIGHT_SHA256,
        "activation_shape": [n_train, EXPECTED_HIDDEN_SIZE],
        "heldout_shape": [n_held, EXPECTED_HIDDEN_SIZE],
        "logit_shape": [n_held, EXPECTED_VOCAB],
        "prompt_fingerprint": prompt_fingerprint(),
        "checksums": assessment["checksums"],
        "codebook_size": len(pack.codebook),
        "n_symbols": len(pack.symbols),
        "n_live": len(live),
        "n_quarantined": sum(1 for symbol in pack.symbols if symbol.quarantined),
        "train_residual": float(pack.reconstruction_error),
        "cluster_residual": cluster_residual,
        "n_empty_clusters": empty_clusters,
        "kappa_avg": None if pack.guards is None else float(pack.guards.kappa_avg),
        "decision": pack.metadata.get("decision"),
        "structural_pass": structural_pass,
        "certify_passed": bool(certificate.passed),
        "certify_failures": list(certificate.failures),
        "unknown_code": UNKNOWN_CODE,
        "unknown_state": unknown_state,
        "receipt_steps": [receipt.step for receipt in pack.receipts],
        "encoder": pack.metadata.get("config", {}).get("encoder"),
        "tau_residual": cfg.tau_residual,
        "tau_kappa": cfg.tau_kappa,
        "geometry": {
            "prototype_residual": prototype_residual,
            "global_mean_residual": global_mean_residual,
            "shuffled_assignment_residual": shuffled_residual,
            "beats": geometry_ok,
            "gate": GEOMETRY_GATE,
        },
        "downstream": {
            "present": downstream_present,
            "quantity": "next_token_logits_at_the_same_final_position",
            "used_in_fit": False,
            "true_residual": true_logit_residual,
            "shuffled_symbol_residual": shuffled_logit_residual,
            "relative_drop": downstream_relative,
            "margin": DOWNSTREAM_RELATIVE_MARGIN,
            "beats": downstream_ok,
            "empty_symbol_fallbacks": fallbacks,
            "gate": DOWNSTREAM_GATE,
        },
        "stability": {
            "nearest_centroid_agreement": nearest_agreement,
            "seed_agreement": seed_agreement,
            "seed_note": "production farthest-first ignores the rng",
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "subsample_agreement": subsample_agreement,
        },
        "cost_seconds": {
            "capture": capture_seconds,
            "cluster": cluster_seconds,
            "lookup": lookup_seconds,
            "certify": certify_seconds,
        },
        "cost_per_observation_seconds": {
            "capture": capture_seconds / n_all,
            "cluster": cluster_seconds / n_train,
            "lookup": lookup_seconds / n_held,
            "certify": certify_seconds / n_train,
        },
        "peak_memory": {
            "capture_rss_kb": meta.get("capture_peak_rss_kb"),
            "capture_tracemalloc_bytes": meta.get("capture_tracemalloc_peak_bytes"),
            "trial_tracemalloc_bytes": int(peak_bytes),
        },
        "capture_stack": {
            "torch": meta.get("torch_version"),
            "transformers": meta.get("transformers_version"),
            "download_seconds": meta.get("download_seconds"),
            "load_seconds": meta.get("load_seconds"),
        },
        "gloss_used_as_evidence": False,
        "gloss_sample_state": glosses[0].state if glosses else None,
        "verdict": verdict,
        "recommendation": recommendation,
        "reason": reason,
        "explicit_failures": explicit,
    }
