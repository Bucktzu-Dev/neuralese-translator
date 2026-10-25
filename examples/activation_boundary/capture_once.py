"""One-time capture for the activation-boundary trial.

This script is not part of the test run and not part of CI. It reads the
pinned pythia-14m snapshot already on disk, checks the file hashes frozen in
``neuralese.activation_source``, and writes ``activations.npz`` plus
``capture_meta.json``. It does not cluster and it does not score. Tests load
the fixture with numpy and do not import this module.

Run from the repository root:

    PYTHONPATH=src python3 examples/activation_boundary/capture_once.py --snapshot /path/to/files
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import sys
import time
import tracemalloc
import zipfile
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from neuralese.activation_boundary import (  # noqa: E402
    CONDITION_3_TOKEN_COORDINATE,
    CONTROLLED_PROPERTY,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    EXPECTED_TUPLE_LENGTH,
    FINAL_LAYER_INDEX,
    FIXTURE_META,
    FIXTURE_NPZ,
    FORWARD_DTYPE,
    HELDOUT_FRAMES,
    INTERMEDIATE_LAYER_INDEX,
    PARENT_HELDOUT_PROMPTS,
    PARENT_TRAIN_PROMPTS,
    SHARED_TERMINAL,
    SHARED_TERMINAL_ID,
    SLOT_INDEX,
    SLOT_TOKEN_IDS,
    SLOT_WORDS,
    STORED_DTYPE,
    TRAIN_FRAMES,
    VARIED_TERMINAL_IDS,
    VARIED_TERMINALS,
    build_pair_rows,
    design_fingerprint,
    pair_texts,
    swap_texts,
    validate_design,
)
from neuralese.activation_source import (  # noqa: E402
    CHECKPOINT_WEIGHT_DTYPE,
    CONFIG_SHA256,
    EXPECTED_ARCHITECTURE,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_LAYERS,
    EXPECTED_VOCAB,
    GENERATION_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_URL,
    SPECIAL_TOKENS_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    WEIGHT_BYTES,
    WEIGHT_FILE,
    WEIGHT_SHA256,
    float32_sha256,
    int32_sha256,
    prompt_fingerprint,
)

PINNED_FILES = {
    "config.json": CONFIG_SHA256,
    "generation_config.json": GENERATION_CONFIG_SHA256,
    WEIGHT_FILE: WEIGHT_SHA256,
    "special_tokens_map.json": SPECIAL_TOKENS_SHA256,
    "tokenizer.json": TOKENIZER_SHA256,
    "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
}
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_snapshot(snapshot: Path) -> None:
    for name, expected in PINNED_FILES.items():
        path = snapshot / name
        if not path.is_file():
            raise SystemExit(f"missing {name} in {snapshot}")
        digest = _sha256(path)
        if digest != expected:
            raise SystemExit(f"{name} sha256 {digest} != pinned {expected}")
    weight = snapshot / WEIGHT_FILE
    if weight.stat().st_size != WEIGHT_BYTES:
        raise SystemExit(f"{WEIGHT_FILE} is {weight.stat().st_size} bytes, expected {WEIGHT_BYTES}")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise SystemExit(f"architecture {config.get('architectures')}")
    if int(config["hidden_size"]) != EXPECTED_HIDDEN_SIZE:
        raise SystemExit("hidden_size mismatch")
    if int(config["num_hidden_layers"]) != EXPECTED_LAYERS:
        raise SystemExit("num_hidden_layers mismatch")
    if int(config["vocab_size"]) != EXPECTED_VOCAB:
        raise SystemExit("vocab_size mismatch")
    if str(config.get("torch_dtype")) != CHECKPOINT_WEIGHT_DTYPE:
        raise SystemExit(f"config torch_dtype {config.get('torch_dtype')}")
    from safetensors import safe_open

    dtypes = set()
    with safe_open(weight, framework="numpy") as handle:
        for key in handle.keys():
            dtypes.add(str(handle.get_slice(key).get_dtype()))
    if dtypes != {"F16"}:
        raise SystemExit(f"weight dtypes {sorted(dtypes)}; expected only F16")


def _pad_ids(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(len(row) for row in rows)
    ids = np.full((len(rows), width), -1, dtype=np.int32)
    lengths = np.zeros(len(rows), dtype=np.int32)
    last = np.zeros(len(rows), dtype=np.int32)
    for index, row in enumerate(rows):
        if not row:
            raise SystemExit("tokenizer produced an empty prompt")
        ids[index, : len(row)] = np.asarray(row, dtype=np.int32)
        lengths[index] = len(row)
        last[index] = row[-1]
    return ids, lengths, last


def _validate_pairs(tokenizer, rows: list[dict], terminal_ids: tuple[int, ...]) -> None:
    encoded: list[list[int]] = []
    for row_index, row in enumerate(rows):
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            raise SystemExit(f"empty tokenization for {row['text']!r}")
        expected_terminal = terminal_ids[(row_index // 2) % len(terminal_ids)]
        if int(ids[-1]) != expected_terminal:
            raise SystemExit(f"terminal id {ids[-1]} != {expected_terminal} for {row['text']!r}")
        if int(ids[SLOT_INDEX]) != SLOT_TOKEN_IDS[row["probe"]]:
            raise SystemExit(f"slot id {ids[SLOT_INDEX]} != probe {row['probe']} for {row['text']!r}")
        encoded.append([int(token) for token in ids])
    for index in range(0, len(encoded), 2):
        left, right = encoded[index], encoded[index + 1]
        if len(left) != len(right) or left[-1] != right[-1]:
            raise SystemExit("pair members do not share a terminal token")
        diff = [pos for pos, (a, b) in enumerate(zip(left, right)) if a != b]
        if diff != [SLOT_INDEX]:
            raise SystemExit(f"pair token diff {diff} is not the slot index")
        if left[SLOT_INDEX] != SLOT_TOKEN_IDS[0] or right[SLOT_INDEX] != SLOT_TOKEN_IDS[1]:
            raise SystemExit("pair slot ids are not the frozen token ids")


def _validate_swap(tokenizer, originals: tuple[str, ...], swapped: list[str]) -> None:
    for index, (original, text) in enumerate(zip(originals, swapped)):
        left = tokenizer(original, add_special_tokens=False)["input_ids"]
        right = tokenizer(text, add_special_tokens=False)["input_ids"]
        expected = VARIED_TERMINAL_IDS[index % len(VARIED_TERMINAL_IDS)]
        if left[:-1] != right[:-1] or int(left[-1]) != SHARED_TERMINAL_ID or int(right[-1]) != expected:
            raise SystemExit(f"swap is not a final-token edit: {original!r} -> {text!r}")


def _forward_split(model, tokenizer, texts: list[str], layers: tuple[int, ...]) -> dict:
    import torch

    activations = {layer: [] for layer in layers}
    token_rows: list[list[int]] = []
    tuple_length = None
    forward_seconds = 0.0
    slice_seconds = {layer: 0.0 for layer in layers}
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise SystemExit(f"unexpected tokenize shape {tuple(input_ids.shape)}")
        if "attention_mask" in encoded and not bool(torch.all(encoded["attention_mask"] == 1)):
            raise SystemExit("tokenizer padded a single unpadded prompt")
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        forward_seconds += time.perf_counter() - start
        if outputs.hidden_states is None or len(outputs.hidden_states) != EXPECTED_TUPLE_LENGTH:
            raise SystemExit(f"hidden-state tuple length {None if outputs.hidden_states is None else len(outputs.hidden_states)}")
        if tuple_length is None:
            tuple_length = len(outputs.hidden_states)
        for layer in layers:
            slice_start = time.perf_counter()
            hidden = outputs.hidden_states[layer][0, -1, :]
            if hidden.shape != (EXPECTED_HIDDEN_SIZE,) or hidden.dtype != torch.float32:
                raise SystemExit(f"activation shape {tuple(hidden.shape)} dtype {hidden.dtype}")
            activations[layer].append(hidden.detach().cpu().numpy())
            slice_seconds[layer] += time.perf_counter() - slice_start
        token_rows.append([int(token) for token in input_ids[0].tolist()])
    ids, lengths, last = _pad_ids(token_rows)
    stacked = {
        layer: np.stack(activations[layer]).astype(np.float32, copy=False) for layer in layers
    }
    for layer, matrix in stacked.items():
        if not np.isfinite(matrix).all():
            raise SystemExit(f"non-finite activation at layer {layer}")
    return {
        "activations": stacked,
        "token_ids": ids,
        "token_lengths": lengths,
        "last_token_id": last,
        "forward_seconds": forward_seconds,
        "slice_seconds": slice_seconds,
        "hidden_states_tuple_length": int(tuple_length),
    }


def _pair_arrays(prefix: str, captured: dict, probes: list[int], layer: int) -> dict[str, np.ndarray]:
    lengths = captured["token_lengths"]
    ids = captured["token_ids"]
    slots = np.zeros(len(probes), dtype=np.int32)
    for index, length in enumerate(lengths.tolist()):
        if int(length) <= SLOT_INDEX:
            raise SystemExit("prompt is shorter than the slot index")
        slots[index] = int(ids[index, SLOT_INDEX])
    return {
        f"{prefix}_activations": np.ascontiguousarray(captured["activations"][layer], dtype=np.float32),
        f"{prefix}_token_ids": ids,
        f"{prefix}_token_lengths": lengths,
        f"{prefix}_last_token_id": captured["last_token_id"],
        f"{prefix}_probe": np.asarray(probes, dtype=np.int32),
        f"{prefix}_slot_token_id": slots,
    }


def _capture(snapshot: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    if model.config.model_type != "gpt_neox":
        raise SystemExit(f"model_type {model.config.model_type}")
    parameter_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if parameter_dtypes != {"torch.float32"}:
        raise SystemExit(f"forward parameter dtypes {sorted(parameter_dtypes)}")
    if torch.__version__ != EXPECTED_TORCH:
        raise SystemExit(f"torch {torch.__version__} != {EXPECTED_TORCH}")
    transformers_version = __import__("transformers").__version__
    if transformers_version != EXPECTED_TRANSFORMERS:
        raise SystemExit(f"transformers {transformers_version} != {EXPECTED_TRANSFORMERS}")

    c2_train_rows = build_pair_rows(TRAIN_FRAMES, VARIED_TERMINALS)
    c2_held_rows = build_pair_rows(HELDOUT_FRAMES, VARIED_TERMINALS)
    c4_train_rows = build_pair_rows(TRAIN_FRAMES, (SHARED_TERMINAL,))
    c4_held_rows = build_pair_rows(HELDOUT_FRAMES, (SHARED_TERMINAL,))
    swap_train = swap_texts(PARENT_TRAIN_PROMPTS)
    swap_held = swap_texts(PARENT_HELDOUT_PROMPTS)
    _validate_pairs(tokenizer, c2_train_rows, VARIED_TERMINAL_IDS)
    _validate_pairs(tokenizer, c2_held_rows, VARIED_TERMINAL_IDS)
    _validate_pairs(tokenizer, c4_train_rows, (SHARED_TERMINAL_ID,))
    _validate_pairs(tokenizer, c4_held_rows, (SHARED_TERMINAL_ID,))
    _validate_swap(tokenizer, PARENT_TRAIN_PROMPTS, swap_train)
    _validate_swap(tokenizer, PARENT_HELDOUT_PROMPTS, swap_held)

    tracemalloc.start()
    try:
        varied_train = _forward_split(
            model, tokenizer, [row["text"] for row in c2_train_rows], (FINAL_LAYER_INDEX, INTERMEDIATE_LAYER_INDEX)
        )
        varied_held = _forward_split(
            model, tokenizer, [row["text"] for row in c2_held_rows], (FINAL_LAYER_INDEX, INTERMEDIATE_LAYER_INDEX)
        )
        shared_train = _forward_split(
            model, tokenizer, [row["text"] for row in c4_train_rows], (FINAL_LAYER_INDEX,)
        )
        shared_held = _forward_split(
            model, tokenizer, [row["text"] for row in c4_held_rows], (FINAL_LAYER_INDEX,)
        )
        swap_train_cap = _forward_split(model, tokenizer, swap_train, (FINAL_LAYER_INDEX,))
        swap_held_cap = _forward_split(model, tokenizer, swap_held, (FINAL_LAYER_INDEX,))
    finally:
        _current, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    for captured in (varied_train, varied_held, shared_train, shared_held, swap_train_cap, swap_held_cap):
        if captured["hidden_states_tuple_length"] != EXPECTED_TUPLE_LENGTH:
            raise SystemExit("hidden-state tuple length changed")
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "varied_train": varied_train,
        "varied_held": varied_held,
        "shared_train": shared_train,
        "shared_held": shared_held,
        "swap_train": swap_train_cap,
        "swap_held": swap_held_cap,
        "c2_train_rows": c2_train_rows,
        "c2_held_rows": c2_held_rows,
        "c4_train_rows": c4_train_rows,
        "c4_held_rows": c4_held_rows,
        "swap_train_texts": swap_train,
        "swap_held_texts": swap_held,
        "load_seconds": load_seconds,
        "capture_peak_rss_kb": rss_kb,
        "capture_tracemalloc_peak_bytes": int(peak_trace),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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


def _checksums(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    out = {}
    for key, array in arrays.items():
        if np.issubdtype(array.dtype, np.floating):
            out[key] = float32_sha256(array)
        else:
            out[key] = int32_sha256(array)
    return out


def _scalar(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture pinned pythia-14m boundary activations once.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Directory that already holds the pinned files")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing fixture")
    args = parser.parse_args()
    if FIXTURE_NPZ.exists() and not args.force:
        raise SystemExit(f"{FIXTURE_NPZ} already exists; refusing to recapture without --force")
    validate_design()
    _verify_snapshot(args.snapshot)
    captured = _capture(args.snapshot)
    arrays: dict[str, np.ndarray] = {
        "c2_layer_index": _scalar(FINAL_LAYER_INDEX),
        "c3_layer_index": _scalar(INTERMEDIATE_LAYER_INDEX),
        "c4_layer_index": _scalar(FINAL_LAYER_INDEX),
        "swap_layer_index": _scalar(FINAL_LAYER_INDEX),
    }
    arrays.update(_pair_arrays("c2_train", captured["varied_train"], [row["probe"] for row in captured["c2_train_rows"]], FINAL_LAYER_INDEX))
    arrays.update(_pair_arrays("c2_heldout", captured["varied_held"], [row["probe"] for row in captured["c2_held_rows"]], FINAL_LAYER_INDEX))
    arrays["c3_train_activations"] = np.ascontiguousarray(
        captured["varied_train"]["activations"][INTERMEDIATE_LAYER_INDEX], dtype=np.float32
    )
    arrays["c3_heldout_activations"] = np.ascontiguousarray(
        captured["varied_held"]["activations"][INTERMEDIATE_LAYER_INDEX], dtype=np.float32
    )
    arrays.update(_pair_arrays("c4_train", captured["shared_train"], [row["probe"] for row in captured["c4_train_rows"]], FINAL_LAYER_INDEX))
    arrays.update(_pair_arrays("c4_heldout", captured["shared_held"], [row["probe"] for row in captured["c4_held_rows"]], FINAL_LAYER_INDEX))
    for prefix, block in (
        ("swap_train", captured["swap_train"]),
        ("swap_heldout", captured["swap_held"]),
    ):
        arrays[f"{prefix}_activations"] = np.ascontiguousarray(block["activations"][FINAL_LAYER_INDEX], dtype=np.float32)
        arrays[f"{prefix}_token_ids"] = block["token_ids"]
        arrays[f"{prefix}_token_lengths"] = block["token_lengths"]
        arrays[f"{prefix}_last_token_id"] = block["last_token_id"]
    varied_forward = captured["varied_train"]["forward_seconds"] + captured["varied_held"]["forward_seconds"]
    varied_slice_2 = (
        captured["varied_train"]["slice_seconds"][FINAL_LAYER_INDEX]
        + captured["varied_held"]["slice_seconds"][FINAL_LAYER_INDEX]
    )
    varied_slice_3 = (
        captured["varied_train"]["slice_seconds"][INTERMEDIATE_LAYER_INDEX]
        + captured["varied_held"]["slice_seconds"][INTERMEDIATE_LAYER_INDEX]
    )
    condition_4 = (
        captured["shared_train"]["forward_seconds"]
        + captured["shared_held"]["forward_seconds"]
        + captured["shared_train"]["slice_seconds"][FINAL_LAYER_INDEX]
        + captured["shared_held"]["slice_seconds"][FINAL_LAYER_INDEX]
    )
    parent_swap = (
        captured["swap_train"]["forward_seconds"]
        + captured["swap_held"]["forward_seconds"]
        + captured["swap_train"]["slice_seconds"][FINAL_LAYER_INDEX]
        + captured["swap_held"]["slice_seconds"][FINAL_LAYER_INDEX]
    )
    meta = {
        "capture_condition_2_forward_seconds": varied_forward,
        "capture_condition_2_slice_seconds": varied_slice_2,
        "capture_condition_3_slice_seconds": varied_slice_3,
        "capture_condition_4_seconds": condition_4,
        "capture_parent_swap_seconds": parent_swap,
        "capture_peak_rss_kb": captured["capture_peak_rss_kb"],
        "capture_tracemalloc_peak_bytes": captured["capture_tracemalloc_peak_bytes"],
        "c2_heldout_prompts": pair_texts(HELDOUT_FRAMES, VARIED_TERMINALS),
        "c2_train_prompts": pair_texts(TRAIN_FRAMES, VARIED_TERMINALS),
        "c4_heldout_prompts": pair_texts(HELDOUT_FRAMES, (SHARED_TERMINAL,)),
        "c4_train_prompts": pair_texts(TRAIN_FRAMES, (SHARED_TERMINAL,)),
        "checksums": _checksums(arrays),
        "condition_3_layer_index": INTERMEDIATE_LAYER_INDEX,
        "condition_3_token_coordinate": CONDITION_3_TOKEN_COORDINATE,
        "controlled_property": CONTROLLED_PROPERTY,
        "design_fingerprint": design_fingerprint(),
        "download_seconds": 0.0,
        "final_layer_index": FINAL_LAYER_INDEX,
        "forward_dtype": FORWARD_DTYPE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "hidden_states_tuple_length": EXPECTED_TUPLE_LENGTH,
        "load_seconds": captured["load_seconds"],
        "model_id": MODEL_ID,
        "parent_prompt_fingerprint": prompt_fingerprint(),
        "revision": MODEL_REVISION,
        "slot_words": list(SLOT_WORDS),
        "stored_dtype": STORED_DTYPE,
        "swap_heldout_prompts": captured["swap_held_texts"],
        "swap_train_prompts": captured["swap_train_texts"],
        "torch_num_threads": 1,
        "torch_version": captured["torch_version"],
        "transformers_version": captured["transformers_version"],
        "url": MODEL_URL,
        "varied_terminals": list(VARIED_TERMINALS),
        "weight_sha256": WEIGHT_SHA256,
    }
    _save_npz(FIXTURE_NPZ, arrays)
    FIXTURE_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(FIXTURE_NPZ),
                "c2_train_shape": list(arrays["c2_train_activations"].shape),
                "c2_heldout_shape": list(arrays["c2_heldout_activations"].shape),
                "c2_train_terminals": sorted(set(int(item) for item in arrays["c2_train_last_token_id"])),
                "c4_terminals": sorted(set(int(item) for item in arrays["c4_train_last_token_id"])),
                "swap_terminals": sorted(set(int(item) for item in arrays["swap_train_last_token_id"])),
                "capture_condition_2_forward_seconds": varied_forward,
                "capture_condition_3_slice_seconds": varied_slice_3,
                "capture_condition_4_seconds": condition_4,
                "capture_parent_swap_seconds": parent_swap,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        raise
