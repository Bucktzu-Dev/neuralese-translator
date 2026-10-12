"""One-time capture of pythia-14m activations.

This script is not part of the test run and not part of CI. It downloads the
pinned checkpoint only when ``--snapshot`` is omitted, checks the file hashes
frozen in ``neuralese.activation_source``, and writes ``activations.npz`` plus
``capture_meta.json``. Tests load that fixture with numpy and do not import
this module.

Run from the repository root:

    PYTHONPATH=src python examples/activation_source/capture_once.py --snapshot /path/to/files
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import sys
import time
import tracemalloc
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from neuralese.activation_source import (
    CONFIG_SHA256,
    EXPECTED_ARCHITECTURE,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_LAYERS,
    EXPECTED_VOCAB,
    FIXTURE_META,
    FIXTURE_NPZ,
    FORWARD_DTYPE,
    GENERATION_CONFIG_SHA256,
    HELDOUT_PROMPTS,
    LAYER_RULE,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_URL,
    SPECIAL_TOKENS_SHA256,
    STORED_DTYPE,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    TOKEN_RULE,
    TRAIN_PROMPTS,
    WEIGHT_BYTES,
    WEIGHT_FILE,
    WEIGHT_SHA256,
    CHECKPOINT_WEIGHT_DTYPE,
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


def _download(revision: str, dest: Path) -> float:
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for name in PINNED_FILES:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}"
        urllib.request.urlretrieve(url, dest / name)
    return time.perf_counter() - start


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
    _verify_weight_dtype(weight)


def _verify_weight_dtype(path: Path) -> None:
    from safetensors import safe_open

    dtypes = set()
    with safe_open(path, framework="numpy") as handle:
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


def _capture(snapshot: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def run_split(prompts: tuple[str, ...]) -> dict:
        activations = []
        logits = []
        token_rows: list[list[int]] = []
        tuple_length = None
        for text in prompts:
            encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"]
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise SystemExit(f"unexpected tokenize shape {tuple(input_ids.shape)}")
            if "attention_mask" in encoded and not bool(torch.all(encoded["attention_mask"] == 1)):
                raise SystemExit("tokenizer padded a single unpadded prompt")
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            if outputs.hidden_states is None or len(outputs.hidden_states) < 1:
                raise SystemExit("model returned no hidden states")
            length = len(outputs.hidden_states)
            if tuple_length is None:
                tuple_length = length
            elif tuple_length != length:
                raise SystemExit("hidden-state tuple length changed between prompts")
            hidden = outputs.hidden_states[-1][0, -1, :]
            logit = outputs.logits[0, -1, :]
            if hidden.shape != (EXPECTED_HIDDEN_SIZE,) or logit.shape != (EXPECTED_VOCAB,):
                raise SystemExit(f"activation shape {tuple(hidden.shape)} logit {tuple(logit.shape)}")
            if hidden.dtype != torch.float32 or logit.dtype != torch.float32:
                raise SystemExit("forward dtype was not float32")
            activations.append(hidden.detach().cpu().numpy())
            logits.append(logit.detach().cpu().numpy())
            token_rows.append([int(token) for token in input_ids[0].tolist()])
        ids, lengths, last = _pad_ids(token_rows)
        return {
            "activations": np.stack(activations).astype(np.float32, copy=False),
            "logits": np.stack(logits).astype(np.float32, copy=False),
            "token_ids": ids,
            "token_lengths": lengths,
            "last_token_id": last,
            "hidden_states_tuple_length": int(tuple_length),
        }

    tracemalloc.start()
    capture_start = time.perf_counter()
    try:
        train = run_split(TRAIN_PROMPTS)
        heldout = run_split(HELDOUT_PROMPTS)
    finally:
        capture_seconds = time.perf_counter() - capture_start
        _current, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    if train["hidden_states_tuple_length"] != heldout["hidden_states_tuple_length"]:
        raise SystemExit("train and held-out hidden-state tuple lengths differ")
    for block in (train, heldout):
        if not np.isfinite(block["activations"]).all() or not np.isfinite(block["logits"]).all():
            raise SystemExit("non-finite activation or logit")
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "train": train,
        "heldout": heldout,
        "capture_seconds": capture_seconds,
        "load_seconds": load_seconds,
        "capture_peak_rss_kb": rss_kb,
        "capture_tracemalloc_peak_bytes": int(peak_trace),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "hidden_states_tuple_length": train["hidden_states_tuple_length"],
        "hidden_state_index": train["hidden_states_tuple_length"] - 1,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture pinned pythia-14m activations once.")
    parser.add_argument("--snapshot", type=Path, help="Directory that already holds the pinned files")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing fixture")
    args = parser.parse_args()
    if FIXTURE_NPZ.exists() and not args.force:
        raise SystemExit(f"{FIXTURE_NPZ} already exists; refusing to recapture without --force")
    download_seconds = 0.0
    if args.snapshot is None:
        snapshot = Path("/tmp/pythia-14m-capture")
        download_seconds = _download(MODEL_REVISION, snapshot)
    else:
        snapshot = args.snapshot
    _verify_snapshot(snapshot)
    captured = _capture(snapshot)
    arrays = {
        "train_activations": np.ascontiguousarray(captured["train"]["activations"], dtype=np.float32),
        "heldout_activations": np.ascontiguousarray(captured["heldout"]["activations"], dtype=np.float32),
        "train_logits": np.ascontiguousarray(captured["train"]["logits"], dtype=np.float32),
        "heldout_logits": np.ascontiguousarray(captured["heldout"]["logits"], dtype=np.float32),
        "train_token_ids": captured["train"]["token_ids"],
        "heldout_token_ids": captured["heldout"]["token_ids"],
        "train_token_lengths": captured["train"]["token_lengths"],
        "heldout_token_lengths": captured["heldout"]["token_lengths"],
        "train_last_token_id": captured["train"]["last_token_id"],
        "heldout_last_token_id": captured["heldout"]["last_token_id"],
    }
    meta = {
        "architecture": EXPECTED_ARCHITECTURE,
        "capture_peak_rss_kb": captured["capture_peak_rss_kb"],
        "capture_seconds": captured["capture_seconds"],
        "capture_tracemalloc_peak_bytes": captured["capture_tracemalloc_peak_bytes"],
        "checkpoint_weight_dtype": CHECKPOINT_WEIGHT_DTYPE,
        "checksums": _checksums(arrays),
        "config_sha256": CONFIG_SHA256,
        "download_seconds": download_seconds,
        "forward_dtype": FORWARD_DTYPE,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "heldout_prompts": list(HELDOUT_PROMPTS),
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "hidden_state_index": captured["hidden_state_index"],
        "hidden_states_tuple_length": captured["hidden_states_tuple_length"],
        "layer_rule": LAYER_RULE,
        "load_seconds": captured["load_seconds"],
        "model_id": MODEL_ID,
        "num_hidden_layers": EXPECTED_LAYERS,
        "prompt_fingerprint": prompt_fingerprint(),
        "revision": MODEL_REVISION,
        "special_tokens_sha256": SPECIAL_TOKENS_SHA256,
        "stored_dtype": STORED_DTYPE,
        "token_rule": TOKEN_RULE,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "torch_version": captured["torch_version"],
        "train_prompts": list(TRAIN_PROMPTS),
        "transformers_version": captured["transformers_version"],
        "url": MODEL_URL,
        "vocab_size": EXPECTED_VOCAB,
        "weight_bytes": WEIGHT_BYTES,
        "weight_file": WEIGHT_FILE,
        "weight_sha256": WEIGHT_SHA256,
    }
    _save_npz(FIXTURE_NPZ, arrays)
    FIXTURE_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(FIXTURE_NPZ),
                "hidden_state_index": meta["hidden_state_index"],
                "hidden_states_tuple_length": meta["hidden_states_tuple_length"],
                "capture_seconds": meta["capture_seconds"],
                "capture_peak_rss_kb": meta["capture_peak_rss_kb"],
                "train_activations_sha256": meta["checksums"]["train_activations"],
                "heldout_activations_sha256": meta["checksums"]["heldout_activations"],
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
