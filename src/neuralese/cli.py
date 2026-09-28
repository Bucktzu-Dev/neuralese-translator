"""CLI: neuralese learn | adapt | translate | unfold | audit | certify | report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from neuralese.activations import load_activation_dump
from neuralese.adapters import (
    load_observations_jsonl,
    load_pack,
    load_stream,
    save_observations_jsonl,
    save_pack,
)
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.encode import ENCODER_CHAR_TRIGRAM, ENCODERS
from neuralese.audit import certify, decodability_report
from neuralese.translator import translate_stream
from neuralese.unfold import recomputed_cluster_residual, unfold_code


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neuralese",
        description="Translate neuralese symbol streams to English and certify decodability.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    learn_p = sub.add_parser("learn", help="reconstruct a symbol alphabet from observations")
    learn_p.add_argument("observations", type=Path)
    learn_p.add_argument("-o", "--output", type=Path, required=True)
    learn_p.add_argument("--n-symbols", type=int, default=8)
    learn_p.add_argument(
        "--encoder",
        choices=list(ENCODERS),
        default=ENCODER_CHAR_TRIGRAM,
        help=(
            "how to embed rows that have text and no vector. "
            "char_trigram is the hashed n-gram default. "
            "word_sentence_svd is a deterministic local TF-IDF SVD of this file, "
            "not a model hidden state. Its width is --n-symbols."
        ),
    )
    learn_p.add_argument("--parent", type=Path, default=None, help="previous SymbolPack for aliases/ΔMDL")
    learn_p.add_argument("--min-cluster-size", type=int, default=2)
    learn_p.add_argument("--tau-residual", type=float, default=0.55)
    learn_p.add_argument("--seed", type=int, default=0)
    learn_p.add_argument(
        "--mdl-exception",
        default=None,
        help="why a description-length increase vs the parent pack is admitted",
    )

    adapt_p = sub.add_parser("adapt", help="convert a hidden-state dump into observation JSONL")
    adapt_p.add_argument("dump", type=Path)
    adapt_p.add_argument("-o", "--output", type=Path, required=True)
    adapt_p.add_argument(
        "--layout",
        choices=["vectors", "tokens", "layers", "hf_layers", "hf_stack"],
        default=None,
    )
    adapt_p.add_argument("--layer", type=int, default=-1)
    adapt_p.add_argument("--pool", choices=["last", "mean"], default="last")

    tr_p = sub.add_parser("translate", help="gloss a code stream using a sealed pack")
    tr_p.add_argument("pack", type=Path)
    tr_p.add_argument("stream", type=Path)

    audit_p = sub.add_parser("audit", help="print an AuditCertificate for a pack")
    audit_p.add_argument("pack", type=Path)
    audit_p.add_argument("--tau-residual", type=float, default=0.55)
    audit_p.add_argument("--allow-unglossed", action="store_true")

    cert_p = sub.add_parser("certify", help="certify a pack; optionally fail closed")
    cert_p.add_argument("pack", type=Path)
    cert_p.add_argument("--fail-on-undecodable", action="store_true")
    cert_p.add_argument("--tau-residual", type=float, default=0.55)
    cert_p.add_argument("--allow-unglossed", action="store_true")

    unfold_p = sub.add_parser("unfold", help="unfold codes back to reservoir observations")
    unfold_p.add_argument("pack", type=Path)
    unfold_p.add_argument(
        "--code",
        type=int,
        action="append",
        dest="codes",
        help="code to unfold; repeat for several. Default: every symbol code",
    )

    report_p = sub.add_parser(
        "report",
        help="print a machine-readable decodability summary for a sealed pack",
    )
    report_p.add_argument("pack", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "learn":
        obs = load_observations_jsonl(args.observations)
        parent = load_pack(args.parent) if args.parent else None
        pack = learn_pack(
            obs,
            config=LearnConfig(
                n_symbols=args.n_symbols,
                min_cluster_size=args.min_cluster_size,
                tau_residual=args.tau_residual,
                seed=args.seed,
                mdl_exception=args.mdl_exception,
                encoder=args.encoder,
            ),
            previous=parent,
        )
        save_pack(pack, args.output)
        print(json.dumps({"pack_id": pack.pack_id, "checksum": pack.checksum, "n_symbols": len(pack.symbols)}, indent=2))
        return 0

    if args.cmd == "adapt":
        rows = load_activation_dump(
            args.dump,
            layout=args.layout,
            layer=args.layer,
            pool=args.pool,
        )
        save_observations_jsonl(rows, args.output)
        print(json.dumps({"n_observations": len(rows), "output": str(args.output)}, indent=2))
        return 0

    if args.cmd == "translate":
        pack = load_pack(args.pack)
        codes = load_stream(args.stream)
        glosses = translate_stream(pack, codes)
        print(json.dumps([g.to_dict() for g in glosses], indent=2))
        return 0

    if args.cmd == "unfold":
        pack = load_pack(args.pack)
        codes = args.codes if args.codes else [symbol.code for symbol in pack.symbols]
        reports = [unfold_code(pack, code).to_dict() for code in codes]
        print(
            json.dumps(
                {
                    "pack_id": pack.pack_id,
                    "pack_checksum": pack.checksum,
                    "reservoir_size": len(pack.observations),
                    "recomputed_cluster_residual": recomputed_cluster_residual(pack),
                    "symbols": reports,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "report":
        pack = load_pack(args.pack)
        print(json.dumps(decodability_report(pack), indent=2, sort_keys=True))
        return 0

    pack = load_pack(args.pack)
    cert = certify(
        pack,
        require_gloss=not args.allow_unglossed,
        tau_residual=args.tau_residual,
    )
    print(json.dumps(cert.to_dict(), indent=2, sort_keys=True))
    if args.cmd == "certify" and args.fail_on_undecodable and not cert.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
