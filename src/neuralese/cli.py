"""CLI: neuralese learn | translate | audit | certify."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from neuralese.adapters import load_observations_jsonl, load_pack, load_stream, save_pack
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.contracts import CERT_POLICIES, UncertifiedPackError
from neuralese.translator import translate_stream


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
    learn_p.add_argument("--parent", type=Path, default=None, help="previous SymbolPack for aliases/ΔMDL")
    learn_p.add_argument("--min-cluster-size", type=int, default=2)
    learn_p.add_argument("--tau-residual", type=float, default=0.55)
    learn_p.add_argument("--seed", type=int, default=0)
    learn_p.add_argument(
        "--include-private",
        action="store_true",
        help="serialize raw example text (off by default)",
    )

    tr_p = sub.add_parser("translate", help="gloss a code stream using a sealed pack")
    tr_p.add_argument("pack", type=Path)
    tr_p.add_argument("stream", type=Path)
    tr_p.add_argument("--policy", default="default", help="certification policy: default|strict")
    tr_p.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="translate even if the pack fails certification (debug only)",
    )
    tr_p.add_argument(
        "--allow-unglossed",
        action="store_true",
        help="translate packs that record [unglossed]; still requires evidence and admission",
    )
    tr_p.add_argument(
        "--source-pack",
        default=None,
        help="parent/source pack checksum whose alias table should be used",
    )
    tr_p.add_argument(
        "--tau-residual",
        type=float,
        default=0.55,
        help="operator residual gate for certification; not read from pack metadata",
    )

    audit_p = sub.add_parser("audit", help="print an AuditCertificate for a pack")
    audit_p.add_argument("pack", type=Path)
    audit_p.add_argument("--tau-residual", type=float, default=0.55)
    audit_p.add_argument("--allow-unglossed", action="store_true")
    audit_p.add_argument(
        "--policy",
        default="default",
        choices=list(CERT_POLICIES),
    )
    audit_p.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="JSONL observations used to recompute evidence hashes",
    )

    cert_p = sub.add_parser("certify", help="certify a pack; optionally fail closed")
    cert_p.add_argument("pack", type=Path)
    cert_p.add_argument("--fail-on-undecodable", action="store_true")
    cert_p.add_argument("--tau-residual", type=float, default=0.55)
    cert_p.add_argument("--allow-unglossed", action="store_true")
    cert_p.add_argument(
        "--policy",
        default="default",
        choices=list(CERT_POLICIES),
    )
    cert_p.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="JSONL observations used to recompute evidence hashes",
    )

    args = parser.parse_args(argv)

    if args.cmd == "learn":
        try:
            obs = load_observations_jsonl(args.observations)
            parent = load_pack(args.parent) if args.parent else None
            pack = learn_pack(
                obs,
                config=LearnConfig(
                    n_symbols=args.n_symbols,
                    min_cluster_size=args.min_cluster_size,
                    tau_residual=args.tau_residual,
                    seed=args.seed,
                    include_private=args.include_private,
                ),
                previous=parent,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        save_pack(pack, args.output)
        print(
            json.dumps(
                {
                    "pack_id": pack.pack_id,
                    "checksum": pack.checksum,
                    "n_symbols": len(pack.symbols),
                    "decision": pack.decision,
                    "status": pack.metadata.get("status"),
                },
                indent=2,
            )
        )
        return 2 if pack.decision == "reject" else 0

    if args.cmd == "translate":
        try:
            pack = load_pack(args.pack)
            codes = load_stream(args.stream)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        try:
            glosses = translate_stream(
                pack,
                codes,
                policy=args.policy,
                require_certified=not args.allow_uncertified,
                require_gloss=not args.allow_unglossed,
                source_pack_checksum=args.source_pack,
                tau_residual=args.tau_residual,
            )
        except UncertifiedPackError as exc:
            print(
                json.dumps(exc.certificate.to_dict(), indent=2, sort_keys=True),
                file=sys.stderr,
            )
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps([g.to_dict() for g in glosses], indent=2))
        return 0

    try:
        pack = load_pack(args.pack)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        observations = (
            load_observations_jsonl(args.observations) if args.observations else None
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    cert = certify(
        pack,
        require_gloss=not args.allow_unglossed,
        tau_residual=args.tau_residual,
        policy=args.policy,
        observations=observations,
    )
    print(json.dumps(cert.to_dict(), indent=2, sort_keys=True))
    if args.cmd == "certify" and args.fail_on_undecodable and not cert.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
