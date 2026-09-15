import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_explicit_null_aliases_are_not_normalized_to_empty():
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = None
    loaded = pack.from_dict(data)
    assert loaded.aliases is None
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("aliases is not an object" in f for f in cert.failures)
    with pytest.raises(TypeError, match="aliases is not an object"):
        loaded.to_dict()
    omitted = dict(pack.to_dict())
    del omitted["aliases"]
    assert pack.from_dict(omitted).aliases == {}


def test_explicit_null_receipts_are_not_normalized_to_empty():
    pack = make_pack()
    original = pack.checksum
    data = pack.to_dict()
    data["receipts"] = None
    loaded = pack.from_dict(data)
    assert loaded.receipts is None
    assert loaded.compute_checksum() != original
    cert = certify(loaded)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("receipts is not an array" in f for f in cert.failures)
    dumped = loaded.to_dict()
    assert dumped["receipts"] is None
    omitted = dict(pack.to_dict())
    del omitted["receipts"]
    assert pack.from_dict(omitted).receipts == []


def test_alias_resolution_is_linear_and_reuses_shared_suffixes():
    from neuralese.aliases import resolve_alias_table

    class CountingMap(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.lookups = 0

        def __getitem__(self, key):
            self.lookups += 1
            return super().__getitem__(key)

    n = 80
    chain = CountingMap({i: i + 1 for i in range(n)})
    terminals = resolve_alias_table(chain)
    assert terminals[0] == n
    assert terminals[n - 1] == n
    assert chain.lookups <= 2 * n

    shared = CountingMap({0: 2, 1: 2, 2: 4})
    resolved = resolve_alias_table(shared)
    assert resolved == {0: 4, 1: 4, 2: 4}
    assert shared.lookups <= 6

    aliases = {i: i + 1 for i in range(1, n)}
    aliases[n] = 0
    pack = make_pack(aliases={"legacy": aliases})
    cert = certify(pack)
    assert cert.addressable, cert.failures
    live = CountingMap(aliases)
    pack.aliases = {"legacy": live}
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert live.lookups <= 2 * n


def test_alias_cycle_still_fails_addressability():
    pack = make_pack(aliases={"legacy": {7: 8, 8: 7}}, checksum="unsealed")
    pack.seal()
    cert = certify(pack)
    assert cert.addressable is False
    assert cert.passed is False
    assert any("alias map contains a cycle" in f for f in cert.failures)

def test_colliding_numeric_codebook_keys_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["codebook"] = {"1": 1, "01": 1}
    with pytest.raises(ValueError, match="duplicate keys"):
        pack.from_dict(data)
    data["codebook"] = {"1": 0, "01": 1}
    with pytest.raises(ValueError, match="duplicate keys"):
        pack.from_dict(data)
    loaded = pack.from_dict({**pack.to_dict(), "codebook": {"0": 0, "1": 1}})
    assert loaded.codebook == {0: 0, 1: 1}


def test_colliding_numeric_alias_keys_fail_from_dict():
    pack = make_pack()
    data = pack.to_dict()
    data["aliases"] = {"legacy": {"7": 0, "07": 0}}
    with pytest.raises(ValueError, match="duplicate keys"):
        pack.from_dict(data)
    loaded = pack.from_dict({**pack.to_dict(), "aliases": {"legacy": {"7": 0}}})
    assert loaded.aliases == {"legacy": {7: 0}}

def test_numpy_checksum_returns_failed_certificate():
    import numpy as np

    pack = make_pack()
    pack.checksum = np.array([1])
    cert = certify(pack)
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("checksum is not full SHA-256" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(pack, [0])


def test_certify_alias_targets_use_precomputed_symbol_codes():
    aliases = {i: i + 1 for i in range(7, 40)}
    aliases[40] = 0
    pack = make_pack(aliases={"legacy": aliases})
    calls = {"n": 0}
    original = pack.symbol_by_code

    def counting(code):
        calls["n"] += 1
        return original(code)

    pack.symbol_by_code = counting  # type: ignore[method-assign]
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert calls["n"] == 0

def test_historical_alias_stops_at_live_code():
    pack = make_pack(aliases={7: 0, 0: 1})
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert cert.passed, cert.failures
    glosses = translate_stream(pack, [7])
    assert glosses[0].state == "aliased"
    assert glosses[0].resolved_code == 0
    bounce = make_pack(aliases={7: 0, 0: 7})
    assert certify(bounce).addressable
    assert bounce.resolve_code(7) == (0, True)


def test_numpy_decoder_version_and_decision_fail_closed():
    import json

    import numpy as np

    pack = make_pack()
    pack.decoder_version = np.array([1, 2])
    cert = certify(pack)
    assert not cert.integrity_valid
    assert any("decoder_version" in f for f in cert.failures)
    json.dumps(cert.to_dict())
    pack = make_pack()
    pack.decision = np.array(["accept", "reject"])
    cert = certify(pack)
    assert not cert.passed
    assert any("decision" in f for f in cert.failures)
    json.dumps(cert.to_dict())


def test_numpy_checksum_certificate_is_json_serializable():
    import json

    import numpy as np

    pack = make_pack()
    pack.checksum = np.array([1])
    cert = certify(pack)
    assert cert.pack_checksum == ""
    json.dumps(cert.to_dict())


def test_certify_codebook_lookups_use_class_index():
    pack = make_pack()
    calls = {"n": 0}
    original = pack.symbol_by_class

    def counting(class_id):
        calls["n"] += 1
        return original(class_id)

    pack.symbol_by_class = counting  # type: ignore[method-assign]
    cert = certify(pack)
    assert cert.addressable, cert.failures
    assert calls["n"] == 0


def test_numpy_quarantined_and_class_id_fail_closed():
    import numpy as np

    pack = make_pack()
    pack.symbols[0].quarantined = np.array([False, True])
    cert = certify(pack)
    assert not cert.passed
    pack = make_pack()
    pack.symbols[0].class_id = np.array([0, 1])
    assert pack.symbol_by_class(0) is None
    cert = certify(pack)
    assert not cert.integrity_valid


def test_certificate_details_are_json_safe_for_numpy_metrics():
    import json

    import numpy as np

    pack = make_pack()
    pack.pack_id = np.array([1, 2])
    pack.reconstruction_error = np.array([0.1, 0.2])
    pack.mdl_bits = np.array([12.0])
    cert = certify(pack, tau_residual=np.array([0.55]))
    dumped = json.dumps(cert.to_dict())
    assert cert.pack_id == ""
    assert '"reconstruction_error": null' in dumped
    assert '"mdl_bits": null' in dumped
    assert '"tau_residual": null' in dumped


def test_numpy_finalize_step_and_example_digest_fail_closed():
    import json

    import numpy as np

    pack = make_pack()
    pack.receipts = [
        Receipt(
            step=np.array(["finalize"]),
            ok=True,
            timestamp=1.0,
            metadata={"decision": pack.decision},
        )
    ]
    cert = certify(pack)
    assert not cert.passed
    json.dumps(cert.to_dict())
    pack = make_pack(include_private=True)
    pack.symbols[0].examples = ["hello"]
    pack.symbols[0].example_hashes = [np.array([1])]
    cert = certify(pack)
    assert not cert.integrity_valid
    json.dumps(cert.to_dict())
    pack = make_pack()
    pack.receipts = [
        Receipt(
            step="finalize",
            ok=True,
            timestamp=1.0,
            metadata={"decision": np.array(["accept"])},
        )
    ]
    cert = certify(pack)
    assert not cert.admission_valid
    json.dumps(cert.to_dict())
