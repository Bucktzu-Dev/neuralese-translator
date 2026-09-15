import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack


def test_none_evidence_key_is_not_stringified():
    pack = make_pack()
    obs_id = pack.symbols[0].observation_ids[0]
    digest = pack.evidence[obs_id]
    data = pack.to_dict()
    data["evidence"] = {None: digest, obs_id: digest}
    loaded = pack.from_dict(data)
    assert None in loaded.evidence
    assert "None" not in loaded.evidence
    cert = certify(loaded)
    assert not cert.evidence_valid
    assert not cert.integrity_valid
    assert not cert.passed
    assert any("not a non-blank string" in f for f in cert.failures)
    with pytest.raises(UncertifiedPackError):
        translate_stream(loaded, [0])
