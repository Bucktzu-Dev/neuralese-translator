from pathlib import Path

import pytest

from neuralese.adapters import load_observations_jsonl
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.audit import certify
from neuralese.contracts import Observation
from neuralese.translator import translate_stream

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy_stream" / "observations.jsonl"


def test_learn_toy_pack_certifies():
    obs = load_observations_jsonl(TOY)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, min_cluster_size=2, seed=0))
    cert = certify(pack)
    assert cert.passed, cert.failures
    assert len(pack.symbols) == 3
    live = [s for s in pack.symbols if not s.quarantined]
    assert live
    assert all(s.observation_ids for s in live)
    assert all(s.definition for s in live)


def test_learn_then_translate_unknown_code():
    obs = load_observations_jsonl(TOY)
    pack = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    glosses = translate_stream(pack, [s.code for s in pack.symbols] + [99])
    assert glosses[-1].state == "unknown"
    assert any(g.state == "ok" for g in glosses)


def test_parent_pack_records_parent_id_and_delta():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    second = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)
    assert second.parent_pack_id == first.pack_id
    finalize = next(r for r in second.receipts if r.step == "finalize")
    assert finalize.delta_mdl_bits is not None


def test_text_only_pack_certifies_against_original_observations():
    obs = [
        Observation(observation_id="t-1", text="hello there friend"),
        Observation(observation_id="t-2", text="hello there pal"),
        Observation(observation_id="t-3", text="audit the trail please"),
        Observation(observation_id="t-4", text="audit receipts stay bound"),
    ]
    pack = learn_pack(obs, config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0))
    cert = certify(pack, observations=obs)
    assert cert.passed, cert.failures
    assert cert.details["observations_checked"] is True


def test_load_observations_rejects_non_object_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_observations_jsonl(path)
    path.write_text("null\n")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_observations_jsonl(path)
    path.write_text('"not-an-object"\n')
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_observations_jsonl(path)


def test_load_observations_rejects_unusable_embedding_type(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"observation_id":"x","embedding":1}\n')
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)


def test_load_observations_rejects_non_string_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"observation_id":"x","text":1}\n')
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)


def test_load_observations_rejects_string_embedding(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"observation_id":"x","embedding":"1"}\n')
    with pytest.raises(ValueError, match="invalid observation record"):
        load_observations_jsonl(path)


def test_learn_rejects_duplicate_observation_ids():
    obs = [
        Observation(observation_id="dup", text="hello there friend"),
        Observation(observation_id="dup", text="hello there pal"),
    ]
    with pytest.raises(ValueError, match="duplicate observation_id"):
        learn_pack(obs, config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0))


def test_public_pack_definition_does_not_copy_raw_observation_text():
    obs = [
        Observation(observation_id="s1", text="99887766 !!!"),
        Observation(observation_id="s2", text="99887766 ???"),
        Observation(observation_id="p1", text="!!! 11223344"),
        Observation(observation_id="p2", text="??? 11223344"),
    ]
    pack = learn_pack(
        obs,
        config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0, include_private=False),
    )
    for symbol in pack.symbols:
        assert symbol.examples == []
        definition = symbol.definition or ""
        assert "99887766" not in definition
        assert "11223344" not in definition
        dumped = symbol.to_dict(include_private=False)
        assert "examples" not in dumped
        if symbol.observation_ids:
            assert symbol.example_hashes
    private = learn_pack(
        obs,
        config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0, include_private=True),
    )
    assert any("99887766" in ex or "11223344" in ex for s in private.symbols for ex in s.examples)


def test_public_llm_gloss_does_not_keep_raw_observation_text():
    from neuralese.gloss import learn_definition

    obs = [
        Observation(observation_id="s1", text="99887766 !!!"),
        Observation(observation_id="s2", text="99887766 ???"),
    ]

    class Echo:
        def generate(self, prompt, max_tokens=80):
            return "keep secret 99887766 !!!"

    gloss = learn_definition(obs, llm_client=Echo(), include_private=False)
    assert "99887766" not in (gloss["definition"] or "")
    assert gloss["examples"] == []


def test_public_single_token_observation_is_not_copied_into_definition():
    from neuralese.gloss import learn_definition

    obs = [Observation(observation_id="s1", text="TOPSECRET1234")]
    gloss = learn_definition(obs, include_private=False)
    definition = (gloss["definition"] or "").lower()
    assert "topsecret1234" not in definition
    assert "topsecret1234" not in gloss["keywords"]
    assert gloss["examples"] == []
    assert gloss["definition"] == "[unglossed]"
    assert gloss["confidence"] == 0.0
    private = learn_definition(obs, include_private=True)
    assert "topsecret1234" in (private["definition"] or "").lower()


def test_private_examples_preserve_observation_whitespace():
    from neuralese.alphabet import LearnConfig, learn_pack
    from neuralese.contracts import example_hash
    from neuralese.gloss import learn_definition

    padded = "  hello there friend  "
    obs = [
        Observation(observation_id="t-1", text=padded),
        Observation(observation_id="t-2", text="hello there pal"),
        Observation(observation_id="t-3", text="audit the trail please"),
        Observation(observation_id="t-4", text="audit receipts stay bound"),
    ]
    gloss = learn_definition(obs, include_private=True)
    assert padded in gloss["examples"]
    assert padded.strip() not in gloss["examples"]
    pack = learn_pack(
        obs,
        config=LearnConfig(n_symbols=2, min_cluster_size=2, seed=0, include_private=True),
    )
    hashed = example_hash(padded)
    stripped = example_hash(padded.strip())
    assert any(padded in symbol.examples for symbol in pack.symbols)
    assert any(hashed in symbol.example_hashes for symbol in pack.symbols)
    assert all(stripped not in symbol.example_hashes for symbol in pack.symbols)


def test_public_llm_echo_of_short_observation_is_discarded():
    from neuralese.gloss import learn_definition

    obs = [
        Observation(observation_id="s1", text="secret"),
        Observation(observation_id="s2", text="hello there friend"),
    ]

    class Echo:
        def generate(self, prompt, max_tokens=80):
            return "this means secret"

    gloss = learn_definition(obs, llm_client=Echo(), include_private=False)
    assert "secret" not in (gloss["definition"] or "").lower()
    assert gloss["examples"] == []
    assert "secret" not in gloss["keywords"]


def test_public_llm_echo_of_observation_prefix_is_discarded():
    from neuralese.gloss import learn_definition

    obs = [
        Observation(observation_id="s1", text="LAUNCHCODE99 is classified"),
        Observation(observation_id="s2", text="hello there friend"),
    ]

    class Echo:
        def generate(self, prompt, max_tokens=80):
            return "keep launchco hidden"

    gloss = learn_definition(obs, llm_client=Echo(), include_private=False)
    definition = (gloss["definition"] or "").lower()
    assert "launchco" not in definition
    assert "launchcode99" not in definition
    assert gloss["examples"] == []


def test_public_llm_echo_with_collapsed_whitespace_is_discarded():
    from neuralese.gloss import learn_definition

    obs = [
        Observation(observation_id="s1", text="alpha   beta"),
        Observation(observation_id="s2", text="hello there friend"),
    ]

    class Echo:
        def generate(self, prompt, max_tokens=80):
            return "this means alpha beta"

    gloss = learn_definition(obs, llm_client=Echo(), include_private=False)
    definition = (gloss["definition"] or "").lower()
    assert "alpha beta" not in definition
    assert gloss["examples"] == []


def test_empty_private_definition_is_unglossed_zero_confidence():
    obs = [
        Observation(observation_id="e1", embedding=[1.0, 0.0, 0.0]),
        Observation(observation_id="e2", embedding=[0.95, 0.05, 0.0]),
        Observation(observation_id="e3", embedding=[0.9, 0.1, 0.0]),
    ]
    pack = learn_pack(
        obs,
        config=LearnConfig(
            n_symbols=1,
            min_cluster_size=2,
            seed=0,
            include_private=True,
        ),
    )
    live = [s for s in pack.symbols if not s.quarantined]
    assert live
    for symbol in live:
        assert symbol.definition == "[unglossed]"
        assert symbol.confidence == 0.0


def test_load_pack_rejects_non_object(tmp_path):
    from neuralese.adapters import load_pack

    path = tmp_path / "pack.json"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="invalid pack"):
        load_pack(path)
    path.write_text("null\n")
    with pytest.raises(ValueError, match="invalid pack"):
        load_pack(path)


def test_load_stream_rejects_unusable_codes(tmp_path):
    from neuralese.adapters import load_stream

    path = tmp_path / "stream.json"
    path.write_text("[null]\n")
    with pytest.raises(ValueError, match="stream codes"):
        load_stream(path)
    path.write_text('{"codes": null}\n')
    with pytest.raises(ValueError, match="stream codes"):
        load_stream(path)
    path.write_text("[1.9]\n")
    with pytest.raises(ValueError, match="stream codes"):
        load_stream(path)
    path.write_text("[true]\n")
    with pytest.raises(ValueError, match="stream codes"):
        load_stream(path)


def test_learn_unsealed_parent_is_rejected():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    first.checksum = ""
    with pytest.raises(ValueError, match="unsealed"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)
    first.checksum = None
    with pytest.raises(ValueError, match="unsealed"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)


def test_learn_mutated_parent_checksum_is_rejected():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    first.symbols[0].proto_embedding = list(first.symbols[0].proto_embedding)
    first.symbols[0].proto_embedding[0] += 1.0
    assert isinstance(first.checksum, str) and len(first.checksum) == 64
    with pytest.raises(ValueError, match="does not match"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)


def test_learn_malformed_unsealed_parent_is_clean_value_error():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    first.checksum = ""
    first.symbols[0].proto_embedding = None
    with pytest.raises(ValueError, match="unsealed"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)


def test_learn_parent_overflowing_proto_is_clean_value_error():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    first.symbols[0].proto_embedding = [10**1000]
    with pytest.raises(ValueError, match="does not match"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)


def test_learn_parent_string_proto_is_clean_value_error():
    obs = load_observations_jsonl(TOY)
    first = learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0))
    first.symbols[0].proto_embedding = [
        str(x) for x in first.symbols[0].proto_embedding
    ]
    first.seal()
    with pytest.raises(ValueError, match="does not match"):
        learn_pack(obs, config=LearnConfig(n_symbols=3, seed=0), previous=first)

