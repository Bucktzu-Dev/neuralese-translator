"""Local text encoders. Hashed trigrams stay the default."""
from __future__ import annotations

import pytest

from neuralese import __version__
from neuralese.alphabet import LearnConfig, learn_pack
from neuralese.contracts import Observation
from neuralese.encode import (
    ENCODER_CHAR_TRIGRAM,
    ENCODER_WORD_SENTENCE_SVD,
    FUNCTION_WORDS,
    content_tokens,
    word_sentence_svd,
)
from neuralese.adapters import hashed_ngram_vector


def test_version_and_default_encoder_thresholds():
    assert __version__ == "0.6.0"
    config = LearnConfig()
    assert config.encoder == ENCODER_CHAR_TRIGRAM
    assert config.tau_residual == 0.55
    assert config.tau_kappa == 0.35


def test_word_sentence_svd_is_a_pure_function_of_the_texts():
    texts = [
        "The skiff rode the tide beside the wharf.",
        "The kiln held the glaze after the wheel stopped.",
        "The skiff line met the buoy at the wharf.",
    ]
    first = word_sentence_svd(texts, rank=2)
    second = word_sentence_svd(list(texts), rank=2)
    assert first == second
    assert len(first) == 3
    assert all(len(row) == 2 for row in first)
    # Topic ids and observation ids are not arguments. A different label on the
    # same strings cannot change the matrix.
    labeled = learn_pack(
        [
            Observation(observation_id="harbor-1", text=texts[0], metadata={"topic": "harbor"}),
            Observation(observation_id="pottery-1", text=texts[1], metadata={"topic": "pottery"}),
            Observation(observation_id="harbor-2", text=texts[2], metadata={"topic": "harbor"}),
        ],
        config=LearnConfig(n_symbols=2, seed=0, encoder=ENCODER_WORD_SENTENCE_SVD),
    )
    relabeled = learn_pack(
        [
            Observation(observation_id="aa-1", text=texts[0], metadata={"topic": "zzz"}),
            Observation(observation_id="bb-1", text=texts[1], metadata={"topic": "yyy"}),
            Observation(observation_id="cc-1", text=texts[2], metadata={"topic": "xxx"}),
        ],
        config=LearnConfig(n_symbols=2, seed=0, encoder=ENCODER_WORD_SENTENCE_SVD),
    )
    assert [labeled.observations["harbor-1"].embedding, labeled.observations["pottery-1"].embedding, labeled.observations["harbor-2"].embedding] == [
        relabeled.observations["aa-1"].embedding,
        relabeled.observations["bb-1"].embedding,
        relabeled.observations["cc-1"].embedding,
    ]
    assert labeled.metadata["config"]["encoder"] == ENCODER_WORD_SENTENCE_SVD
    assert labeled.metadata["config"]["tau_residual"] == 0.55


def test_existing_embeddings_are_left_in_place():
    placed = [1.0, 0.0, 0.0, 0.0]
    pack = learn_pack(
        [
            Observation(observation_id="a", text="alpha beta", embedding=list(placed)),
            Observation(observation_id="b", text="alpha gamma", embedding=list(placed)),
        ],
        config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0, encoder=ENCODER_WORD_SENTENCE_SVD),
    )
    assert pack.observations["a"].embedding == placed
    assert pack.observations["b"].embedding == placed


def test_char_trigram_default_still_hashes_text_only_rows():
    text = "alpha beta gamma"
    pack = learn_pack(
        [Observation(observation_id="a", text=text)],
        config=LearnConfig(n_symbols=1, min_cluster_size=1, seed=0),
    )
    assert pack.observations["a"].embedding == hashed_ngram_vector(text, dim=32, n=3)
    assert pack.metadata["config"]["encoder"] == ENCODER_CHAR_TRIGRAM


def test_function_words_are_not_topic_names():
    topics = {"harbor", "orchard", "ledger", "bakery", "joinery", "weather", "apiary", "pottery"}
    assert topics.isdisjoint(FUNCTION_WORDS)
    assert content_tokens("The harbor skiff was at the wharf") == ["harbor", "skiff", "wharf"]


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match="unknown encoder"):
        learn_pack(
            [Observation(observation_id="a", text="alpha beta")],
            config=LearnConfig(n_symbols=1, min_cluster_size=1, encoder="logit_lens"),
        )
