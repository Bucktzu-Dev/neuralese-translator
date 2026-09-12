from neuralese.dynamics import SubjectiveSymbolDynamics
from packutil import make_pack


def test_transition_probs_sum_to_one():
    pack = make_pack()
    ssd = SubjectiveSymbolDynamics(pack, beta_s=1.0)
    probs = ssd.transition_probs(prev_class_id=0)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert set(probs) == {0, 1}


def test_higher_beta_is_more_peaked_on_low_energy():
    pack = make_pack()
    wide = SubjectiveSymbolDynamics(pack, beta_s=0.2).transition_probs(0)
    peaked = SubjectiveSymbolDynamics(pack, beta_s=4.0).transition_probs(0)
    # entropy-like: max prob should not fall when beta rises
    assert max(peaked.values()) >= max(wide.values()) - 1e-9
