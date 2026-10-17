"""Known-answer checks for the methods (no downloads)."""
import numpy as np
import pandas as pd

from ideology import caucus, cluster, embed, evaluate, polarization, simulate


def small():
    return simulate.simulate(n_members=120, n_rollcalls=300, seed=1)


def test_simulation_has_the_designed_factions_and_missingness():
    c = small()
    assert set(c.members.faction) == set(simulate.FACTIONS)
    assert 0.02 < c.votes.isna().mean().mean() < 0.15
    share = c.votes.mean()
    assert ((share >= 0.025) & (share <= 0.975)).all()


def test_logistic_pca_recovers_known_positions_better_than_pca():
    c = small()
    right = (c.members.party == "R").to_numpy()
    zl, info = embed.lpca(c.votes, 1, right_mask=right)
    zp = embed.pca(c.votes, 1, right)
    sl = evaluate.position_scores(zl[:, 0], c.members.x1, c.members.party)
    sp = evaluate.position_scores(zp[:, 0], c.members.x1, c.members.party)
    assert info["converged"]
    assert sl["within_D_spearman"] > 0.9 and sl["within_R_spearman"] > 0.9
    assert sl["within_D_spearman"] + sl["within_R_spearman"] >= sp["within_D_spearman"] + sp["within_R_spearman"]
    assert zl[right, 0].mean() > zl[~right, 0].mean()       # oriented: Republicans positive


def test_heldout_beats_party_rule_on_synthetic():
    c = small()
    acc = embed.heldout_accuracy(c.votes, 2, 1.0)["accuracy"]
    base = embed.party_baseline_accuracy(c.votes, c.members.party)
    assert acc > base


def test_agreement_matrix_by_hand():
    v = pd.DataFrame([[1, 0, np.nan], [1, 1, 0], [0, 1, 0]], dtype=float)
    A = cluster.agreement(v)
    assert A[0, 1] == 0.5          # both voted on 2 roll calls, agreed on 1
    assert A[1, 2] == 2 / 3
    assert A[0, 0] == 1.0


def test_caucus_match_perfect_and_scattered():
    party = np.array(["R"] * 6)
    cau = np.array(["FC", "FC", "FC", "none", "none", "none"])
    assert evaluate.caucus_match([0, 0, 0, 1, 1, 1], cau, party, "FC", "R")["f1"] == 1.0
    assert evaluate.caucus_match([0, 1, 2, 0, 1, 2], cau, party, "FC", "R")["f1"] < 0.5


def test_separation_and_overlap_by_hand():
    x = np.array([-2.0, -1.0, 1.0, 2.0, -0.5])
    party = np.array(["Democrat", "Democrat", "Republican", "Republican", "Republican"])
    sep, ov = polarization._sep_overlap(x, party)
    assert ov == 1 / 5            # the Republican at -0.5 is closer to the Democratic median
    assert sep > 0


def test_seat_parsing():
    assert caucus._seat("AZ-5") == ("AZ", 5)
    assert caucus._seat("D-CA-24") == ("CA", 24)
    assert caucus._seat("Nick Begich (AK-AL)") == ("AK", 1)
    assert caucus._seat("no seat here") is None
