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


def test_breakaway_finds_the_dimension_rebels_sit_on():
    # 40 members of one party; the 10 lowest on dim2 vote against the party on roll calls 1-5
    rng = np.random.default_rng(0)
    pos = pd.DataFrame({"dim1": rng.normal(size=40), "dim2": np.r_[rng.normal(-2, 0.3, 10), rng.normal(1, 0.3, 30)]})
    votes = pd.DataFrame(1.0, index=pos.index, columns=range(1, 8))
    votes.iloc[:10, :5] = 0.0
    votes.loc[0, 6] = np.nan          # roll call 6: unanimous among voters -> no dissent, skipped
    b = evaluate.breakaway(votes, pos, ["R"] * 40)
    assert set(b.rollnumber) == {1, 2, 3, 4, 5}
    assert np.allclose(b.dissent, 0.25)
    assert (b.r_dim2 > 0.9).all() and (b.r_dim1.abs() < 0.5).all()


def test_vote_type():
    from ideology import voteview
    assert voteview.vote_type("On Motion to Suspend the Rules and Pass, as Amended") == "suspension"
    assert voteview.vote_type("On Agreeing to the Amendment") == "amendment"
    assert voteview.vote_type("On Agreeing to the Resolution") == "rule"
    assert voteview.vote_type("On Passage") == "passage"
    assert voteview.vote_type(None) == "other"


def test_coverage2d_ignores_rotation_and_scale():
    from ideology import irt
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(60, 2))
    theta = 0.4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    est = 3 * truth @ rot + 1.0                         # same configuration, rotated, scaled, shifted
    draws = est[None] + rng.normal(0, 0.3, size=(400, 60, 2))
    cov = irt.coverage2d(draws, truth, ["g"] * 60)
    assert (cov.loc["g"] > 0.8).all()


def test_anchor_rollcall_is_the_party_line_vote():
    from ideology import irt
    votes = pd.DataFrame({1: [1, 1, 0, 0], 2: [1, 0, 1, 0], 3: [1, 1, 1, 0]}, dtype=float)
    assert irt.anchor_rollcall(votes, ["R", "R", "D", "D"]) == 0
