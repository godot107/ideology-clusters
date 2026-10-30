"""Build the notebooks, then execute them so outputs are embedded:

    .venv/bin/python notebooks/build.py          # all (or: build.py 01)
    for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace $n; done

Needs scripts/run_analysis.py and scripts/fit_irt.py run first (they write results/ and data/irt_*.npy).
Written for a general reader: short code cells importing src/, each after plain-language text.
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """
%matplotlib inline
import os, json, warnings
os.chdir('..')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ideology import plot
plot.style()
pd.set_option('display.width', 160)
"""


def _nb():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731
    return cells, md, code


def nb01():
    c, md, code = _nb()
    md("""
# 1 · Placing every member of Congress on a map, from their votes alone

**The idea:** members who vote alike probably think alike. Lay every recorded vote out as a table,
one row per member, one column per vote, yes = 1, no = 0, and find a position for each member that
explains the pattern. Political scientists call it an *ideal point*.

**The data:** Voteview's record of every House roll call in the 118th Congress (2023-24). We drop
near-unanimous votes (fewer than 2.5% on the losing side: naming post offices tells us nothing) and
members with fewer than 20 votes.

**The known answers:**
1. a *synthetic* Congress where we chose every member's position, so recovery can be measured exactly;
2. **DW-NOMINATE**, political science's standard scores. A useful comparison, but itself a model,
   not the truth.
""")
    code(SETUP + """
from ideology import matrix, embed, evaluate, simulate, polarization, cluster
m = matrix.build(118, 'H')
info = matrix.with_members(m, 118, 'H')
print(matrix.summary(m))
print(info.party.value_counts().to_dict())
print('roll calls where most Democrats and most Republicans voted opposite ways:',
      round(polarization.party_line_share(m[info.party.isin(['Democrat', 'Republican']).values],
                                          info.party[info.party.isin(['Democrat', 'Republican'])].values), 3))
""")
    md("""
## 1 · What the vote table looks like

How often each pair of members voted the same way, members sorted left to right by the position found
below. Two solid blocks: within a party, members agree on most contested votes; across parties, rarely.
""")
    code("""
right = (info.party == 'Republican').to_numpy()
z, _ = embed.lpca(m, 2, right_mask=right)
order = np.argsort(z[:, 0])
A = cluster.agreement(m)[np.ix_(order, order)]
fig, ax = plt.subplots(figsize=(5.5, 4.6))
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
ax.set_title('Share of votes cast the same way'); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('members, left to right'); ax.set_ylabel('members, left to right')
fig.colorbar(im, ax=ax, fraction=0.046); ax.grid(False)
""")
    md("""
## 2 · Which method recovers positions? The synthetic Congress

We generated 440 members in six factions with known positions on two dimensions: left-right, and an
"against the leadership" dimension on which one Republican faction stands apart. Votes are noisy, and
missed votes are not random (chronic absentees, members who served part of the term).

- **PCA** treats yes/no as numbers and fills each missed vote with that vote's average.
- **Logistic PCA** models the *probability* of a yes and simply skips missed votes.

Scores: correlation with the true left-right position across everyone, then the real test, the
rank correlation *within each party*, and how well the full 2-D map is recovered (1.0 = exact).
""")
    code("pd.read_csv('results/synthetic_positions.csv').round(3)")
    md("""
Both separate the parties (overall r ≈ 0.99). Within a party, logistic PCA orders members almost
perfectly and recovers the 2-D map far better. Where PCA goes wrong: absentees.
""")
    code("""
c = simulate.simulate()
R = (c.members.party == 'R').to_numpy()
zp, zl = embed.pca(c.votes, 1, R)[:, 0], embed.lpca(c.votes, 1, right_mask=R)[0][:, 0]
t = c.members.x1.to_numpy()
miss = c.votes.isna().mean(1).to_numpy()
def err(z):  # distance from the truth after putting both on the same scale
    a, b = np.polyfit(z, t, 1); return np.abs(a * z + b - t)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.scatter(miss, err(zp), s=12, color=plot.SERIES[1], label='PCA')
ax.scatter(miss, err(zl), s=12, color=plot.SERIES[0], label='logistic PCA')
ax.set_xlabel('share of votes the member missed'); ax.set_ylabel('position error')
ax.set_title('PCA pulls absentees toward the middle'); ax.legend()
plt.tight_layout()
""")
    md("""
## 3 · Real votes: how many dimensions, and is the model worth it?

No answer key here, so we hide a random 10% of the recorded votes, fit on the rest, and predict the
hidden ones. The rival to beat is trivial: **"vote with your party's majority"**.
""")
    code("""
h = pd.read_csv('results/heldout_118H.csv')
h['setting'] = np.where(h.d == 0, 'party-majority rule', 'logistic PCA, ' + h.d.astype(str) + ' dim, penalty ' + h.lam.astype(str))
h[['setting', 'accuracy', 'log_loss']].sort_values('accuracy', ascending=False).round(4).head(8)
""")
    md("""
The party rule alone gets **92.8%** of hidden votes right. One dimension gets 95.5%, two 96.1%, three
96.3%: one left-right dimension does most of the work, matching decades of political-science findings.
We keep two dimensions (the second matters for clustering) with a light penalty.

## 4 · Against DW-NOMINATE
""")
    code("""
v = pd.read_csv('results/vs_nominate.csv')
v[['congress', 'chamber', 'method', 'dim1_overall_r', 'dim1_within_Democrat_spearman', 'dim1_within_Republican_spearman']].round(2)
""")
    md("""
Across everyone, both methods match NOMINATE at r ≈ 0.95-0.98: that is the parties separating. **Within
a party the agreement is only moderate (0.18-0.82), and neither method consistently wins.** That is not
necessarily our error: DW-NOMINATE estimates each member's position from their *whole career*, and uses a
different model of voting, so within-party orderings legitimately differ. On the synthetic Congress,
where the truth is known, logistic PCA was clearly better.

## 5 · Positions with error bars: a Bayesian ideal-point model

The model (Clinton, Jackman & Rivers, 2004) gives each member a position and each vote a cut line and a
direction, and NumPyro samples the plausible values. The result is a **range** for each member.

First, check it on the synthetic Congress: do the 90% ranges contain the true position 90% of the time?
""")
    code("""
from ideology import irt
xs = np.load('data/sim_irt_x.npy'); s = irt.summarize(xs)
a, b = np.polyfit(c.members.x1, s['mean'], 1); tt = a * c.members.x1.to_numpy() + b
inside = (tt >= s['lo']) & (tt <= s['hi'])
pd.Series(inside, index=c.members.faction).groupby(level=0).mean().round(2).rename('share of true positions inside the 90% range')
""")
    md("""
Close to 90% for most factions, but the **hard-line faction is the exception**: they differ from their
party on the *second* dimension, which this one-dimensional model can't represent, so it is confidently
wrong about them. Error bars are only as honest as the model behind them.

Now the real 118th House, every member ordered by position:
""")
    code("""
x = np.load('data/irt_118H.npy'); s = irt.summarize(x)
diag = json.load(open('results/irt_118H_diagnostics.json')); print('sampler diagnostics:', diag)
o = np.argsort(s['mean'])
fig, ax = plt.subplots(figsize=(12, 3.8))
for p, col in plot.PARTY.items():
    sel = (info.party.to_numpy()[o] == p)
    if not sel.any():
        continue
    idx = np.arange(len(o))[sel]
    ax.vlines(idx, s['lo'][o][sel], s['hi'][o][sel], color=col, linewidth=0.8, alpha=0.6)
    ax.scatter(idx, s['mean'][o][sel], s=4, color=col, label=p)
ax.set_xlabel('members, ordered by estimated position'); ax.set_ylabel('position (90% range)')
ax.set_title('118th House: every member, with uncertainty'); ax.legend(loc='upper left')
plt.tight_layout()
out = info[['bioname', 'party', 'state_abbrev']].assign(position=s['mean'], lo=s['lo'], hi=s['hi'],
        width=s['hi'] - s['lo'], votes_cast=m.notna().sum(1).values, nominate_dim1=info.nominate_dim1)
out.to_csv('results/irt_118H_positions.csv')
""")
    md("""
**Least certain positions:** members with few votes (sworn in late after a special election, or who left
early), and members at the **far edge** of their party: no vote separates them from their neighbours,
so the model knows they are far out but not how far.
""")
    code("out.sort_values('width', ascending=False)[['bioname', 'party', 'votes_cast', 'position', 'width']].head(6).round(2)")
    code("""
print('Bayesian positions vs NOMINATE:', evaluate.position_scores(out.position, out.nominate_dim1, out.party))
""")
    md("""
## What this shows

- Separating the parties is easy and proves little; **ordering members within their party** is the
  real test.
- Logistic PCA beats PCA where the truth is known, mainly by not dragging absentees to the middle.
- One dimension predicts 95.5% of votes; the "vote with your party" rule already gets 92.8%.
- The Bayesian model agrees best with DW-NOMINATE within party (Republicans 0.90, Democrats 0.72,
  against 0.75 / 0.67 for logistic PCA), and adds **honest uncertainty**: wide ranges for members with few votes, and a warning
  that a 1-D model can be confidently wrong about a group defined by a second dimension.
""")
    return c


def nb02():
    c, md, code = _nb()
    md("""
# 2 · Finding factions: clustering beyond k-means

**The question:** inside each party, are there distinct groups? We compare five methods, each choosing
its own number of groups by its standard rule (none is told the answer):

| Method | Idea | Picks the number of groups by |
|---|---|---|
| k-means | round clusters around centres | silhouette score |
| Gaussian mixture | elliptical clusters, *soft* membership ("70% this group") | BIC (fit vs complexity) |
| HDBSCAN | dense regions; members between them are "noise" | automatic |
| Ward (hierarchical) | merge closest groups step by step | silhouette score |
| Network communities | link each member to their 15 most-agreeing colleagues, find communities | automatic (Louvain) |

All except the network method cluster the 2-D logistic-PCA positions from notebook 1.
""")
    code(SETUP + "\nfrom ideology import cluster, embed")
    md("""
## 1 · Answer key: the synthetic Congress

Six factions: in each party a wing, a mainstream and moderates that differ by *degree* along left-right,
plus a Republican hard-line faction that differs in *kind* (the second dimension).
*ARI* compares a clustering with the true factions (0 = chance, 1 = identical).
""")
    code("pd.read_csv('results/synthetic_clusters.csv').round(2)")
    md("""
- k-means, the mixture and Ward each find **3 groups: Democrats, Republicans, and the hard-liners**.
  Factions that differ only by degree blend into a continuum: **there is no gap to find**, and forcing
  k-means to make 6 groups (told the right answer) scores *worse*, because it cuts the continuum at
  arbitrary points.
- HDBSCAN chains the two parties together through their moderates.
- Network communities come closest to the six factions.

## 2 · The real House: the 119th Congress (2025-)

To check clusters against something the votes never saw, we use **caucus membership** from Wikipedia's
current member lists: the Freedom Caucus (which doesn't publish its roster; Wikipedia lists publicly
identified members), the Progressive Caucus, the Blue Dog Coalition and the bipartisan Problem Solvers.
""")
    code("""
mem = pd.read_csv('results/members_119H.csv')
rep = json.load(open('results/caucus_match_119H.json'))
{k: f"{v['matched']} of {v['listed']} matched (list retrieved {v['retrieved']})" for k, v in rep.items()}
""")
    code("""
fig, ax = plt.subplots(figsize=(8, 5.2))
lab = mem.gmm
for k, g in mem.groupby('gmm'):
    ax.scatter(g.dim1, g.dim2, s=14, color=plot.SERIES[k], label=f'mixture group {k}', alpha=0.8)
for name, marker in (('Freedom Caucus', 'x'), ('Progressive Caucus', '+')):
    g = mem[mem.caucus.str.contains(name)]
    ax.scatter(g.dim1, g.dim2, s=36, marker=marker, color=plot.INK, linewidths=1, label=name)
ax.set_xlabel('dimension 1 (left - right)'); ax.set_ylabel('dimension 2')
ax.set_title('119th House: Gaussian-mixture groups, with two caucuses marked')
ax.legend(fontsize=8, ncol=2, loc='lower left')
plt.tight_layout()
""")
    md("""
**How well each method captures each caucus** (best-matching cluster's F1 within the caucus's party:
1.0 = a cluster containing exactly that caucus), **and how stable it is**: we resample the roll calls
20 times, redo everything and compare with the original clustering (median ARI; 1.0 = always the same).
""")
    code("pd.read_csv('results/clusters_119H.csv').round(2)")
    md("""
**The trade-off:** the methods that find the most faction structure are the least stable.

- **HDBSCAN** and **k-means** are very stable, but mostly return the two parties.
- **Ward** captures the Freedom Caucus best (F1 0.70).
- **Network communities** isolate most of the Progressive Caucus (F1 0.82), but change most between
  resamples.
- The **Gaussian mixture** is the most even across caucuses, and its *soft* membership says which
  members sit between groups:
""")
    code("""
bic = pd.read_csv('results/gmm_bic_119H.csv')
print('mixture groups chosen by BIC:', int(bic.loc[bic.bic.idxmin(), 'k']))
unsure = mem[mem.gmm_max_membership < 0.8]
print(f'{len(unsure)} members have no group above 80% membership; by party:', unsure.party.value_counts().to_dict())
""")
    md("""
## 3 · What is dimension 2?

The map's second axis has no label of its own. To name it, we look at the votes: for every roll call
where at least 5% of a party broke with that party's majority, how strongly does "voted with my party"
go with a member's position on each dimension? (`scripts/dimension2.py`; positive = members high on
the dimension stay with the party.)

*Suspension* bills are brought up under a fast-track rule: no amendments, two-thirds needed to pass.
In practice they're the bipartisan bills the majority leadership schedules.
""")
    code("pd.read_csv('results/dim2_summary_119H.csv').round(2)")
    md("""
Among **Republicans**, left-right (dim 1) barely says who breaks with the party (0.07 across all such
votes). Dimension 2 does (0.36), and most clearly on suspension bills (0.61): members low on it vote
*no* on bipartisan bills most Republicans support.

Among **Democrats**, the same members who are more moderate on dim 1 also break away on dim 2, crossing
over to vote for Republican-led bills. The two parties look opposite in the table, but one reading fits
both: dimension 2 tracks **how often a member votes yes on bills that both parties' majorities support**.
""")
    code("""
d2 = json.load(open('results/dim2_119H.json'))
print(f"votes where both parties' majorities voted yes: {d2['bipartisan_votes_both_majorities_yes']}")
print('dimension 2 vs yes-rate on those votes (rank correlation): '
      f"Democrats {d2['democrats_dim2_vs_bipartisan_yes_spearman']:.2f}, "
      f"Republicans {d2['republicans_dim2_vs_bipartisan_yes_spearman']:.2f}")
print(f"Freedom Caucus: yes on {d2['freedom_caucus_median_bipartisan_yes']:.0%} of them (median member), "
      f"other Republicans {d2['freedom_caucus_rest_of_party_median_bipartisan_yes']:.0%}")
print(f"Progressive Caucus: yes on {d2['progressive_caucus_median_bipartisan_yes']:.0%}, "
      f"other Democrats {d2['progressive_caucus_rest_of_party_median_bipartisan_yes']:.0%}")
""")
    md("""
That is why the Freedom Caucus sits at the bottom of the map: its median member voted yes on about
half of the bipartisan bills, where other Republicans voted yes on nearly all of them. The Progressive
Caucus sits lower than other Democrats too, but far less so.

Two cautions. The positions were estimated from these same votes, so this *describes* the dimension
rather than testing it; the caucus comparison is the independent part, since caucus lists were never
used. And Voteview carries no text for the amendment votes, so we can say *how* members split on
them, not what the amendments proposed.

## What this shows

- **Groups that differ in kind** (voting against their own leadership) show up as clusters; groups that
  differ only by degree form a continuum, and any line through it is a choice, not a discovery.
- In today's House that second kind of difference is real: dimension 2 tracks voting yes or no on
  bipartisan bills, and it is what sets the Freedom Caucus apart from other Republicans.
- No single best method: report clusters **with** their stability, and prefer soft membership when
  members sit between groups.
- Party-level clustering is the trivial result. Caucus membership, which the votes never saw, is
  the meaningful check.
""")
    return c


def nb03():
    c, md, code = _nb()
    md("""
# 3 · Polarisation, 1973 to now

Each Congress from the 93rd (1973) to the 119th is fitted separately, so positions from different
Congresses are on different scales. We therefore compare only **scale-free** measures:

- **Separation:** the gap between party medians, measured in "typical party widths".
- **Overlap:** the share of members closer to the *other* party's median than their own.
- **Party-line votes:** the share of contested roll calls where most Democrats and most Republicans
  voted opposite ways, straight from the raw votes, no model.

The same separation and overlap from DW-NOMINATE serve as the known answer.
""")
    code(SETUP + "\np = pd.read_csv('results/polarization.csv')\np.groupby('chamber')[['separation', 'nominate_separation']].corr().round(2)")
    code("""
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for j, (ch, name) in enumerate((('H', 'House'), ('S', 'Senate'))):
    d = p[p.chamber == ch]
    ax = axes[0, j]
    ax.plot(d.start_year, d.separation, color=plot.SERIES[0], label='logistic PCA')
    ax.plot(d.start_year, d.nominate_separation, color=plot.SERIES[1], label='DW-NOMINATE')
    ax.set_title(f'{name}: separation (party widths apart)'); ax.legend()
    ax = axes[1, j]
    ax.plot(d.start_year, d.overlap * 100, color=plot.SERIES[0], label='logistic PCA')
    ax.plot(d.start_year, d.nominate_overlap * 100, color=plot.SERIES[1], label='DW-NOMINATE')
    ax.set_title(f'{name}: % members closer to the other party')
    ax = axes[2, j]
    ax.plot(d.start_year, d.party_line_share * 100, color=plot.SERIES[2])
    ax.set_title(f'{name}: % of contested votes along party lines')
for ax in axes[2]: ax.set_xlabel('Congress start year')
plt.tight_layout()
""")
    code("p[p.congress.isin([93, 103, 113, 119])][['chamber', 'start_year', 'separation', 'overlap', 'nominate_separation', 'nominate_overlap', 'party_line_share']].round(2)")
    md("""
## What this shows

- All three measures, from two different models and from the raw votes, tell the same story: the
  parties have moved apart steadily since the 1970s.
- Overlap has vanished: since the early 2000s essentially no member is closer to the other party.
  How much overlap there was in the 1970s depends on the model: about 1 member in 5 by our
  one-dimensional fit, about 1 in 15-20 by DW-NOMINATE. That era's cross-party votes (notably
  Southern Democrats) ran partly along a second dimension, which a 1-D fit squeezes onto left-right.
- The two models' separation series track each other closely (correlation 0.98 House, 0.89 Senate).
- Party-line voting is noisier year to year (it depends on which bills reach a vote) but trends the same way.
- Absolute separation numbers differ between our model and DW-NOMINATE (different models, different
  scales); the *trend* is what agrees, and it is the trend that is reported.
""")
    return c


NOTEBOOKS = {"01": ("01_positions", nb01), "02": ("02_clusters", nb02), "03": ("03_polarization", nb03)}

if __name__ == "__main__":
    for key in sys.argv[1:] or NOTEBOOKS:
        name, fn = NOTEBOOKS[key]
        nb = nbf.v4.new_notebook(cells=fn())
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, HERE / f"{name}.ipynb")
        print(HERE / f"{name}.ipynb")
