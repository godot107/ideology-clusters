# Plan

**Question:** what is the ideological structure of the US Congress, as revealed by how members vote,
and which methods find it best? Beyond "two parties": factions inside each party, the members who
sit between them, and how that has changed since the 1970s.

**Known answer:** DW-NOMINATE (Poole & Rosenthal), the political-science standard, ships with the
data. It's a spatial model of the same votes, so matching it is a *check* (did we recover the
known structure?), not a discovery. Caucus membership (Freedom Caucus, Progressive Caucus, Blue Dog
Coalition, Problem Solvers) is an external label the votes never saw.

## The trap, measured on day one (118th Congress)

| | House | Senate |
|---|---|---|
| \|corr(PCA dim 1, NOMINATE dim 1)\|, all members | 0.966 | 0.953 |
| same, Democrats only | 0.235 | 0.149 |
| same, Republicans only | 0.615 | 0.653 |

Any method separates the two parties; that gives a near-perfect headline correlation for free.
**The real test is ordering members within their own party**, so every method is scored
within-party too. Same for clusters: "found two clusters that match party" is the trivial result.

## Stages

### 1 · Vote matrix + EDA
Legislator × roll call, yea = 1, nay = 0, missing = NaN; drop near-unanimous votes (minority < 2.5%)
and members with < 20 votes (NOMINATE's conventions). EDA: missingness (who misses votes and why:
mid-term arrivals, retirements, campaigns), party-line vote share, agreement matrix heat map.

### 2 · Dimension reduction: from PCA to ideal-point models
1. **PCA** on mean-imputed votes (baseline). Problems: imputing pulls absentees to the centre, and
   votes are yes/no, not continuous.
2. **Logistic PCA / binary matrix factorisation**: models the probability of a yea and simply skips
   missing votes. Small numpy/PyTorch implementation, checked on simulated legislators with known
   positions.
3. **Bayesian IRT ideal points** (Clinton, Jackman & Rivers 2004) in NumPyro: each member has a
   position, each roll call a cut point and direction; posterior gives **uncertainty per member**
   ("we can't tell these 30 moderates apart"). 1-D and 2-D. Ties into the thinking-bayesian project.
4. Score each against NOMINATE: overall and **within-party** correlation, plus a recovery test on
   synthetic Congresses generated from known ideal points.

### 3 · Clustering: more than k-means
On the ideal points (and, for comparison, the raw vote matrix):
- **k-means**: the baseline the backlog item asks to beat; forced, round, equal-ish clusters.
- **Gaussian mixture + BIC**: soft membership ("70% Freedom-Caucus-like") and lets the data pick k.
- **HDBSCAN**: density-based; finds tight factions and labels the in-between members *noise*,
  which is itself the interesting group (the mavericks).
- **Hierarchical (Ward) clustering**: a dendrogram shows factions inside factions.
- **Community detection on the co-voting network** (Leiden/Louvain on "voted the same way" weights):
  no embedding at all, a different lens on the same question.

Validation: party (ARI, NMI) as the trivial check; caucus membership as the real one (do the
Freedom Caucus members land together?); **stability**: re-cluster on bootstrap resamples of roll
calls and measure agreement (ARI), so we don't report factions that vanish with a different set of votes.

### 4 · Polarisation over time
Repeat across Congresses (e.g. 93rd, 1973 to 119th, now): distance between party medians, overlap
(members closer to the other party's median), number of clusters BIC/HDBSCAN find per Congress.
Compare with Voteview's published polarisation series.

### 5 · Docs
README + blog for a general audience, executed notebook per stage, an interactive scatter
(UMAP or ideal points; hover a dot to see the member) as the demo.

## Decisions / constraints
- Data downloaded at runtime from voteview.com, never committed; cite Voteview.
- Names of sitting politicians appear in results: public officials' public votes, reported neutrally.
  No editorial labels ("extreme") beyond the methods' own terms.
- CPU only: the IRT model at ~450 members x ~1,100 roll calls is a few minutes with SVI/NUTS on CPU;
  bigger multi-Congress runs go in a GPU script if needed (see gpu-work-holds).
