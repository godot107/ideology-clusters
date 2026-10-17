# ideology-clusters

**Can you find the political factions in Congress just from how members vote, and which methods
find them best?**

Every recorded vote in the House and Senate is public. Lay them out as a giant table (one row per
member, one column per vote, yes/no in each cell) and patterns appear: members who vote alike sit
close together. This project goes past the usual "run k-means, the two parties separate" tutorial:

- **Positions with error bars:** a Bayesian ideal-point model places every member on a
  liberal-conservative scale and says how sure it is.
- **Clustering that can say "neither":** Gaussian mixtures, HDBSCAN, hierarchical clustering and
  network community detection, compared against k-means.
- **Checked against a known answer:** DW-NOMINATE, political science's standard scores, plus caucus
  membership the votes never saw.

**Status:** scaffolded 2026-10-17. Data download and vote matrix work; the first check is below.
Plan: [docs/PLAN.md](docs/PLAN.md).

## The first finding: why "it separates the parties" proves little

Simple PCA on the 118th House's 1,119 contested votes gives a first dimension that matches
DW-NOMINATE at a correlation of **0.97**. That sounds excellent, but it is mostly just telling
Democrats from Republicans, which any method does. **Within** each party the correlation drops to
0.24 (Democrats) and 0.62 (Republicans). Ordering members inside their own party is the real
test, and every method here is scored on it.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
pytest -q
python -c "from ideology import matrix; print(matrix.summary(matrix.build(118, 'H')))"
```

## Data

Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet.
*Voteview: Congressional Roll-Call Votes Database.* https://voteview.com/. Downloaded at runtime,
not redistributed here.
