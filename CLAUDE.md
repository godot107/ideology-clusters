# ideology-clusters

Ideological structure of the US Congress from roll-call votes: dimension reduction (PCA → logistic
PCA → Bayesian IRT ideal points) and clustering beyond k-means (GMM+BIC, HDBSCAN, Ward, co-voting
network communities), checked against DW-NOMINATE and caucus membership. From Trello Project
Checklist #4 item 11, Trello #103 https://trello.com/c/2o4lbama. Started 2026-10-17. Portfolio piece; docs plain-language for general readers.
Plan: docs/PLAN.md. Open work: TODO.md. Prior art: docs/PRIOR_ART.md.

## Run

```bash
source .venv/bin/activate        # own venv, CPU only (jax cpu + numpyro)
pytest -q
python scripts/run_analysis.py   # all non-Bayesian results -> results/ (~10 min; downloads 54 Voteview files first time)
python scripts/fit_irt.py 118S 118H 119H   # Bayesian ideal points -> data/irt_*.npy (~5-15 min each)
python notebooks/build.py && for n in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $n; done
```

Modules: voteview (download + codes), matrix (vote matrix + filters), simulate (synthetic Congress =
answer key), embed (PCA, logistic PCA, held-out CV, party baseline), irt (NumPyro), cluster (5 methods +
bootstrap stability), caucus (Wikipedia lists -> members), evaluate, polarization, plot.

## Decisions

- **Data:** Voteview CSVs (members / votes / rollcalls per Congress+chamber), fetched at runtime by
  `voteview.py` (single source of truth for URLs and vote codes) into data/ (gitignored).
- **Known answer:** DW-NOMINATE dims in the members file. Matching it is a check, not a finding.
- **Score within party, always** (Spearman). 118th House: PCA dim 1 vs NOMINATE |r| = 0.97 overall,
  but 0.52 (D) / 0.82 (R) within party. Party separation is free.
- **DW-NOMINATE is a comparison, not ground truth within party** (it pools whole careers). The synthetic
  Congress (`simulate.py`) is the real answer key.
- Logistic PCA settings chosen by held-out votes (hide 10%), never against NOMINATE.
- Caucus labels: 119th House only (Wikipedia lists are current), matched by state+district then surname.
- Matrix filters = NOMINATE conventions: drop roll calls with minority < 2.5%, members < 20 votes.
- Public officials' public votes; report neutrally, no editorial labels.

## Gotchas

- **Voteview votes files include the President's announced positions as a "member"** (icpsr 99913 in
  the 118th). `matrix.build` keeps only icpsr in that chamber's members file. Found 2026-10-17; IRT and
  CV were re-run after the fix.
- `irt.fit` returns read-only JAX arrays unless copied (`np.array`); diagnostics are wrapped so a
  finished 5-15 min run is never lost to a summary error.
- Don't register a user-level Jupyter kernel named python3: it hijacks every project's notebooks.
  Each venv's own jupyter provides python3.
- Never pkill/pgrep -f a pattern that appears in your own command line; kill by PID.
- NumPyro 1 chain: split R-hat; min ESS is low for a few members (absentees). Report it.

## Data notes

- Cast codes: 1-3 yea, 4-6 nay, 7-9 present/not voting, 0 not a member (`voteview.YEA/NAY`).
- members file includes the President (chamber "President"): filtered out.
- Party codes: 100 D, 200 R, 328 Independent. 118th Senate has 4 independents/others.
- 118th House: 452 members after filters (special-election replacements included), 1,119 roll calls,
  7% missing. Senate: 105 members, 659 roll calls, 9% missing.
