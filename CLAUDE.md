# ideology-clusters

Ideological structure of the US Congress from roll-call votes: dimension reduction (PCA → logistic
PCA → Bayesian IRT ideal points) and clustering beyond k-means (GMM+BIC, HDBSCAN, Ward, co-voting
network communities), checked against DW-NOMINATE and caucus membership. From Trello Project
Checklist #4 item 11. Started 2026-10-17. Portfolio piece; docs plain-language for general readers.
Plan: docs/PLAN.md. Open work: TODO.md. Prior art: docs/PRIOR_ART.md.

## Run

```bash
source .venv/bin/activate        # own venv, CPU only
pytest -q
python -c "from ideology import matrix; print(matrix.summary(matrix.build(118, 'H')))"   # downloads on first use
```

## Decisions

- **Data:** Voteview CSVs (members / votes / rollcalls per Congress+chamber), fetched at runtime by
  `voteview.py` (single source of truth for URLs and vote codes) into data/ (gitignored).
- **Known answer:** DW-NOMINATE dims in the members file. Matching it is a check, not a finding.
- **Score within party, always.** Day-one smoke test (118th): PCA dim 1 vs NOMINATE |r| = 0.97
  overall but 0.24 (D) / 0.62 (R) within party in the House. Party separation is free.
- Matrix filters = NOMINATE conventions: drop roll calls with minority < 2.5%, members < 20 votes.
- Public officials' public votes; report neutrally, no editorial labels.

## Data notes

- Cast codes: 1-3 yea, 4-6 nay, 7-9 present/not voting, 0 not a member (`voteview.YEA/NAY`).
- members file includes the President (chamber "President"): filtered out.
- Party codes: 100 D, 200 R, 328 Independent. 118th Senate has 4 independents/others.
- 118th House: 452 members after filters (special-election replacements included), 1,119 roll calls,
  7% missing. Senate: 105 members, 659 roll calls, 9% missing.
