# TODO

Built 2026-10-17: all stages 1-4 with results in results/ and three executed notebooks.

## Done 2026-10-30
- [x] Proofread README + notebooks against results/ (fixed NB1 range 0.18-0.82, README R-hat note)
- [x] Dimension 2 = yes/no on bipartisan bills (scripts/dimension2.py, NB2 section 3, README, blog)
- [x] Interactive explorer docs/explorer.html (scripts/build_explorer.py): 119H map + 119H ranges, light/dark, mobile
- [x] IRT: 119H refit converged (2 chains, R-hat 1.02; within-party vs NOMINATE R 0.89 / D 0.75); 2-D IRT on synthetic fixes hard-line coverage 25% -> 93% (irt.fit2d/align, scripts/fit_irt_synthetic.py)
- [x] Verified citations (docs/PRIOR_ART.md, README References); blog.md drafted

## Dropped
- Senate caucus check: the Progressive Caucus has ~1 Senate member (Sanders) and the cached list is House-only; one member can't validate a clustering.

## Open
- [ ] Willie proofreads blog.md, then publish
- [ ] (optional) 2-D Bayesian fit on the real 119th House: error bars for the Freedom Caucus on both dimensions (~1 h CPU)
- [ ] Host docs/explorer.html (GitHub Pages?) and link it from the blog
