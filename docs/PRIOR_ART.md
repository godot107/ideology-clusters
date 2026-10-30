# Prior art

The field is political science, not Kaggle; the standard methods are well established.
Citation details verified 2026-10-30.

- **NOMINATE / DW-NOMINATE.** Poole, K. T., & Rosenthal, H. (1985). A Spatial Model for Legislative
  Roll Call Analysis. *American Journal of Political Science*, 29(2), 357-384. Book: Poole & Rosenthal
  (1997), *Congress: A Political-Economic History of Roll Call Voting*, Oxford University Press.
  Spatial model of roll-call voting; scores ship with the Voteview data. The consensus finding: one
  dimension (liberal-conservative) explains most voting; a second dimension mattered historically
  (race/civil rights) and much less today.
- **Bayesian IRT ideal points.** Clinton, J., Jackman, S., & Rivers, D. (2004). The Statistical
  Analysis of Roll Call Data. *American Political Science Review*, 98(2), 355-370.
  doi:10.1017/S0003055404001194. Same spatial idea as a Bayesian model, with uncertainty per member.
  (Some sources cite the subtitle "A Unified Approach".)
- **Voteview** (Lewis, Poole, Rosenthal, Boche, Rudkin & Sonnet): data + published polarisation series
  to compare with.
- **Freedom Caucus as a bloc:** Pew Research Center (2015-10-20), "House Freedom Caucus: What is it
  and who's in it?", found its members more conservative on average than other House Republicans on
  DW-NOMINATE dimension 1. Consistent with Ward/GMM picking it out here. Our addition: it also separates
  on dimension 2, which in the 119th House tracks voting yes or no on bipartisan bills
  (scripts/dimension2.py).
- Common in data-science tutorials: k-means or PCA on the vote matrix, then "the parties separate".
  That is the trivial result this project goes past (within-party scoring, caucus validation,
  stability, uncertainty).
