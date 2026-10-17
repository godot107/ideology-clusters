# Prior art

The field is political science, not Kaggle; the standard methods are well established.

- **NOMINATE / DW-NOMINATE** (Poole & Rosenthal, 1985 onward; *Congress: A Political-Economic History
  of Roll Call Voting*, 1997). Spatial model of roll-call voting; scores ship with the Voteview data.
  The consensus finding: one dimension (liberal-conservative) explains most voting; a second
  dimension mattered historically (race/civil rights) and much less today.
- **Bayesian IRT ideal points** (Clinton, Jackman & Rivers, "The Statistical Analysis of Roll Call
  Data", APSR 2004). Same spatial idea as a Bayesian model, with uncertainty per member.
- **Voteview** (Lewis, Poole, Rosenthal et al.): data + published polarisation series to compare with.
- Common in data-science tutorials: k-means or PCA on the vote matrix, then "the parties separate".
  That is the trivial result this project goes past (within-party scoring, caucus validation,
  stability, uncertainty).

To do: verify citation details before the blog; look for published analyses of the Freedom Caucus
as a voting bloc (to compare against, not copy).
