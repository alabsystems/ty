---- MODULE PC_3ballots ----
EXTENDS PaxosCommit, TLC

\* Model values - 2 acceptors, 2 RMs, but 3 ballots (like full config)
CONSTANTS
a1, a2

CONSTANTS
rm1, rm2

AcceptorSet == {a1, a2}
RMSet == {rm1, rm2}
BallotSet == {0, 1, 2}  \* 3 ballots like full config
MajoritySet == {{a1, a2}}

\* SYMMETRY sets
SYMM == Permutations(AcceptorSet) \union Permutations(RMSet)

====
