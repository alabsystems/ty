------------------------ MODULE PC_3A_2B ------------------------
EXTENDS PaxosCommit, TLC

\* Model values - 3 acceptors, 2 RMs, 2 ballots (same as original)
CONSTANTS
a1, a2, a3

CONSTANTS
rm1, rm2

AcceptorSet == {a1, a2, a3}
RMSet == {rm1, rm2}
BallotSet == {0, 1}
MajoritySet == {{a1, a2}, {a1, a3}, {a2, a3}}

\* SYMMETRY sets
SYMM == Permutations(AcceptorSet) \union Permutations(RMSet)
=============================================================
