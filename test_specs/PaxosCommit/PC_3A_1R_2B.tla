------------------------ MODULE PC_3A_1R_2B ------------------------
EXTENDS PaxosCommit, TLC

\* Model values - 3 acceptors, 1 RM, 2 ballots (minimal config with multiple majorities)
CONSTANTS
a1, a2, a3

CONSTANTS
rm1

AcceptorSet == {a1, a2, a3}
RMSet == {rm1}
BallotSet == {0, 1}
MajoritySet == {{a1, a2}, {a1, a3}, {a2, a3}}

\* SYMMETRY sets
SYMM == Permutations(AcceptorSet)
=============================================================
