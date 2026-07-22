---- MODULE PC_3A_1maj ----
EXTENDS PaxosCommit, TLC

\* 3 acceptors, 2 RMs, 3 ballots, but SINGLE majority
CONSTANTS
a1, a2, a3

CONSTANTS
rm1, rm2

AcceptorSet == {a1, a2, a3}
RMSet == {rm1, rm2}
BallotSet == {0, 1, 2}
MajoritySet == {{a1, a2}}  \* Only ONE majority

\* SYMMETRY sets
SYMM == Permutations(AcceptorSet) \union Permutations(RMSet)

====
