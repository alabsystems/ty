---- MODULE PC_small ----
EXTENDS PaxosCommit, TLC

\* Model values - just 2 acceptors, 2 RMs
CONSTANTS
a1, a2

CONSTANTS  
rm1, rm2

AcceptorSet == {a1, a2}
RMSet == {rm1, rm2}
BallotSet == {0, 1}
MajoritySet == {{a1, a2}}

\* SYMMETRY sets
SYMM == Permutations(AcceptorSet) \union Permutations(RMSet)

====
