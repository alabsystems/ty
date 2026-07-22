---- MODULE PC_2ballot ----
EXTENDS PaxosCommit, TLC

\* Model values
CONSTANTS
a1, a2, a3

CONSTANTS
rm1, rm2

\* Acceptor = {a1, a2, a3}
AcceptorSet == {a1, a2, a3}

\* RM = {rm1, rm2}
RMSet == {rm1, rm2}

\* Ballot = {0, 1} - 2 ballots like original benchmark
BallotSet == {0, 1}

\* Majority = {{a1, a2}, {a1, a3}, {a2, a3}}
MajoritySet == {{a1, a2}, {a1, a3}, {a2, a3}}

\* SYMMETRY sets
SYMM == Permutations(AcceptorSet) \union Permutations(RMSet)

====
