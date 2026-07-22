---- MODULE MaxEquiv ----
EXTENDS Integers, TLC

\* Recursive Maximum - same as original PaxosCommit
MaxRec(S) ==
  LET Max[T \in SUBSET S] ==
        IF T = {} THEN -1
                  ELSE LET n    == CHOOSE n \in T : TRUE
                           rmax == Max[T \ {n}]
                       IN  IF n >= rmax THEN n ELSE rmax
  IN  Max[S]

\* Non-recursive Maximum - same as local test version
MaxNonRec(S) ==
  IF S = {} THEN -1
            ELSE CHOOSE n \in S : \A m \in S : n >= m

\* Test values
TestSet == {0, 1, 2}

\* Invariant: both implementations return same value
MaxEquivalent ==
  \A s \in SUBSET TestSet : MaxRec(s) = MaxNonRec(s)

VARIABLE dummy

Init == dummy = 0

Next == UNCHANGED dummy

Spec == Init /\ [][Next]_dummy

====
