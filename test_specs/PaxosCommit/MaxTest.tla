---- MODULE MaxTest ----
EXTENDS Integers

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

CONSTANT Values

VARIABLE x

Init == x = 0

Next ==
  \E s \in SUBSET Values :
    /\ s # {}
    /\ x' = MaxRec(s)

Spec == Init /\ [][Next]_x

====
