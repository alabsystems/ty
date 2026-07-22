---- MODULE MaxDebug ----
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

\* Test with different set values
VARIABLE x

RecVals == {MaxRec({0}), MaxRec({1}), MaxRec({0,1}), MaxRec({0,1,2})}
NonRecVals == {MaxNonRec({0}), MaxNonRec({1}), MaxNonRec({0,1}), MaxNonRec({0,1,2})}

Init ==
  x \in RecVals \union NonRecVals

Next == UNCHANGED x

Spec == Init /\ [][Next]_x

\* This should be TRUE if both produce same values
RecEqualsNonRec == RecVals = NonRecVals

====
