---- MODULE SimpleRec ----
EXTENDS Integers

\* Simplest recursive function
Sum(S) ==
  LET F[T \in SUBSET S] ==
    IF T = {} THEN 0
    ELSE LET n == CHOOSE x \in T : TRUE
         IN n + F[T \ {n}]
  IN F[S]

VARIABLE x

\* Just test sum of {0,1}
Init == x = Sum({0, 1})

Next == UNCHANGED x

Spec == Init /\ [][Next]_x

====
