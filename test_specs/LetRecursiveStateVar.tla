---- MODULE LetRecursiveStateVar ----
EXTENDS Integers

VARIABLES x

Init == x = 3

\* Operator with LET RECURSIVE referencing state variable x
SumToX ==
    LET RECURSIVE Sum(_)
        Sum(n) == IF n <= 0 THEN 0 ELSE x + Sum(n - 1)
    IN Sum(x)

Next == x' = IF x > 0 THEN x - 1 ELSE x

Inv == SumToX >= 0

Spec == Init /\ [][Next]_x
====
