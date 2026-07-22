---- MODULE BindingTest ----
\* Minimal test for mixed scope bindings
\* Pattern: operator param + EXISTS in value expression

EXTENDS Integers

VARIABLES x

Init == x = 0

\* Operator with param that uses EXISTS
Op(a) ==
    \E m \in {1, 2, 3} :
        /\ x' = m + a  \* Both m (inner EXISTS) and a (outer param) used

Next ==
    \E a \in {10, 20} : Op(a)

Spec == Init /\ [][Next]_x

====
