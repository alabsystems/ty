---- MODULE RecordExcept ----
\* Test EXCEPT with record field access

EXTENDS Integers

CONSTANTS
    A,    \* Set of indices (e.g., {a1, a2})
    B     \* Set of indices (e.g., {b1, b2})

VARIABLES
    f     \* f[a][b] is a record [x |-> Int, y |-> Int]

Init ==
    f = [a \in A |-> [b \in B |-> [x |-> 0, y |-> 0]]]

\* Simple action: increment x field of f[a][b]
IncrX(a, b) ==
    /\ f[a][b].x < 2
    /\ f' = [f EXCEPT ![a][b].x = f[a][b].x + 1]

\* Simple action: copy x to y
CopyXtoY(a, b) ==
    /\ f[a][b].y < f[a][b].x
    /\ f' = [f EXCEPT ![a][b].y = f[a][b].x]

Next ==
    \/ \E a \in A, b \in B : IncrX(a, b)
    \/ \E a \in A, b \in B : CopyXtoY(a, b)

Spec == Init /\ [][Next]_f

====
