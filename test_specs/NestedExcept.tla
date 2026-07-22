---- MODULE NestedExcept ----
\* Test nested EXCEPT without record fields

EXTENDS Integers

CONSTANTS
    A,    \* First dimension (e.g., {a1, a2})
    B     \* Second dimension (e.g., {b1, b2})

VARIABLES
    f,    \* f[a][b] - nested function
    msgs  \* Set of (a, b, val) triples

Init ==
    /\ f = [a \in A |-> [b \in B |-> 0]]
    /\ msgs = {}

\* Action: update f[a][b] to some value based on message
Update(a, b) ==
    \E m \in msgs :
        /\ m.a = a
        /\ m.b = b
        /\ m.val > f[a][b]
        /\ f' = [f EXCEPT ![a][b] = m.val]
        /\ msgs' = msgs \cup {[a |-> a, b |-> b, val |-> m.val + 1]}

\* Action: add initial message
AddMsg(a, b) ==
    /\ [a |-> a, b |-> b, val |-> 1] \notin msgs
    /\ msgs' = msgs \cup {[a |-> a, b |-> b, val |-> 1]}
    /\ UNCHANGED f

Next ==
    \/ \E a \in A, b \in B : AddMsg(a, b)
    \/ \E a \in A, b \in B : Update(a, b)

Spec == Init /\ [][Next]_<<f, msgs>>

====
