---- MODULE RecordInExcept ----
\* Test EXCEPT with record field access in the path index

EXTENDS Integers

CONSTANTS
    A,    \* Set of indices (e.g., {a1, a2})
    B     \* Set of values for inner (e.g., {b1})

VARIABLES
    f,    \* f[a].val - simple function of records
    msgs  \* Set of messages [{a: A, newval: Int}]

Init ==
    /\ f = [a \in A |-> [val |-> 0]]
    /\ msgs = {}

\* Add a message
AddMsg(a, v) ==
    /\ v \in {1, 2}
    /\ [a |-> a, newval |-> v] \notin msgs
    /\ msgs' = msgs \cup {[a |-> a, newval |-> v]}
    /\ UNCHANGED f

\* Process a message - update f[m.a].val to m.newval
\* This is the key pattern: m.a is a record field access in EXCEPT path
ProcessMsg ==
    \E m \in msgs :
        /\ m.newval > f[m.a].val
        /\ f' = [f EXCEPT ![m.a].val = m.newval]
        /\ UNCHANGED msgs

Next ==
    \/ \E a \in A, v \in {1, 2} : AddMsg(a, v)
    \/ ProcessMsg

Spec == Init /\ [][Next]_<<f, msgs>>

====
