---- MODULE ThreeLevel ----
\* Test 3-level nested EXCEPT: f[m.a][b].x = v

EXTENDS Integers

CONSTANTS
    A,    \* First dimension
    B     \* Second dimension

VARIABLES
    f,    \* f[a][b] is a record [x |-> Int]
    msgs  \* Set of messages

Init ==
    /\ f = [a \in A |-> [b \in B |-> [x |-> 0]]]
    /\ msgs = {}

\* Add message specifying which f[a][b] to update
AddMsg(a, b) ==
    /\ [a |-> a, b |-> b, newval |-> 1] \notin msgs
    /\ msgs' = msgs \cup {[a |-> a, b |-> b, newval |-> 1]}
    /\ UNCHANGED f

\* Process message - KEY PATTERN: f[m.a][m.b].x = m.newval
ProcessMsg ==
    \E m \in msgs :
        /\ m.newval > f[m.a][m.b].x
        /\ f' = [f EXCEPT ![m.a][m.b].x = m.newval]
        /\ UNCHANGED msgs

Next ==
    \/ \E a \in A, b \in B : AddMsg(a, b)
    \/ ProcessMsg

Spec == Init /\ [][Next]_<<f, msgs>>

====
