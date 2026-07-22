---- MODULE ExceptBindingTest ----
\* Test EXCEPT with mixed scope bindings (operator param + EXISTS)
\* Pattern from PaxosCommit: EXCEPT path uses outer var, value uses inner var

EXTENDS Integers

CONSTANTS ACC  \* Acceptor set

VARIABLES state, msgs

Init ==
    /\ state = [a \in ACC |-> [mbal |-> 0, bal |-> -1]]
    /\ msgs = {}

\* Operator with param (acc) that uses EXISTS (m) in EXCEPT
UpdateState(acc) ==
    \E m \in msgs :
        /\ m.type = "req"
        \* The critical pattern: path uses acc (outer), value uses m.bal (inner)
        /\ state' = [state EXCEPT ![acc].mbal = m.bal]
        /\ msgs' = msgs \cup {[type |-> "resp", acc |-> acc, bal |-> m.bal]}

AddMsg ==
    /\ msgs' = msgs \cup {[type |-> "req", bal |-> 1]}
    /\ UNCHANGED state

Next ==
    \/ AddMsg
    \/ \E acc \in ACC : UpdateState(acc)

Spec == Init /\ [][Next]_<<state, msgs>>

====
