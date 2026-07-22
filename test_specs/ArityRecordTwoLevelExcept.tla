---- MODULE ArityRecordTwoLevelExcept ----
\* Minimal canary for arity-positive RecordExcept and TwoLevelExcept split specialization dispatch.
EXTENDS Integers

CONSTANTS INS, ACC

VARIABLES rec, tab

Init ==
    /\ rec = [i \in INS |-> [a \in ACC |-> [x |-> 0, y |-> 0]]]
    /\ tab = [i \in INS |-> [a \in ACC |-> [bal |-> 0]]]

UpdateRecordField(i, a) ==
    /\ rec' = [rec EXCEPT ![i][a].x = rec[i][a].x + 1]

UpdateNested(i, a) ==
    /\ tab' = [tab EXCEPT ![i][a].bal = tab[i][a].bal + 1]

Next ==
    \/ \E i \in INS, a \in ACC : UpdateRecordField(i, a)
    \/ \E i \in INS, a \in ACC : UpdateNested(i, a)

Spec == Init /\ [][Next]_<<rec, tab>>

====
