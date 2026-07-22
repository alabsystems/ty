---- MODULE TwoLevelExcept ----
\* Test two-level EXCEPT with mixed scope bindings
\* Pattern: ![inner_var][outer_var].field = inner_var

EXTENDS Integers

CONSTANTS INS, ACC

VARIABLES state, msgs

Init ==
    /\ state = [ins \in INS |-> [acc \in ACC |-> [mbal |-> 0, bal |-> -1]]]
    /\ msgs = {}

Send(msg) == msgs' = msgs \cup {msg}

\* Two-level EXCEPT with:
\* - Path index 1: m.ins (inner EXISTS)
\* - Path index 2: acc (outer param)
\* - Value: m.bal (inner EXISTS)
\* Note: Uses Send() operator like OuterInnerBind
UpdateState(acc) ==
    \E m \in msgs :
        /\ m.type = "req"
        /\ state[m.ins][acc].mbal < m.bal
        /\ state' = [state EXCEPT ![m.ins][acc].mbal = m.bal]
        /\ Send([type |-> "resp",
                 ins  |-> m.ins,
                 mbal |-> m.bal,
                 bal  |-> state[m.ins][acc].bal,
                 acc  |-> acc])

AddMsg(ins) ==
    /\ msgs' = msgs \cup {[type |-> "req", ins |-> ins, bal |-> 1]}
    /\ UNCHANGED state

Next ==
    \/ \E ins \in INS : AddMsg(ins)
    \/ \E acc \in ACC : UpdateState(acc)

Spec == Init /\ [][Next]_<<state, msgs>>

====
