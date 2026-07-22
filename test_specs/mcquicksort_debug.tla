---- MODULE mcquicksort_debug ----
(* Minimal reproduction of MCQuicksort bug *)
EXTENDS Integers, Sequences

CONSTANT Values

VARIABLE seq

\* TypeOK0: seq is in Seq(Values) - no set difference
TypeOK0 == seq \in Seq(Values)

\* TypeOK1: seq is in Seq(Values) \ {<<>>}
TypeOK1 == seq \in Seq(Values) \ {<<>>}

Init == seq = <<1, 1, 1, 1>>

Next == UNCHANGED seq

Spec == Init /\ [][Next]_seq
====
