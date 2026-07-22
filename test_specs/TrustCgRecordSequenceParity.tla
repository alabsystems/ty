---- MODULE TrustCgRecordSequenceParity ----
\* Small aggregate-state fixture for trust-cg cross-backend state-graph parity.

EXTENDS Naturals, Sequences

VARIABLES rec, seq, x, y

vars == << rec, seq, x, y >>

Init ==
  /\ rec = [a |-> 0, b |-> 1, c |-> 2, d |-> 3]
  /\ seq = <<0, 1, 2>>
  /\ x = 0
  /\ y = 0

IncX ==
  /\ x < 3
  /\ x' = x + 1
  /\ UNCHANGED <<rec, seq, y>>

IncY ==
  /\ y < 3
  /\ y' = y + 1
  /\ UNCHANGED <<rec, seq, x>>

Next ==
  \/ IncX
  \/ IncY

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ rec \in [a: 0..3, b: 1..3, c: 2..2, d: 3..3]
  /\ seq = <<0, 1, 2>>
  /\ x \in 0..3
  /\ y \in 0..3

====
