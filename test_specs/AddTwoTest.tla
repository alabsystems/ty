---- MODULE AddTwoTest ----
(*
 * Simplified version of AddTwo from tlaplus-examples/LearnProofs
 * Modified to have a bounded state space for model checking.
 * Tests TLAPS, PTL, and Zenon operators.
 *)
EXTENDS Naturals, TLAPS

VARIABLE x

Init == x = 0
Next == x' = (x + 2) % 10  \* Bounded to 0..9

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..9

\* Using TLAPS operators in invariants
\* PTL (Propositional Temporal Logic) and Zenon both return TRUE
InvariantWithTLAPS == TypeOK /\ PTL /\ Zenon

====
