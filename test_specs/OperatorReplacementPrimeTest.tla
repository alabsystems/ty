---- MODULE OperatorReplacementPrimeTest ----
(* Test operator replacement with primed variables - mimics MemoryInterface Send pattern *)

EXTENDS Naturals

\* Constant operator that relates current and next state
CONSTANT F(_, _, _, _)

VARIABLE x, y

\* Replacement operator: sets y' to a tuple of first two args
MCF(a, b, old, new) == new = <<a, b>>

Init == /\ x = 0
        /\ y = <<0, 0>>

\* Use F like Send is used: F(x, 1, y, y') should become y' = <<x, 1>>
Next == /\ x' = x + 1
        /\ F(x, 1, y, y')

Spec == Init /\ [][Next]_<<x, y>>

TypeOK == /\ x \in 0..5
          /\ y \in (0..5) \X (0..5)
====
