---- MODULE OperatorReplacementTest ----
(* Test that operator replacement works correctly *)

EXTENDS Naturals

CONSTANT F(_, _)  \* Constant operator with 2 parameters
VARIABLE x

(* The replacement operator *)
MCF(a, b) == a + b

Init == x = 0

(* Use the constant operator - should be replaced by MCF *)
Next == x' = F(x, 1)

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..10
====
