-------- MODULE BidirectionalTransitions --------
(* Adapted from TLC baseline test-model for TY TLC-parity testing.
   Tests weak fairness with disjunctive actions and modular arithmetic.
   Original: ~/tlaplus/tlatools/org.lamport.tlatools/test-model/BidirectionalTransitions.tla *)

EXTENDS Naturals
VARIABLE x

-------------------------------------------------
(* Test 1: WF_x(A) with disjunctive Next A \/ B *)
(* x cycles through 0, 1, 2 with modular arithmetic *)

A == x' = (x + 1) % 3
B == x' \in 0..2

Init1 == x = 0
Next1 == A \/ B

Spec1A == Init1 /\ [][Next1]_x /\ WF_x(A)
Prop1A == Spec1A /\ WF_x(A) /\ []<><<A>>_x

-------------------------------------------------
(* Test 2: WF_x(D) with different modular arithmetic *)
(* x cycles through 0, 1, 2, 3 with two competing actions *)

C == x' = (x + 1) % 4
D == IF x = 0 THEN x' = 3 ELSE x' = x - 1

Init2 == x = 0
Next2 == C \/ D

Spec2 == Init2 /\ [][Next2]_x /\ WF_x(D)
Prop2 == Spec2 /\ WF_x(D) /\ []<><<D>>_x

=================================================
