----------------------------- MODULE SystemLoop -----------------------------
(* System LOOP as described by Manna & Pneuli on page 423ff *)
(* Tests liveness violation detection: without fairness, stuttering can occur *)

VARIABLES x
vars == <<x>>

Init == x = 0

One == x = 0 /\ x' = 1
Two == x = 1 /\ x' = 2
Three == x = 2 /\ x' = 3
Back == x = 3 /\ x' = 0

Next == One \/ Two \/ Three \/ Back

(* Without fairness - allows stuttering, liveness should FAIL *)
Spec == Init /\ [][Next]_vars

(* With weak fairness - prevents stuttering, liveness should PASS *)
SpecWeakFair == Spec /\ WF_vars(Next)

(* Liveness property: x will infinitely often be 3 *)
Liveness == []<>(x=3)

=============================================================================
