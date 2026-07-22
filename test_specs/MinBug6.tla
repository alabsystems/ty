---- MODULE MinBug6 ----
(* Test IsSafe with just cardinality comparison *)
EXTENDS Integers, FiniteSets

CONSTANTS SetA, SetB

All == SetA \cup SetB

VARIABLE loc

Init == loc = All

IsSafe(S) == Cardinality(S \cap SetB) =< Cardinality(S \cap SetA)

Next == \E S \in SUBSET loc :
          /\ Cardinality(S) \in {1, 2}
          /\ IsSafe(loc \ S)
          /\ loc' = loc \ S
====
