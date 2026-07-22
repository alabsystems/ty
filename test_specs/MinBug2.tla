---- MODULE MinBug2 ----
(* Test without function - just direct set *)
EXTENDS Integers, FiniteSets

CONSTANTS SetA, SetB

All == SetA \cup SetB

VARIABLE loc

Init == loc = All

IsSafe(S) == \/ S \subseteq SetB
             \/ Cardinality(S \cap SetB) =< Cardinality(S \cap SetA)

(* Same but loc is directly a set, not a function *)
Next == \E S \in SUBSET loc :
          /\ Cardinality(S) \in {1, 2}
          /\ IsSafe(loc \ S)
          /\ loc' = loc \ S
====
