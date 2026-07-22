---- MODULE TwoSetsBug ----
(* Test case with two sets like MissionariesAndCannibals *)
EXTENDS Integers, FiniteSets

CONSTANTS SetA, SetB

VARIABLE loc, boat

All == SetA \cup SetB

TypeOK ==
    /\ boat \in {"E", "W"}
    /\ loc \in [{"E", "W"} -> SUBSET All]

Init ==
    /\ boat = "E"
    /\ loc = [x \in {"E", "W"} |-> IF x = "E" THEN All ELSE {}]

OtherBank(b) == IF b = "E" THEN "W" ELSE "E"

(* IsSafe like MissionariesAndCannibals *)
IsSafe(S) == \/ S \subseteq SetB
             \/ Cardinality(S \cap SetB) =< Cardinality(S \cap SetA)

Move(S, from) ==
    /\ S \subseteq loc[from]
    /\ Cardinality(S) \in {1, 2}
    /\ IsSafe(loc[from] \ S)
    /\ IsSafe(loc[OtherBank(from)] \cup S)
    /\ boat' = OtherBank(from)
    /\ loc' = [x \in {"E", "W"} |->
                IF x = from THEN loc[from] \ S ELSE loc[OtherBank(from)] \cup S]

Next == \E S \in SUBSET loc[boat] : Move(S, boat)
====
