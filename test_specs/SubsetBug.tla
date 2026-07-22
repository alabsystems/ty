---- MODULE SubsetBug ----
(* Minimal reproduction of MissionariesAndCannibals state count bug *)
EXTENDS Integers, FiniteSets

CONSTANTS Items

VARIABLE loc

TypeOK == loc \in [{"A", "B"} -> SUBSET Items]

Init == loc = [x \in {"A", "B"} |-> IF x = "A" THEN Items ELSE {}]

OtherBank(b) == IF b = "A" THEN "B" ELSE "A"

Move(S, from) ==
    /\ Cardinality(S) \in {1, 2}
    /\ loc' = [x \in {"A", "B"} |->
                IF x = from
                THEN loc[from] \ S
                ELSE loc[OtherBank(from)] \cup S]

Next == \/ \E S \in SUBSET loc["A"] : Move(S, "A")
        \/ \E S \in SUBSET loc["B"] : Move(S, "B")

AllOnB == loc["A"] /= {}
====
