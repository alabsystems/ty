---- MODULE PrimedDomain ----
(* Test case for #183: Domain with primed function in IF condition
   TLC: works, TY: should work after fix *)
EXTENDS Naturals, FiniteSets

VARIABLE f, result

Init == /\ f = [x \in {1} |-> x]
        /\ result = 0

(* Domain with primed function in IF condition *)
Next == /\ Cardinality(DOMAIN f) < 3
        /\ f' = [x \in DOMAIN f \cup {Cardinality(DOMAIN f) + 1} |-> x]
        /\ IF Cardinality(DOMAIN f') = 2
             THEN result' = 100
             ELSE result' = result + 1

Spec == Init /\ [][Next]_<<f, result>>
====
