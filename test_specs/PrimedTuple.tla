---- MODULE PrimedTuple ----
(* Test case for #183: Tuple with primed elements in IF condition
   TLC: 3 states, TY: should work after fix *)
EXTENDS Naturals

VARIABLE x, y, result

Init == /\ x = 1 /\ y = 2 /\ result = 0

(* Tuple with primed variables in IF condition *)
Next == /\ x < 3
        /\ x' = x + 1
        /\ y' = y + 1
        /\ IF <<x', y'>> = <<2, 3>>
             THEN result' = 100
             ELSE result' = result + 1

Spec == Init /\ [][Next]_<<x, y, result>>
====
