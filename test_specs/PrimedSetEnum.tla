---- MODULE PrimedSetEnum ----
(* Test case for #183: SetEnum with primed elements in IF condition
   TLC: works, TY: should work after fix *)
EXTENDS Naturals

VARIABLE x, result

Init == /\ x = 1 /\ result = 0

(* SetEnum with primed variable in IF condition *)
Next == /\ x < 3
        /\ x' = x + 1
        /\ IF {x', x' + 1} = {2, 3}
             THEN result' = 100
             ELSE result' = result + 1

Spec == Init /\ [][Next]_<<x, result>>
====
