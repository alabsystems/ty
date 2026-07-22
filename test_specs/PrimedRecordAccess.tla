---- MODULE PrimedRecordAccess ----
(* Test case for #183: RecordAccess with primed base in IF condition
   TLC: 5 states, TY: should work after fix *)
EXTENDS Naturals

VARIABLE r, x

Init == /\ r = [a |-> 1, b |-> 2]
        /\ x = 0

(* RecordAccess with prime in IF condition - bounded *)
Next == /\ r.a < 5
        /\ r' = [a |-> r.a + 1, b |-> r.b]
        /\ IF r'.a = 3
             THEN x' = 10
             ELSE x' = x + 1

Spec == Init /\ [][Next]_<<r, x>>
====
