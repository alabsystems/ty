---- MODULE MinStutter ----
(* Minimal reproduction for #147 - stuttering machinery *)
EXTENDS Naturals

VARIABLE s, x

top == [top |-> "top"]

Inc == /\ x < 2
       /\ IF s = top
          THEN /\ x' = x + 1
               /\ s' = [id |-> "inc", val |-> 2]
          ELSE /\ s.id = "inc"
               /\ UNCHANGED x
               /\ s' = IF s.val = 1 THEN top ELSE [s EXCEPT !.val = s.val - 1]

Init == x = 0 /\ s = top
Next == Inc
====
