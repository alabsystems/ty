---- MODULE test_setenum_prime ----
EXTENDS Naturals

VARIABLE x, result

Init == /\ x = 1 /\ result = 0

Next == /\ x < 3
        /\ x' = x + 1
        /\ IF {x', x' + 1} = {2, 3}
             THEN result' = 100
             ELSE result' = result + 1

Spec == Init /\ [][Next]_<<x, result>>
====
