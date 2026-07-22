---- MODULE test_tuple_prime ----
EXTENDS Naturals

VARIABLE x, y, result

Init == /\ x = 1 /\ y = 2 /\ result = 0

Next == /\ x < 3
        /\ x' = x + 1
        /\ y' = y + 1
        /\ IF <<x', y'>> = <<2, 3>>
             THEN result' = 100
             ELSE result' = result + 1

Spec == Init /\ [][Next]_<<x, y, result>>
====
