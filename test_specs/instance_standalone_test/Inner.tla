---- MODULE Inner ----
CONSTANT XInit(_)
VARIABLE x

Init == x = 0 /\ XInit(x)
Next == x' = (x + 1) % 3
====
