---- MODULE TestRandomElement ----
\* Test that RandomElement returns varied elements from a set
\* Print the values to verify randomness

EXTENDS Integers, TLC

VARIABLES x, y

\* Pick two random elements from {1, 2, 3, 4, 5}
Init ==
    /\ x = RandomElement({1, 2, 3, 4, 5})
    /\ y = RandomElement({1, 2, 3, 4, 5})
    /\ PrintT(<<"x =", x, "y =", y>>)

Next == UNCHANGED <<x, y>>

====
