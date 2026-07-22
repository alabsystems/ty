---- MODULE EventuallyAlways ----
EXTENDS Integers

VARIABLE x

Init ==
    x = 0

Next ==
    x' \in {0, 1}

Spec ==
    Init /\ [][Next]_x /\ WF_x(Next)

Stable ==
    x = 1

Live ==
    <>[]Stable

====
