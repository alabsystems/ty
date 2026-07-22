---- MODULE Outer ----
\* Outer module that instances Inner with operator substitutions
\* This mimics the EWD998PCal pattern where:
\* - EWD998PCal has variables network, active, color, counter
\* - EWD998 has variables active, color, counter, pending, token
\* - pending and token are substituted with operators

EXTENDS Integers

VARIABLE data  \* Our actual state variable

\* These operators read from our state
GetX == data[1]
GetY == data[2]

\* Instance Inner with operator substitutions
\* x <- GetX means x is replaced by GetX operator
\* y <- GetY means y is replaced by GetY operator
I == INSTANCE Inner WITH x <- GetX, y <- GetY

Init ==
    data = <<0, 0>>

Next ==
    \/ data' = <<data[1] + 1, data[2]>>
    \/ data' = <<data[1], data[2] + 1>>

Spec == Init /\ [][Next]_<<data>>

\* This property checks that the instanced module's spec holds
\* The issue is in UNCHANGED I!vars where I!vars uses substituted operators
InnerSpec == I!Init /\ [][I!Next]_I!vars

\* Invariant to limit state space
Constraint == data[1] < 3 /\ data[2] < 3

====
