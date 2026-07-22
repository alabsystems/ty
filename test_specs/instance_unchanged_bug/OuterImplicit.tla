---- MODULE OuterImplicit ----
\* Outer module that instances InnerImplicit with implicit substitutions
\* This mimics EWD998PCal where:
\* - INSTANCE without WITH uses implicit substitution
\* - Variables x and y from Inner are implicitly bound to operators x and y here

EXTENDS Integers

VARIABLE data  \* Our actual state variable

\* These operators have same names as Inner's variables - implicit substitution!
x == data[1]
y == data[2]

\* Instance InnerImplicit with implicit substitutions
\* Since x and y are defined here with same names as Inner's variables,
\* they are implicitly substituted
I == INSTANCE InnerImplicit

Init ==
    data = <<0, 0>>

Next ==
    \/ data' = <<data[1] + 1, data[2]>>
    \/ data' = <<data[1], data[2] + 1>>

Spec == Init /\ [][Next]_<<data>>

\* This property checks that the instanced module's spec holds
\* The key is: UNCHANGED I!vars where I!vars = <<x, y>> and x, y are operators
InnerSpec == I!Init /\ [][I!Next]_I!vars

\* Invariant to limit state space
Constraint == data[1] < 3 /\ data[2] < 3

====
