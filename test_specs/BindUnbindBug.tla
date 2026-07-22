---- MODULE BindUnbindBug ----
\* Minimal reproduction of bind/unbind state undercount bug (#175)
\* Pattern: EXISTS with OR inside - this is what breaks bind/unbind mode

EXTENDS Integers, FiniteSets

VARIABLE x, y

Domain == {1, 2}

\* Simple actions that each modify state differently
ActionA(v) ==
    /\ x' = v
    /\ y' = 0

ActionB(v) ==
    /\ x' = v
    /\ y' = 1

\* Pattern that triggers the bug: EXISTS with OR
\* Expected: 4 distinct states (x \in {1,2}, y \in {0,1})
Next ==
    \E v \in Domain :
        ActionA(v) \/ ActionB(v)

Init == x = 0 /\ y = 0

====
