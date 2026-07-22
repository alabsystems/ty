---- MODULE ParamShadowLeak2 ----
\* Regression test for #86: Operator parameter leaking into sibling conjuncts
\* This version more closely matches OuterInnerBind structure with 4 conjuncts
\* and guards before the primed assignments.

EXTENDS Integers

VARIABLES x, y, enabled

\* Operator with parameter 'm' - this will shadow outer 'm' if bug exists
SetY(m) == y' = m

Init == x = 0 /\ y = 0 /\ enabled = TRUE

\* Structure matches OuterInnerBind:
\*   \E m : guard1 /\ guard2 /\ primed1 /\ primed2(with op call)
Next ==
    \E m \in {100} :
        /\ enabled = TRUE          \* Guard 1
        /\ m > 0                   \* Guard 2 (uses outer m)
        /\ x' = m                  \* Should use outer m = 100
        /\ SetY(999)               \* SetY binds m=999
        /\ enabled' = enabled

\* Invariant: x should never be 999 (that would mean param leakage)
Inv == x # 999

Spec == Init /\ [][Next]_<<x, y, enabled>>

\* Expected: (0,0,T) -> (100,999,T)
\* Bug: (0,0,T) -> (999,999,T) which violates Inv
====
