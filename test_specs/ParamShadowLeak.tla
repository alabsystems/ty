---- MODULE ParamShadowLeak ----
\* Regression test for #86: Operator parameter leaking into sibling conjuncts
\*
\* This test demonstrates the bug where an operator's parameter `m` shadows
\* an outer existential's `m` in sibling conjuncts due to work stack passing.
\*
\* Bug pattern:
\*   \E m \in {outer_val} :
\*       /\ x' = m                  \* Should use outer m
\*       /\ Op(different_val)       \* Op has param named 'm'
\*
\* With bug: x' sees Op's m (different_val) instead of outer m (outer_val)
\* Correct: x' sees outer m (outer_val)

EXTENDS Integers

VARIABLES x, y

\* Operator with parameter 'm' - this will shadow outer 'm' if bug exists
SetY(m) == y' = m

Init == x = 0 /\ y = 0

\* KEY: The outer m is 100, but SetY's parameter m will be 999
\* If x' sees SetY's m, we get x=999 (BUG)
\* If x' sees outer m, we get x=100 (CORRECT)
Next ==
    \E m \in {100} :
        /\ x' = m           \* Should be 100 (outer m)
        /\ SetY(999)        \* SetY binds m=999

\* Invariant: x should never be 999 (that would mean param leakage)
Inv == x # 999

Spec == Init /\ [][Next]_<<x, y>>

\* Expected states: (0,0), (100,999)
\* Bug would produce: (0,0), (999,999) - then Inv fails or we miss (100,999)
====
