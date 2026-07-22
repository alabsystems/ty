---- MODULE OpParamShadow ----
\* Minimal test: operator parameter shadowing in sibling conjuncts
\* This tests that Op(arg)'s parameter doesn't leak into sibling conjuncts

EXTENDS Integers

VARIABLES x, y

\* Operator with parameter 'm' that shadows existentially bound 'm'
UpdateY(m) == y' = m

Init == x = 0 /\ y = 0

\* The bug: when m=1 is existentially bound, Op(m+10) binds m to 11
\* If the x' assignment sees Op's m instead of the outer m, x gets 11 instead of 1
Next ==
    \E m \in {1, 2} :
        /\ x' = m           \* Should use outer m (1 or 2)
        /\ UpdateY(m + 10)  \* Binds parameter m to 11 or 12

\* If bug: x ends up as 11 or 12 (Op's parameter)
\* If correct: x ends up as 1 or 2 (outer existential)

Spec == Init /\ [][Next]_<<x, y>>

\* With correct scoping: reachable states are (0,0), (1,11), (2,12)
\* With bug: x' sees UpdateY's m, so (0,0), (11,11), (12,12) - but 11,12 not in {1,2}!
====
