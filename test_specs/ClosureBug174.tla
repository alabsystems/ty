---- MODULE ClosureBug174 ----
\* Reproduction test for #174: Higher-order operator closure bug with quantified variables
\* When a LET-defined operator captures a quantified variable, and is passed
\* to a higher-order operator, the closure doesn't properly capture the variable.

EXTENDS Integers, FiniteSets

VARIABLE x

\* Simple higher-order operator - applies P to y
ApplyPred(P(_), y) == P(y)

\* Test 1: LET operator with constant - should work
Test1 ==
    LET S == {1, 2, 3}
        InS(y) == y \in S
    IN ApplyPred(InS, 2)  \* Should be TRUE

\* Test 2: LET operator with quantified variable - BUG
\* R is bound by \A, and InR captures R
Test2 ==
    \A R \in {{1, 2, 3}} :
        LET InR(y) == y \in R   \* InR captures R from quantifier
        IN ApplyPred(InR, 2)    \* Should be TRUE (2 \in {1,2,3})

\* Test 3: LET operator with EXISTS-bound variable
Test3 ==
    \E R \in {{1, 2, 3}} :
        LET InR(y) == y \in R
        IN ApplyPred(InR, 2)    \* Should be TRUE

\* All tests should be TRUE
AllPass == Test1 /\ Test2 /\ Test3

Init == x = 0
Next == x' = x

\* Invariant - if any test fails, this will be violated
Inv == AllPass
====
