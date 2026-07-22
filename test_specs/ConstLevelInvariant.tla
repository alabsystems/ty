---- MODULE ConstLevelInvariant ----
\* Adapted from TLC test-model/ConstLevelInvariant.tla
\* Tests that constant-level invariants are evaluated correctly

EXTENDS Naturals

CONSTANT C

VARIABLE b

Init ==
    b = FALSE

Next ==
    b' = TRUE

Spec ==
    Init /\ [][Next]_b

\* This invariant only references constant C, not the state variable b
ConstLevelInvariant ==
    C \subseteq Nat

=============================================================================
