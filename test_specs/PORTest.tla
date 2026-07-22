-------------------------------- MODULE PORTest --------------------------------
(* Test spec for Partial Order Reduction (POR).
 *
 * Two counters that SEMANTICALLY operate independently, but due to TLA+
 * requiring explicit specification of all primed variables (y' = y),
 * static variable-set analysis finds no independent pairs.
 *
 * This demonstrates a fundamental limitation of static POR analysis:
 * TLA+ actions must specify all variables, even unchanged ones, which
 * creates syntactic dependencies that mask semantic independence.
 *
 * Phase 3 (ay commutativity verification) will address this by proving
 * that A(B(s)) = B(A(s)) even when actions share variables.
 *
 * For now, this spec verifies that POR code paths execute correctly,
 * even when no reduction is achieved.
 *
 * Part of #541 - POR Phase 2
 * Author: Andrew Yates <andrewyates.name@gmail.com>
 *)

EXTENDS Naturals

VARIABLE x, y

vars == <<x, y>>

Init ==
    /\ x = 0
    /\ y = 0

(* Increment x only - y not mentioned at all *)
IncX ==
    /\ x < 3
    /\ x' = x + 1
    /\ y' = y

(* Increment y only - x not mentioned at all *)
IncY ==
    /\ y < 3
    /\ x' = x
    /\ y' = y + 1

Next ==
    \/ IncX
    \/ IncY

(* Type invariant - only checks x to avoid visibility issues *)
TypeOK ==
    /\ x \in 0..3
    /\ y \in 0..3

Spec == Init /\ [][Next]_vars

================================================================================
