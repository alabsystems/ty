---- MODULE IsFiniteSetNativeFold ----
\* Cross-backend soundness fixture for the native IsFiniteSet trust-ir lowering.
\*
\* The Step action gates on IsFiniteSet over a known-FINITE set ({1, 2, 3}). At
\* compile time the trust-ir pipeline classifies that set shape as finite and folds
\* the predicate to the constant boolean TRUE (no VM helper, no allocation). The
\* action also updates a record field (r.n) so it is NOT eligible for the direct
\* all-scalar fast path and MUST route through the trust-ir pipeline that natively
\* lowers IsFiniteSet. The counter `n` advances each step, and the invariant
\* `n < 3` is reachable-violated once n reaches 3 — so every backend must agree
\* on the violation.

EXTENDS Naturals, FiniteSets

VARIABLES n, r

vars == << n, r >>

Init ==
  /\ n = 0
  /\ r = [n |-> 0]

\* Advance the counter only while {1,2,3} is finite (always true). The record
\* update is what keeps this action off the all-scalar direct fast path, forcing
\* the compiled backend through the trust-ir IsFiniteSet lowering.
Step ==
  /\ IsFiniteSet({1, 2, 3})
  /\ n < 3
  /\ n' = n + 1
  /\ r' = [r EXCEPT !.n = r.n + 1]

Next == Step

Spec == Init /\ [][Next]_vars

\* Violated once the counter has advanced to 3, a reachable terminal state of the
\* Step chain (Step's IsFiniteSet guard is constantly TRUE, so n climbs to 3).
InvCounterLtThree == n < 3

====
