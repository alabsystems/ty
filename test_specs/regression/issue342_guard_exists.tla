--------------------------- MODULE issue342_guard_exists ---------------------------
(* Test for issue #342 - EXISTS as pure guard should disable action when false.
 *
 * This spec tests that `\E x \in S : guard(x)` when used as an action guard
 * properly disables the action when the guard is false for all x.
 *)

EXTENDS Naturals

VARIABLE counter

Init == counter = 0

(* Guard that is TRUE only when counter < 3 *)
GuardExists == \E x \in {1,2,3} : x > counter

(* Action that should only fire when GuardExists is TRUE *)
Increment ==
  /\ GuardExists
  /\ counter' = counter + 1

Next == Increment

(* TypeInvariant - counter should never exceed 3 because GuardExists becomes
 * false when counter = 3 (no x in {1,2,3} satisfies x > 3).
 * If EXISTS guard is not properly checked, counter could go to 4 or beyond. *)
TypeInvariant == counter \in 0..3

Spec == Init /\ [][Next]_counter

=============================================================================
