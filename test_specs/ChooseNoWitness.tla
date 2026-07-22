---- MODULE ChooseNoWitness ----
(*
 * No-witness fixture for native bounded CHOOSE lowering (trust-cg).
 *
 * `CHOOSE k \in 0..2 : k > 5` has NO satisfying element: nothing in the
 * interval 0..2 is greater than 5.  Both the tree-walking interpreter
 * (`eval_choose_single`) and the native general CHOOSE path must raise the
 * same "CHOOSE with no satisfying value" runtime error rather than silently
 * returning some value.  The error is reached in the initial state, so both
 * backends fail closed identically.
 *)
EXTENDS Integers

VARIABLE x

Impossible == CHOOSE k \in 0..2 : k > 5

Init == x = Impossible

Next == UNCHANGED x

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..2

====
