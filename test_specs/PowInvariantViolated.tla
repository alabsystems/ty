---- MODULE PowInvariantViolated ----
(*
 * Soundness fixture for native compilation of integer exponentiation (`^`).
 *
 * The counter `x` increments from 0 upward. The invariant
 * `PowBelow16 == 2 ^ x < 16` holds for x in {0,1,2,3} (1,2,4,8 < 16) but is
 * genuinely VIOLATED once x = 4 is reached, since 2 ^ 4 = 16, which is not
 * < 16.
 *
 * The exponents stay small (<= 4), so the i64 result fits comfortably and the
 * trust-cg direct-LLVM lowering of `PowInt` computes the SAME value as the
 * tree-walking interpreter and bytecode VM (no overflow trap, no BigInt
 * promotion). Both the interpreter and the native bytecode-VM path must agree
 * on the VIOLATED verdict, and neither may spuriously overflow or diverge.
 *)
EXTENDS Naturals

VARIABLE x

Init == x = 0

\* Bounded counter so the state space is finite; reaches x = 4.
Next == x' = IF x < 4 THEN x + 1 ELSE x

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..4

\* Violated once x = 4 is reached: 2 ^ 4 = 16, not < 16.
PowBelow16 == 2 ^ x < 16

====
