---- MODULE RangeInvariantViolated ----
(*
 * Soundness fixture for native compilation of the `Range` operator.
 *
 * `Range(f) == { f[x] : x \in DOMAIN f }` (Functions / SequencesExt).
 *
 * The function variable `f` starts with a range that excludes 0, but a
 * reachable Next step assigns a function whose range contains 0. The
 * invariant `RangeExcludesZero == 0 \notin Range(f)` is therefore genuinely
 * VIOLATED in a reachable state. Both the tree-walking interpreter and the
 * bytecode-VM native path must agree on the VIOLATED verdict.
 *)
EXTENDS Naturals, Functions

VARIABLE f

\* Two total functions over the domain 1..2.
Good == [i \in 1..2 |-> i]      \* Range = {1, 2}
Bad  == [i \in 1..2 |-> 0]      \* Range = {0}

Init == f = Good

\* Reachable transition that drives the range to include 0.
Next == \/ f' = Good
        \/ f' = Bad

Spec == Init /\ [][Next]_f

TypeOK == f \in [1..2 -> 0..2]

\* Violated once `f = Bad` is reached, since 0 \in Range(Bad).
RangeExcludesZero == 0 \notin Range(f)

====
