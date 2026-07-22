---- MODULE SubsetQuantifierViolated ----
(*
 * Soundness fixture for native lowering of power-set quantifiers:
 *   \E S \in SUBSET T : P(S)   and   \A S \in SUBSET T : P(S)
 *
 * The trust-ir/trust-cg backend lowers a quantifier whose domain is `SUBSET T`
 * (an exact, statically-known scalar/int set T) by enumerating the 2^|T|
 * submasks of T's element universe rather than falling back to the
 * tree-walking interpreter. This fixture pins the *verdict* parity: the
 * invariant below is genuinely VIOLATED in a reachable state, and every
 * backend (tree-walking interpreter, bytecode VM, and — where LLVM is
 * available — native trust-cg) must agree on that VIOLATED verdict and on
 * the reachable state count.
 *
 * T == {1, 2}, so SUBSET T = { {}, {1}, {2}, {1,2} } (4 subsets).
 *
 * Invariant `NoMemberCovered` asserts that the current value `x` belongs to
 * NONE of the subsets of T:
 *     \A S \in SUBSET T : x \notin S
 * This holds while x = 0 (0 is in no subset of {1,2}) but is FALSE once a
 * reachable Next step sets x = 1, because the subset {1} (and {1,2})
 * contains 1. Equivalently `\E S \in SUBSET T : x \in S` becomes TRUE.
 *
 * Reachable states: x = 0 (Init) and x = 1 (Next). The violation is
 * discovered at x = 1.
 *)
EXTENDS Naturals

VARIABLE x

T == {1, 2}

Init == x = 0

\* Reachable transition: x can stay 0 or move to 1.
Next == \/ x' = 0
        \/ x' = 1

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..2

\* Power-set quantifier invariant. VIOLATED once x = 1 is reached, since
\* {1} \in SUBSET T and 1 \in {1}.
NoMemberCovered == \A S \in SUBSET T : x \notin S

====
