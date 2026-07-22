---- MODULE ChooseInvariantViolated ----
(*
 * Soundness fixture for native bounded CHOOSE lowering (trust-cg).
 *
 * `NextEven(n)` is a bounded `CHOOSE k \in S : P(k)` over the integer interval
 * domain `0..6`.  An interval lowers through the general materialized-set
 * CHOOSE path (ascending = TLC-normalized slot order), so the native engine
 * MUST pick the same witness as the tree-walking interpreter
 * (`eval_choose_single`) and the bytecode VM (`choose_begin`): the FIRST
 * element of `0..6` (smallest first) that is both even and STRICTLY greater
 * than n.
 *
 * The CHOOSE is the right-hand side of the (single) prime assignment in Next,
 * so it is compiled into the native next-state action and its witness directly
 * determines the reachable state graph:
 *
 *   correct (smallest even > n):  0 -> 2 -> 4 -> 6 -> (self-loop)   {0,2,4,6}
 *   wrong  (e.g. largest match):  0 -> 6 -> (self-loop)             {0,6}
 *
 * Because the predicate is satisfied by MULTIPLE elements (for n = 0 the
 * witnesses are {2, 4, 6}), witness selection is order-sensitive and changes
 * both the reachable states AND the transition/state counts.  This makes the
 * fixture a genuine cross-backend parity probe rather than a tautology.
 *
 * The IF-guard keeps Next a single `x' = <expr>` assignment (so the action is
 * admitted to the native next-state pipeline) while preventing the CHOOSE from
 * being evaluated with no witness once the walk reaches the top (x = 6).
 *
 * The invariant `Below5` is GENUINELY VIOLATED at the reachable state x = 6
 * (6 > 4).  For every earlier reachable state (0, 2, 4) it holds, so the
 * violation appears only once the canonical walk reaches x = 6.
 *)
EXTENDS Integers

VARIABLE x

\* Bounded CHOOSE over an integer interval: smallest even value strictly
\* greater than n, drawn from 0..6.
NextEven(n) == CHOOSE k \in 0..6 : (k % 2 = 0) /\ (k > n)

Init == x = 0

\* Single prime assignment: walk upward through the even numbers via CHOOSE,
\* then self-loop at the top (x = 6) so the CHOOSE always has a witness.
Next == x' = IF x < 6 THEN NextEven(x) ELSE x

Spec == Init /\ [][Next]_x

\* Type correctness: x stays inside the modeled range.
TypeOK == x \in 0..6

\* VIOLATED at the reachable state x = 6 (6 > 4); holds for x in {0, 2, 4}.
Below5 == x <= 4

====
