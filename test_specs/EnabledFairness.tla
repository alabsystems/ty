---------------------------- MODULE EnabledFairness ----------------------------
EXTENDS Naturals
(****************************************************************************)
(* A simple spec to test ENABLED semantics with WF_ fairness constraints.   *)
(* This spec has a counter that can increment until it reaches MAX, then    *)
(* it can reset to 0. The liveness property asserts that under weak         *)
(* fairness on the entire Next relation, the counter will repeatedly cycle. *)
(*                                                                          *)
(* Key behaviors to verify:                                                 *)
(* 1. Inc is ENABLED when x < MAX                                           *)
(* 2. Reset is ENABLED when x = MAX                                         *)
(* 3. WF_vars(Next) ensures the system doesn't stutter indefinitely         *)
(* 4. This tests that TY correctly evaluates ENABLED for disjunctive Next *)
(****************************************************************************)

CONSTANT MAX

VARIABLE x

Init == x = 0

Inc == x < MAX /\ x' = x + 1

Reset == x = MAX /\ x' = 0

Next == Inc \/ Reset

vars == <<x>>

(* The spec with weak fairness on the entire Next relation *)
(* This ensures no action stutters indefinitely when some action is enabled *)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* Invariant: x is always in range *)
TypeOK == x \in 0..MAX

(* Liveness: x will infinitely often be MAX *)
(* With WF_vars(Next), the system will always make progress *)
(* so x will eventually reach MAX and keep cycling *)
InfOftenMax == []<>(x = MAX)

(* Liveness: x will infinitely often be reset to 0 *)
InfOftenZero == []<>(x = 0)

=============================================================================
