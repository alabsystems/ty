---- MODULE SubSeqNativeFold ----
\* Cross-backend soundness fixture for the native SubSeq trust-ir lowering.
\*
\* The Shrink action drives a genuinely-violated invariant by repeatedly taking
\* SubSeq(s, 2, Len(s)) (drop the head, a length-CHANGING sequence op). It also
\* updates a record field (r.n) so the action is NOT eligible for the direct
\* fast path and MUST route through the trust-ir pipeline that natively lowers
\* SubSeq. The invariant `Len(s) > 1` is reachable-violated once s shrinks to a
\* single element.

EXTENDS Naturals, Sequences

VARIABLES s, r

vars == << s, r >>

Init ==
  /\ s = <<10, 20, 30, 40>>
  /\ r = [n |-> 0]

\* Drop the head via SubSeq while bumping a record counter. The record update is
\* what keeps this action off the all-scalar direct fast path, forcing the
\* compiled backend through the trust-ir SubSeq lowering.
Shrink ==
  /\ Len(s) > 1
  /\ s' = SubSeq(s, 2, Len(s))
  /\ r' = [r EXCEPT !.n = r.n + 1]

Next == Shrink

Spec == Init /\ [][Next]_vars

\* Violated once s has been shrunk down to a single element (length 1), which is
\* a reachable terminal state of the Shrink chain.
InvLenGtOne == Len(s) > 1

====
