---- MODULE AppendEmptyNativeFold ----
\* Cross-backend soundness fixture for the native Append trust-ir lowering over a
\* STATICALLY-EMPTY source sequence (the Huang corpus pattern).
\*
\* `inbox` is a function from Procs to sequences that stays all-empty (`<<>>`)
\* in this configuration, so each `inbox[p]` has an inferred capacity-0 sequence
\* shape. `Append(inbox[p], msg)` is therefore unconditionally the single-element
\* sequence `<<msg>>`, where `msg` is a record whose shape disagrees with the
\* empty source's tracked element shape -- exactly the over-conservative refusal
\* the native Append lowering now accepts.
\*
\* The single-element result is stored into a SEPARATE capacity-1 record-sequence
\* variable `out`. `out` is seeded non-empty in Init so its inferred layout has
\* capacity 1 with a record element, letting the single-element Append result be
\* stored with no VM fallback. A record-counter bump keeps the action off the
\* all-scalar direct fast path, forcing the trust-ir Append lowering.

EXTENDS Naturals, Sequences

CONSTANTS P1, P2

Procs == {P1, P2}

VARIABLES inbox, out, r

vars == << inbox, out, r >>

Init ==
  /\ inbox = [p \in Procs |-> <<>>]
  /\ out = << [num |-> 0, den |-> 1] >>
  /\ r = [n |-> 0]

\* Append the record onto an always-empty inbox entry, storing the single-element
\* result into the separate capacity-1 record-sequence `out`.
Fire(p) ==
  /\ Len(out) = 1
  /\ out[1].num = 0
  /\ inbox' = inbox
  /\ out' = Append(inbox[p], [num |-> 1, den |-> 2])
  /\ r' = [r EXCEPT !.n = r.n + 1]

Next == \E p \in Procs : Fire(p)

Spec == Init /\ [][Next]_vars

\* Violated once `out[1].num` becomes 1 (reachable after one Fire step).
InvOutNumZero == out[1].num = 0

====
