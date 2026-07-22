---- MODULE args_union_synth ----
\* Synthetic spec matching btree's `args` sum-type var shape:
\*   Init : v = NIL          (scalar model value)
\*   Req1 : v' = <<i>>       (arity-1 tuple, i \in Keys = Int)
\*   Req2 : v' = <<i, m>>    (arity-2 tuple, i \in Keys Int, m \in Vals model-value)
\*   Use  : reads v[1] (and v[2]) under a v # NIL guard, then resets v = NIL.
\* Exercises the writer-analysis TaggedUnion promotion + heterogeneous tuple
\* variant + (when enabled) the native tag-dispatch store/read.
EXTENDS Naturals

CONSTANTS Vals, MaxKey, NIL

Keys == 1..MaxKey

VARIABLES v, seen, phase

Init == /\ v = NIL
        /\ seen = 0
        /\ phase = "ready"

Req1 == /\ phase = "ready"
        /\ \E i \in Keys : v' = <<i>>
        /\ phase' = "one"
        /\ seen' = seen

Req2 == /\ phase = "ready"
        /\ \E i \in Keys, m \in Vals : v' = <<i, m>>
        /\ phase' = "two"
        /\ seen' = seen

UseOne == /\ phase = "one"
          /\ v # NIL
          /\ seen' = seen + v[1]
          /\ v' = NIL
          /\ phase' = "ready"

UseTwo == /\ phase = "two"
          /\ v # NIL
          /\ seen' = seen + v[1]
          /\ v' = NIL
          /\ phase' = "ready"

Next == \/ Req1
        \/ Req2
        \/ UseOne
        \/ UseTwo

Spec == Init /\ [][Next]_<<v, seen, phase>>

\* Keep the state space finite: bound the accumulator.
Bounded == seen <= 100
====
