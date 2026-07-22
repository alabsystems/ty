---- MODULE FunctionOverrideTest ----
\* Adapted from TLC test-model/Github726.tla
\* Tests @@ (function override) operator with nested records

EXTENDS TLC

VARIABLE x

Init ==
  x = 0

Next ==
  UNCHANGED x

Spec ==
  Init /\ [][Next]_x

-------------------------------------------------

\* Test nested record with @@ override (from Github726)
msgs == [ a |-> [ data |-> 15 ] ]

GetMsg(dst) ==
    [ dst |-> dst ] @@ msgs[dst]

Pair ==
    LET m == GetMsg("a")
    IN <<m, m>>

\* Both elements of the pair should be equal
ASSUME Pair[1] = Pair[2]

\* The merged record should have both fields
ASSUME GetMsg("a") = [ data |-> 15, dst |-> "a" ]
ASSUME GetMsg("a").data = 15
ASSUME GetMsg("a").dst = "a"

-------------------------------------------------

\* Basic @@ override tests

\* Override a single field
ASSUME ([a |-> 1] @@ [a |-> 2]) = [a |-> 1]

\* Merge disjoint records
ASSUME ([a |-> 1] @@ [b |-> 2]) = [a |-> 1, b |-> 2]

\* Left record takes precedence for overlapping keys
ASSUME ([a |-> 1, b |-> 2] @@ [b |-> 3, c |-> 4]) = [a |-> 1, b |-> 2, c |-> 4]

\* Override with nested records
ASSUME ([x |-> [a |-> 1]] @@ [y |-> [b |-> 2]]) = [x |-> [a |-> 1], y |-> [b |-> 2]]

\* Override sequence positions
ASSUME (1 :> "a" @@ <<>>) = <<"a">>
ASSUME (1 :> "a" @@ 2 :> "b") = <<"a", "b">>
ASSUME (2 :> "x" @@ <<"a", "b", "c">>) = <<"a", "x", "c">>

\* Multiple overrides
ASSUME (1 :> "first" @@ 3 :> "third" @@ 2 :> "second") = <<"first", "second", "third">>

==================================
