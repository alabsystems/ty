---- MODULE shadow_minimal ----
\* Andrew Yates <andrewyates.name@gmail.com>
CONSTANTS a1, a2, v1, v2, Acceptor, Value
VARIABLES votes

MCAcceptor == {a1, a2}
MCValue == {v1, v2}

VotedFor(a, b, v) == <<b, v>> \in votes[a]

\* Test invariant: bound vars must not shadow CONSTANTs (TLC rejects such specs)
\* This tests the OneValuePerBallot property: all votes at same ballot have same value
TestRenamed ==
    \A ax, ay \in Acceptor, b \in {0}, vx, vy \in Value :
        VotedFor(ax, b, vx) /\ VotedFor(ay, b, vy) => (vx = vy)

Init == votes = [a \in Acceptor |-> {}]
Next == \E a \in Acceptor, v \in Value :
          votes' = [votes EXCEPT ![a] = @ \cup {<<0, v>>}]
====
