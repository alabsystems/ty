------------------------ MODULE SafeAtDebug ------------------------
\* Debug SafeAt evaluation

EXTENDS Integers, TLC

CONSTANT Value, Acceptor, Quorum, Ballot

VARIABLES votes, maxBal

VotedFor(a, b, v) == <<b, v>> \in votes[a]

DidNotVoteIn(a, b) == \A v \in Value : ~ VotedFor(a, b, v)

SafeAt(b, v) ==
  LET SA[bb \in Ballot] ==
        \/ bb = 0
        \/ \E Q \in Quorum :
             /\ \A a \in Q : maxBal[a] >= bb
             /\ \E c \in -1..(bb-1) :
                  /\ (c # -1) => /\ SA[c]
                                 /\ \A a \in Q :
                                      \A w \in Value :
                                         VotedFor(a, c, w) => (w = v)
                  /\ \A d \in (c+1)..(bb-1), a \in Q : DidNotVoteIn(a, d)
  IN  SA[b]

Init ==
  /\ votes = [a \in Acceptor |-> {}]
  /\ maxBal = [a \in Acceptor |-> -1]

\* Test: Print SafeAt values from initial state
TestInit ==
  /\ Init
  /\ PrintT(<<"SafeAt(0,v1)=", SafeAt(0, "v1")>>)
  /\ PrintT(<<"SafeAt(0,v2)=", SafeAt(0, "v2")>>)
  /\ PrintT(<<"SafeAt(1,v1)=", SafeAt(1, "v1")>>)
  /\ PrintT(<<"SafeAt(1,v2)=", SafeAt(1, "v2")>>)
  /\ PrintT(<<"SafeAt(2,v1)=", SafeAt(2, "v1")>>)
  /\ PrintT(<<"SafeAt(2,v2)=", SafeAt(2, "v2")>>)

Next == UNCHANGED <<votes, maxBal>>

Spec == TestInit /\ [][Next]_<<votes, maxBal>>

=============================================================================
