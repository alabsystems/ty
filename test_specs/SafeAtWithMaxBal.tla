------------------------ MODULE SafeAtWithMaxBal ------------------------
\* Full VotingWithSafeAt behavior - includes IncreaseMaxBal

EXTENDS Integers

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

IncreaseMaxBal(self, b) ==
  /\ b > maxBal[self]
  /\ maxBal' = [maxBal EXCEPT ![self] = b]
  /\ UNCHANGED votes

VoteFor(self, b, v) ==
  /\ maxBal[self] <= b
  /\ DidNotVoteIn(self, b)
  /\ \A p \in Acceptor \ {self} :
       \A w \in Value : VotedFor(p, b, w) => (w = v)
  /\ SafeAt(b, v)
  /\ votes' = [votes EXCEPT ![self] = votes[self] \cup {<<b, v>>}]
  /\ maxBal' = [maxBal EXCEPT ![self] = b]

acceptor(self) ==
  \E b \in Ballot:
    \/ IncreaseMaxBal(self, b)
    \/ \E v \in Value: VoteFor(self, b, v)

Next == \E self \in Acceptor: acceptor(self)

Spec == Init /\ [][Next]_<<votes, maxBal>>

=============================================================================
