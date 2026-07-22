------------------------ MODULE SafeAtMinimal ------------------------
\* Test SafeAt evaluation in guard position

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

\* This action should be enabled only for b=0 from init
\* Because SafeAt(b, v) is FALSE for b>0 when maxBal is all -1
Vote0(self, v) ==
  /\ maxBal[self] <= 0
  /\ DidNotVoteIn(self, 0)
  /\ \A p \in Acceptor \ {self} :
       \A w \in Value : VotedFor(p, 0, w) => (w = v)
  /\ SafeAt(0, v)  \* Should be TRUE
  /\ votes' = [votes EXCEPT ![self] = votes[self] \cup {<<0, v>>}]
  /\ maxBal' = [maxBal EXCEPT ![self] = 0]

\* This action should be DISABLED from init because SafeAt(1, v) = FALSE
Vote1(self, v) ==
  /\ maxBal[self] <= 1
  /\ DidNotVoteIn(self, 1)
  /\ \A p \in Acceptor \ {self} :
       \A w \in Value : VotedFor(p, 1, w) => (w = v)
  /\ SafeAt(1, v)  \* Should be FALSE from init
  /\ votes' = [votes EXCEPT ![self] = votes[self] \cup {<<1, v>>}]
  /\ maxBal' = [maxBal EXCEPT ![self] = 1]

\* Count Vote1 executions - should be 0 from init
Vote1Count == \E self \in Acceptor, v \in Value: Vote1(self, v)

Next ==
  \/ \E self \in Acceptor, v \in Value: Vote0(self, v)
  \/ \E self \in Acceptor, v \in Value: Vote1(self, v)

Spec == Init /\ [][Next]_<<votes, maxBal>>

=============================================================================
