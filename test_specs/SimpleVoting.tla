------------------------ MODULE SimpleVoting ------------------------
\* Simplified VoteProof to isolate config constant override issue
\* Spec has `Ballot == Nat` but config overrides with {0, 1, 2}

EXTENDS Integers

CONSTANT Value, Acceptor, Ballot

VARIABLES votes, maxBal

Init ==
  /\ votes = [a \in Acceptor |-> {}]
  /\ maxBal = [a \in Acceptor |-> -1]

IncreaseMaxBal(self, b) ==
  /\ b > maxBal[self]
  /\ maxBal' = [maxBal EXCEPT ![self] = b]
  /\ UNCHANGED votes

VoteFor(self, b, v) ==
  /\ maxBal[self] <= b
  /\ \A p \in Acceptor \ {self} :
       \A w \in Value : <<b, w>> \in votes[p] => (w = v)
  /\ votes' = [votes EXCEPT ![self] = votes[self] \cup {<<b, v>>}]
  /\ maxBal' = [maxBal EXCEPT ![self] = b]

acceptor(self) ==
  \E b \in Ballot:
    \/ IncreaseMaxBal(self, b)
    \/ \E v \in Value: VoteFor(self, b, v)

Next == \E self \in Acceptor: acceptor(self)

Spec == Init /\ [][Next]_<<votes, maxBal>>

TypeOK ==
  /\ votes \in [Acceptor -> SUBSET (Ballot \X Value)]
  /\ maxBal \in [Acceptor -> Ballot \cup {-1}]

=============================================================================
