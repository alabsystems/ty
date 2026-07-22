------------------------ MODULE ExistsOrBug4 ------------------------
\* Minimal repro for #154: EXISTS + Or pending chain bug
\* Shows 50% under-exploration (12 vs 24 states)
\* Root cause: BUG FIX #153 breaks pending chain across nested Apply

EXTENDS Naturals, FiniteSets

VARIABLES rcvd, pc, sent

Proc == {1, 2}

Init ==
  /\ rcvd = [p \in Proc |-> {}]
  /\ pc = [p \in Proc |-> "A"]
  /\ sent = {<<1, "M">>}

Receive(self) ==
  /\ pc[self] = "A"
  /\ \E msgs \in SUBSET sent:
       /\ rcvd[self] \subseteq msgs
       /\ rcvd' = [rcvd EXCEPT ![self] = msgs]

Accept(self) ==
  /\ pc' = [pc EXCEPT ![self] = "B"]
  /\ sent' = sent \cup {<<self, "M">>}

Nothing ==
  /\ UNCHANGED << pc, sent >>

\* Bug pattern: Receive processed with fresh pending, Or never sees EXISTS iterations
Step(self) ==
  /\ Receive(self)
  /\ \/ Accept(self)
     \/ Nothing

Next == \E self \in Proc: Step(self)

Spec == Init /\ [][Next]_<<rcvd, pc, sent>>

\* TY finds 12 states, TLC finds 24 states
=============================================================================
