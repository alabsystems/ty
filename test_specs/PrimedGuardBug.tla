------------------------ MODULE PrimedGuardBug ------------------------
\* Minimal repro: guard in disjunction uses primed variable set by EXISTS above

EXTENDS Naturals, FiniteSets

VARIABLES sent, rcvd, pc

Proc == {1, 2}
M == { "M" }

ProcM == Proc \X M

vars == << sent, rcvd, pc >>

Init ==
  /\ sent = {<<1, "M">>}  \* Start with something sent so SUBSET is non-trivial
  /\ rcvd = [ p \in Proc |-> {} ]
  /\ pc = [ p \in Proc |-> "A" ]

\* Receive sets rcvd' via EXISTS - msgs can be any subset of sent
Receive(self) ==
  /\ pc[self] = "A"
  /\ \E msgs \in SUBSET sent:  \* Simpler: just subset of sent
       /\ rcvd[self] \subseteq msgs
       /\ rcvd' = [rcvd EXCEPT ![self] = msgs ]

\* UponAccept - guard uses rcvd' from Receive EXISTS
UponAccept(self) ==
  /\ rcvd'[self] # {}    \* PRIMED GUARD: depends on EXISTS above
  /\ pc' = [pc EXCEPT ![self] = "B"]
  /\ sent' = sent \cup { <<self, "M">> }

\* UponNothing
UponNothing ==
  UNCHANGED << pc, sent >>

\* Pattern: EXISTS sets rcvd', then disjunction with primed guard
Step(self) ==
  /\ Receive(self)
  /\ \/ UponAccept(self)
     \/ UponNothing

Next == \E self \in Proc: Step(self)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ sent \in SUBSET ProcM
  /\ rcvd \in [ Proc -> SUBSET ProcM ]
  /\ pc \in [ Proc -> {"A", "B"} ]

=============================================================================
