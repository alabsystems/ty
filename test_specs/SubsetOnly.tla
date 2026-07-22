------------------------ MODULE SubsetOnly ------------------------
\* Test SUBSET enumeration in isolation

EXTENDS Naturals

CONSTANTS N

VARIABLES pc, msgs_chosen

Proc == 1 .. N
M == { "M" }

vars == << pc, msgs_chosen >>

Init ==
  /\ pc \in [ Proc -> {"A", "B"} ]
  /\ msgs_chosen = {}

\* Action that enumerates SUBSET
ChooseMsgs(self) ==
  /\ pc[self] = "A"
  /\ \E msgs \in SUBSET (Proc \times M):
       msgs_chosen' = msgs
  /\ pc' = [pc EXCEPT ![self] = "B"]

Next == \E self \in Proc: ChooseMsgs(self)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ pc \in [ Proc -> {"A", "B"} ]
  /\ msgs_chosen \in SUBSET (Proc \times M)

=============================================================================
