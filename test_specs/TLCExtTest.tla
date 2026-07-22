---- MODULE TLCExtTest ----

EXTENDS TLCExt, TLC, FiniteSets, Sequences, Naturals

VARIABLE mvs

Init ==
  mvs = { TLCModelValue("T_MV1"), TLCModelValue("T_MV2"), TLCModelValue("T_MV3") }

Next ==
  UNCHANGED mvs

Spec ==
  Init /\ [][Next]_mvs

Inv ==
  /\ Cardinality(mvs) = 3
  /\ TLCModelValue("T_MV1") \in mvs
  /\ TLCModelValue("T_MV2") \in mvs
  /\ TLCModelValue("T_MV3") \in mvs

-------------------------------------------------

\* Test model value equality via ASSUME
ASSUME TLCModelValue("T_MyMV") = TLCModelValue("T_MyMV")
ASSUME TLCModelValue("T_MyMV") # TLCModelValue("T_YourMV")

==================================
