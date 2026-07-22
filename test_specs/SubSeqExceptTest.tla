---- MODULE SubSeqExceptTest ----
\* Adapted from TLC test-model/Github1145.tla
\* Tests SubSeq semantics with function constructors and EXCEPT

EXTENDS Naturals, Sequences

VARIABLE x

\* Helper function from Github1145
F(seq) == SubSeq(seq, 2, 2)

Init ==
  x = 0

Next ==
  UNCHANGED x

Spec ==
  Init /\ [][Next]_x

-------------------------------------------------

\* Test: SubSeq on function constructor with EXCEPT equals SubSeq on tuple
\* [i \in 1..4 |-> 0] EXCEPT ![2] = 1 should equal <<0, 1, 0, 0>>
ASSUME F([[i \in 1..4 |-> 0] EXCEPT ![2] = 1])[1]
     = F(<<0, 1, 0, 0>>)[1]

\* Test: SubSeq on function constructor over explicit set equals SubSeq on tuple
\* [i \in {1, 2, 3, 4} |-> 0] EXCEPT ![2] = 1 should equal <<0, 1, 0, 0>>
ASSUME F([[i \in {1, 2, 3, 4} |-> 0] EXCEPT ![2] = 1])[1]
     = F(<<0, 1, 0, 0>>)[1]

\* Additional SubSeq tests
ASSUME SubSeq(<<1,2,3,4,5>>, 2, 4) = <<2,3,4>>
ASSUME SubSeq(<<1,2,3>>, 1, 3) = <<1,2,3>>
ASSUME SubSeq(<<1,2,3>>, 1, 1) = <<1>>
ASSUME SubSeq(<<1,2,3>>, 2, 2) = <<2>>
ASSUME SubSeq(<<1,2,3>>, 3, 3) = <<3>>

\* SubSeq with empty result
ASSUME SubSeq(<<1,2,3>>, 2, 1) = <<>>

\* EXCEPT semantics with function constructors
ASSUME [[i \in 1..3 |-> i] EXCEPT ![2] = 42] = <<1, 42, 3>>
ASSUME [[i \in {1,2,3} |-> i*2] EXCEPT ![1] = 0] = <<0, 4, 6>>

==================================
