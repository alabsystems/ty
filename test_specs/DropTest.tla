---- MODULE DropTest ----
EXTENDS Sequences, Naturals, TLC

\* Remove element at index i from a sequence.
\* Assumes that i \in 1..Len(seq)
Drop(seq, i) == SubSeq(seq, 1, i-1) \circ SubSeq(seq, i+1, Len(seq))

VARIABLES x

Init ==
    x = Drop(<<1, 2, 3>>, 1)

Next == UNCHANGED x

Spec == Init /\ [][Next]_x

\* Should be TRUE: Drop(<<1,2,3>>, 1) = <<2, 3>>
DropCorrect == x = <<2, 3>>

====
