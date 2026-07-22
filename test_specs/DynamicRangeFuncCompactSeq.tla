---- MODULE DynamicRangeFuncCompactSeq ----
EXTENDS Naturals, Sequences

CONSTANT MaxLen

VARIABLES xs, k

vars == <<xs, k>>

Init ==
    /\ xs = <<1, 2>>
    /\ k \in 0..1

Grow ==
    /\ Len(xs) < MaxLen
    /\ xs' =
        [i \in 1..(Len(xs) + 1) |->
            IF i <= Len(xs)
            THEN xs[i]
            ELSE k]
    /\ k' = k

Next ==
    \/ Grow
    \/ UNCHANGED vars

TypeOK ==
    /\ Len(xs) \in 2..MaxLen
    /\ xs \in Seq(0..2)
    /\ k \in 0..1

====
