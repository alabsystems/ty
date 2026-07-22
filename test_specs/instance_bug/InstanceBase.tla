---- MODULE InstanceBase ----
EXTENDS Sequences, Naturals

Drop(seq, i) == SubSeq(seq, 1, i-1) \o SubSeq(seq, i+1, Len(seq))

CONSTANTS Clients, Resources

VARIABLES sched, unsat, alloc

available == Resources \ (UNION {alloc[c] : c \in Clients})

Init ==
    /\ sched = <<"c1", "c2">>
    /\ unsat = [c \in Clients |-> IF c = "c1" THEN {"r1"} ELSE IF c = "c2" THEN {"r1"} ELSE {}]
    /\ alloc = [c \in Clients |-> {}]

Allocate(c, S) ==
    /\ S # {}
    /\ S \subseteq available \cap unsat[c]
    /\ \E i \in DOMAIN sched :
        /\ sched[i] = c
        /\ \A j \in 1..(i-1) : unsat[sched[j]] \cap S = {}
        /\ sched' = IF S = unsat[c] THEN Drop(sched, i) ELSE sched
    /\ alloc' = [alloc EXCEPT ![c] = @ \cup S]
    /\ unsat' = [unsat EXCEPT ![c] = @ \ S]

Next == \E c \in Clients, S \in SUBSET Resources : Allocate(c, S)

AllocInv == \A i \in DOMAIN sched : unsat[sched[i]] # {}

vars == <<sched, unsat, alloc>>
Spec == Init /\ [][Next]_vars
====
