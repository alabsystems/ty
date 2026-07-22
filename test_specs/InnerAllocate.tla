---- MODULE InnerAllocate ----
EXTENDS Sequences, Naturals, FiniteSets

CONSTANTS Clients, Resources
VARIABLES unsat, alloc, sched

Drop(seq, i) == SubSeq(seq, 1, i-1) \circ SubSeq(seq, i+1, Len(seq))

available == Resources \ (UNION {alloc[c] : c \in Clients})

Allocate(c, S) ==
    /\ S # {} /\ S \subseteq available \cap unsat[c]
    /\ \E i \in DOMAIN sched :
        /\ sched[i] = c
        /\ sched' = IF S = unsat[c] THEN Drop(sched, i) ELSE sched
    /\ alloc' = [alloc EXCEPT ![c] = @ \cup S]
    /\ unsat' = [unsat EXCEPT ![c] = @ \ S]

====
