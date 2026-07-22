---- MODULE AllocateTest ----
EXTENDS Sequences, Naturals, FiniteSets, TLC

CONSTANTS Clients, Resources

Drop(seq, i) == SubSeq(seq, 1, i-1) \circ SubSeq(seq, i+1, Len(seq))

VARIABLES unsat, alloc, sched

TypeOK ==
    /\ unsat \in [Clients -> SUBSET Resources]
    /\ alloc \in [Clients -> SUBSET Resources]
    /\ sched \in Seq(Clients)

available == Resources \ (UNION {alloc[c] : c \in Clients})

Init ==
    /\ unsat = [c \in Clients |-> IF c = "a" THEN {"r1"} ELSE {}]
    /\ alloc = [c \in Clients |-> {}]
    /\ sched = <<"a", "b">>

Allocate(c, S) ==
    /\ S # {} /\ S \subseteq available \cap unsat[c]
    /\ \E i \in DOMAIN sched :
        /\ sched[i] = c
        /\ sched' = IF S = unsat[c] THEN Drop(sched, i) ELSE sched
    /\ alloc' = [alloc EXCEPT ![c] = @ \cup S]
    /\ unsat' = [unsat EXCEPT ![c] = @ \ S]

Next == Allocate("a", {"r1"})

vars == <<unsat, alloc, sched>>
Spec == Init /\ [][Next]_vars

\* After allocating all of "a"'s request, "a" should be dropped from schedule
\* Schedule should become <<"b">>, NOT <<"a">>
AllocateDrops == unsat["a"] = {} => sched = <<"b">>

====
