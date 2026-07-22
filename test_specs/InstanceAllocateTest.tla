---- MODULE InstanceAllocateTest ----
\* Test if INSTANCE call affects sched' computation
EXTENDS Sequences, Naturals, FiniteSets, TLC

CONSTANTS Clients, Resources

Drop(seq, i) == SubSeq(seq, 1, i-1) \circ SubSeq(seq, i+1, Len(seq))

VARIABLES unsat, alloc, sched

Inner == INSTANCE InnerAllocate

TypeOK ==
    /\ unsat \in [Clients -> SUBSET Resources]
    /\ alloc \in [Clients -> SUBSET Resources]
    /\ sched \in Seq(Clients)

available == Resources \ (UNION {alloc[c] : c \in Clients})

Init ==
    /\ unsat = [c \in Clients |-> IF c = "a" THEN {"r1"} ELSE {}]
    /\ alloc = [c \in Clients |-> {}]
    /\ sched = <<"a", "b">>

\* Use the INSTANCE version
Next == Inner!Allocate("a", {"r1"})

vars == <<unsat, alloc, sched>>
Spec == Init /\ [][Next]_vars

\* After allocating all of "a"'s request, "a" should be dropped from schedule
AllocateDrops == unsat["a"] = {} => sched = <<"b">>

====
