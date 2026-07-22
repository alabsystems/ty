---- MODULE InstanceTest ----
\* Test INSTANCE with the same Allocate pattern

EXTENDS Sequences, Naturals

CONSTANTS Clients, Resources

VARIABLES sched, unsat, alloc, network

Base == INSTANCE InstanceBase

Messages == [type : {"allocate"}, clt : Clients, rsrc : SUBSET Resources]

Init ==
    /\ Base!Init
    /\ network = {}

AllocateWithMsg(c, S) ==
    /\ Base!Allocate(c, S)
    /\ network' = network \cup {[type |-> "allocate", clt |-> c, rsrc |-> S]}

Next == \E c \in Clients, S \in SUBSET Resources : AllocateWithMsg(c, S)

Inv == Base!AllocInv

vars == <<sched, unsat, alloc, network>>
Spec == Init /\ [][Next]_vars
====
