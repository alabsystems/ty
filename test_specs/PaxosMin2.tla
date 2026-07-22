---- MODULE PaxosMin2 ----
\* Even more minimal reproducer

EXTENDS Integers

CONSTANTS
    RM,        \* Set of resource managers
    Acceptor   \* Set of acceptors

\* Simplified message type - just phase1a
Message ==
    [type : {"phase1a"}, ins : RM, bal : {1}]
        \cup
    [type : {"phase1b"}, ins : RM, mbal : {0, 1}, bal : {-1, 0, 1},
     val : {"none"}, acc : Acceptor]

VARIABLES
    aState,  \* aState[ins][acc] = [mbal, bal]
    msgs     \* Set of messages sent

TypeOK ==
    /\ aState \in [RM -> [Acceptor -> [mbal : {0, 1}, bal : {-1, 0, 1}]]]
    /\ msgs \in SUBSET Message

Init ==
    /\ aState = [ins \in RM |->
                  [acc \in Acceptor |-> [mbal |-> 0, bal |-> -1]]]
    /\ msgs = {}

Send(m) == msgs' = msgs \cup {m}

Phase1a(rm) ==
    /\ Send([type |-> "phase1a", ins |-> rm, bal |-> 1])
    /\ UNCHANGED aState

Phase1b(acc) ==
    \E m \in msgs :
        /\ m.type = "phase1a"
        /\ aState[m.ins][acc].mbal < m.bal
        /\ aState' = [aState EXCEPT ![m.ins][acc].mbal = m.bal]
        /\ Send([type |-> "phase1b",
                 ins  |-> m.ins,
                 mbal |-> m.bal,
                 bal  |-> aState[m.ins][acc].bal,
                 val  |-> "none",
                 acc  |-> acc])

Next ==
    \/ \E rm \in RM : Phase1a(rm)
    \/ \E acc \in Acceptor : Phase1b(acc)

Spec == Init /\ [][Next]_<<aState, msgs>>

====
