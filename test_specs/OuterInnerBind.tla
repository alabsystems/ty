---- MODULE OuterInnerBind ----
\* Test pattern: outer bound var (acc) + inner existential (m) in EXCEPT path

EXTENDS Integers

CONSTANTS
    INS,  \* Instance set (like RM)
    ACC   \* Acceptor set

VARIABLES
    aState,  \* aState[ins][acc] = record [mbal, bal]
    msgs     \* Set of messages

Init ==
    /\ aState = [ins \in INS |-> [acc \in ACC |-> [mbal |-> 0, bal |-> -1]]]
    /\ msgs = {}

Send(m) == msgs' = msgs \cup {m}

\* Add a message (like Phase1a)
AddMsg(ins) ==
    /\ Send([type |-> "phase1a", ins |-> ins, bal |-> 1])
    /\ UNCHANGED aState

\* Process message with OUTER (acc) and INNER (m) bindings
\* This is the exact PaxosCommit Phase1b pattern
Process(acc) ==
    \E m \in msgs :
        /\ m.type = "phase1a"
        /\ aState[m.ins][acc].mbal < m.bal
        /\ aState' = [aState EXCEPT ![m.ins][acc].mbal = m.bal]
        /\ Send([type |-> "phase1b",
                 ins  |-> m.ins,
                 mbal |-> m.bal,
                 bal  |-> aState[m.ins][acc].bal,
                 acc  |-> acc])

Next ==
    \/ \E ins \in INS : AddMsg(ins)
    \/ \E acc \in ACC : Process(acc)

Spec == Init /\ [][Next]_<<aState, msgs>>

====
