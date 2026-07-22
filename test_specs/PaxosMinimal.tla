---- MODULE PaxosMinimal ----
\* Minimal reproducer for PaxosCommit state space bug

EXTENDS Integers

CONSTANTS
    RM,        \* Set of resource managers
    Acceptor,  \* Set of acceptors
    Majority,  \* Set of majority sets (set of sets)
    Ballot     \* Set of ballot numbers

ASSUME
  /\ Ballot \subseteq Nat
  /\ 0 \in Ballot
  /\ Majority \subseteq SUBSET Acceptor

\* Message types
Message ==
    [type : {"phase1a"}, ins : RM, bal : Ballot \ {0}]
        \cup
    [type : {"phase1b"}, ins : RM, mbal : Ballot, bal : Ballot \cup {-1},
     val : {"prepared", "aborted", "none"}, acc : Acceptor]

VARIABLES
    aState,  \* aState[ins][acc] = [mbal, bal, val]
    msgs     \* Set of messages sent

TypeOK ==
    /\ aState \in [RM -> [Acceptor -> [mbal : Ballot,
                                       bal  : Ballot \cup {-1},
                                       val  : {"prepared", "aborted", "none"}]]]
    /\ msgs \in SUBSET Message

Init ==
    /\ aState = [ins \in RM |->
                  [acc \in Acceptor
                     |-> [mbal |-> 0, bal |-> -1, val |-> "none"]]]
    /\ msgs = {}

Send(m) == msgs' = msgs \cup {m}

\* Phase 1a: Leader initiates ballot
Phase1a(bal, rm) ==
    /\ Send([type |-> "phase1a", ins |-> rm, bal |-> bal])
    /\ UNCHANGED aState

\* Phase 1b: Acceptor responds to phase 1a
Phase1b(acc) ==
    \E m \in msgs :
        /\ m.type = "phase1a"
        /\ aState[m.ins][acc].mbal < m.bal
        /\ aState' = [aState EXCEPT ![m.ins][acc].mbal = m.bal]
        /\ Send([type |-> "phase1b",
                 ins  |-> m.ins,
                 mbal |-> m.bal,
                 bal  |-> aState[m.ins][acc].bal,
                 val  |-> aState[m.ins][acc].val,
                 acc  |-> acc])

Next ==
    \/ \E bal \in Ballot \ {0}, rm \in RM : Phase1a(bal, rm)
    \/ \E acc \in Acceptor : Phase1b(acc)

Spec == Init /\ [][Next]_<<aState, msgs>>

====
