---- MODULE MajorityComplex ----
\* More complex reproducer including LET-in with nested set operations
\* Inspired by PaxosCommit Phase2a pattern

EXTENDS Integers, FiniteSets

CONSTANTS Acceptor, Majority

Maximum(S) ==
    IF S = {} THEN -1
    ELSE CHOOSE x \in S : \A y \in S : x >= y

VARIABLE msgs

TypeOK == msgs \subseteq [type: {"1b", "2a"}, acc: Acceptor, bal: 0..1, val: {0, 1}]

Init == msgs = {}

\* Phase2a-like action with LET and Maximum
DoAction(rm) ==
    /\ ~\E m \in msgs : m.type = "2a" /\ m.val = rm
    /\ \E MS \in Majority :
        LET mset == {m \in msgs : m.type = "1b" /\ m.acc \in MS}
            maxbal == Maximum({m.bal : m \in mset})
            val == IF maxbal = -1 THEN 0 ELSE 1
        IN  /\ \A ac \in MS : \E m \in mset : m.acc = ac
            /\ msgs' = msgs \union {[type |-> "2a", acc |-> rm, bal |-> maxbal, val |-> val]}

\* Simple Phase1b-like action
Send1b(a) ==
    /\ ~\E m \in msgs : m.type = "1b" /\ m.acc = a
    /\ msgs' = msgs \union {[type |-> "1b", acc |-> a, bal |-> 0, val |-> 0]}

Next == \E a \in Acceptor : Send1b(a) \/ DoAction(a)

Spec == Init /\ [][Next]_msgs
====
