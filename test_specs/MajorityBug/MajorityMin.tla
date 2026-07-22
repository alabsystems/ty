---- MODULE MajorityMin ----
\* Minimal reproducer for PaxosCommit under-exploration bug
\* Bug: 3 acceptors causes 67% under-exploration

EXTENDS Integers, FiniteSets

CONSTANTS Acceptor, Majority

VARIABLE msgs

TypeOK == msgs \subseteq [type: {"msg"}, acc: Acceptor, val: {0, 1}]

Init == msgs = {}

\* Simple action using EXISTS over Majority
SendMsg(a) ==
    /\ \E MS \in Majority :
        /\ \A ac \in MS : ~\E m \in msgs : m.acc = ac
        /\ msgs' = msgs \union {[type |-> "msg", acc |-> a, val |-> 0]}

Next == \E a \in Acceptor : SendMsg(a)

Spec == Init /\ [][Next]_msgs
====
