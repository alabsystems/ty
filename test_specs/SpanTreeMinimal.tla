---------------------------- MODULE SpanTreeMinimal ----------------------------
(* Closer repro for SpanTree with both mom and dist variables *)
EXTENDS Integers

CONSTANTS Nodes, Edges, MaxCardinality, Root

Nbrs(n) == {m \in Nodes : {m, n} \in Edges}

VARIABLES mom, dist

Init == /\ mom = [n \in Nodes |-> n]
        /\ dist = [n \in Nodes |-> IF n = Root THEN 0 ELSE MaxCardinality]

Next == \E n \in Nodes :
          \E m \in Nbrs(n) :
             /\ dist[m] < 1 + dist[n]
             /\ \E d \in (dist[m]+1) .. (dist[n] - 1) :
                    /\ dist' = [dist EXCEPT ![n] = d]
                    /\ mom'  = [mom  EXCEPT ![n] = m]

Spec == Init /\ [][Next]_<<mom, dist>>
=============================================================================
