---------------------------- MODULE Reachability ----------------------------
(***************************************************************************)
(* Model code reachability in a call graph.                                *)
(*                                                                         *)
(* Functions are nodes, calls are directed edges.                          *)
(* Entry points are a designated subset of functions.                      *)
(* A function is reachable if there exists a path from any entry point.    *)
(*                                                                         *)
(* Author: Andrew Yates <andrewyates.name@gmail.com>                               *)
(* Issue: #384                                                              *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Functions,       \* Set of all function identifiers
    EntryPoints,     \* Subset of Functions that are entry points
    Calls            \* Function: Functions -> SUBSET Functions (call edges)

ASSUME EntryPoints \subseteq Functions
ASSUME \A f \in Functions: Calls[f] \subseteq Functions

----------------------------------------------------------------------------
(* Reachability via transitive closure *)

\* Iterative reachability computation (more efficient than recursive)
RECURSIVE ReachableIter(_, _)
ReachableIter(frontier, visited) ==
    IF frontier = {} THEN visited
    ELSE LET newNodes == UNION {Calls[f] : f \in frontier} \ visited
         IN ReachableIter(newNodes, visited \cup newNodes)

\* All functions reachable from entry points
Reachable == ReachableIter(EntryPoints, EntryPoints)

\* Functions not reachable from any entry point
Orphans == Functions \ Reachable

----------------------------------------------------------------------------
(* Properties *)

\* THEOREM: No orphan code
NoOrphans == Orphans = {}

\* THEOREM: Entry points are always reachable (by definition)
EntryPointsReachable == EntryPoints \subseteq Reachable

\* Sanity check: calls only reference valid functions
CallsValid == \A f \in Functions: Calls[f] \subseteq Functions

----------------------------------------------------------------------------
(* State machine for exploring reachability *)

VARIABLE explored

Init == explored = EntryPoints

Next ==
    \E f \in explored:
        \E c \in Calls[f]:
            /\ c \notin explored
            /\ explored' = explored \cup {c}

Spec == Init /\ [][Next]_explored

\* Eventually all reachable functions are explored
Liveness == <>(explored = Reachable)

\* Invariant: explored is subset of reachable
ExploredSubsetReachable == explored \subseteq Reachable

=============================================================================
