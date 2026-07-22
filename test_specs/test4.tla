--------------- MODULE test4 -------------

(* Test of fingerprinting of sets.
   Different representations of the same set should hash to the same state.
   All Next transitions should produce the same state as Init. *)

EXTENDS Naturals, Sequences

VARIABLE x, y, z, w

Type ==
  /\ x \in {{1, 2, 3}}
  /\ y \in {{"a", "b", "c"}}
  /\ z \in {[a : {1, 2, 3}, b : {1, 2, 3}, c : {1, 2, 3}]}
  /\ w \in {[{1, 2, 3} -> {1, 2, 3}]}


Init ==
  /\ x = {1, 2, 3}
  /\ y = {"a", "b", "c"}
  /\ z = [a : {1, 2, 3}, b : {1, 2, 3}, c : {1, 2, 3}]
  /\ w = [{1, 2, 3} -> {1, 2, 3}]

Inv  ==
  /\ TRUE
  /\ x = {1, 2, 3}
  /\ y = {"a", "b", "c"}
  /\ z = [a : {1, 2, 3}, b : {1, 2, 3}, c : {1, 2, 3}]
  /\ w = [{1, 2, 3} -> {1, 2, 3}]


Next ==
  \* Test 1: Set with duplicates
  \/ /\ x' = {3, 3, 2, 1}
     /\ UNCHANGED <<y, z, w>>

  \* Test 2: Range expression
  \/ /\ x' = 1..3
     /\ UNCHANGED <<y, z, w>>

  \* Test 3: Set filter
  \/ /\ x' = {i \in {5, 4, 3, 3, 2, 2, 1} : i \leq 3}
     /\ UNCHANGED <<y, z, w>>

  \* Test 4: Set map (different source)
  \/ /\ x' = {i-3 : i \in 4..6}
     /\ UNCHANGED <<y, z, w>>

  \* Test 5: Set map (source with dups)
  \/ /\ x' = {i-3 : i \in {6, 6, 5, 4, 4, 5, 5}}
     /\ UNCHANGED <<y, z, w>>

  \* Test 6: Set map over function space
  \/ /\ x' = { f[i] : i \in 1..3, f \in [{1,2,3} -> {1,2,3}] }
     /\ UNCHANGED <<y, z, w>>

  \* Test 7: DOMAIN of record equals set of strings
  \/ /\ y' = DOMAIN [a |-> 1, b |-> 2, c |-> 3]
     /\ UNCHANGED <<x, z, w>>

  \* Test 8: Record set equals function set with string domain
  \/ /\ z' = [{"a", "b", "c"} -> {1, 2, 3}]
     /\ UNCHANGED <<y, x, w>>

  \* Test 9: Function set with reordered domain/codomain
  \/ /\ w' = [{3,1, 2} -> {1, 3, 2}]
     /\ UNCHANGED <<y, x, z>>

  \* Test 10: Function set with duplicates
  \/ /\ w' = [{3,1, 3, 3, 3, 2} -> {2, 2, 1, 3, 2}]
     /\ UNCHANGED <<y, x, z>>


============================================
