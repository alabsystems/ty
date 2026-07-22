---------------------------- MODULE DirectLetInExists ----------------------------
\* Test case: Recursive LET directly inside EXISTS scope
\* This tests whether enumerate.rs correctly captures local_stack bindings
\* when a recursive LET references EXISTS-bound variables.

EXTENDS Integers

CONSTANT S

VARIABLE x

\* The key pattern: LET f[y] == ... inside \E z \in S: ... where f uses z
\* z is bound by EXISTS (on local_stack), not as an operator parameter

Init == x = 0

Next ==
  \E z \in S:
    LET
      f[y \in S] ==
        IF y = 0
        THEN z        \* <-- References EXISTS-bound z directly!
        ELSE f[y - 1] \* Recursive call
    IN
      x' = f[z]

Spec == Init /\ [][Next]_x

=============================================================================
