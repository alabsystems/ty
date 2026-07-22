------------------------ MODULE RecursiveStress2566 ------------------------
\* Stress test for #2566: Multiple RECURSIVE operators in Next-state evaluation.
\* Models the DllToBst pattern: 9 RECURSIVE definitions evaluated during
\* state exploration (not Init filters). Tests that bind_all_fast() prevents
\* OOM from O(n) env cloning across deep recursive call stacks.
\*
\* With ADDRS={a1,a2,a3,a4}, VALUES={1,2}: TLC produces 16 distinct states.

EXTENDS Integers, FiniteSets

CONSTANTS ADDRS, VALUES

VARIABLE mem

\* --- 9 RECURSIVE operators (matching DllToBst operator count) ---

RECURSIVE SetSize(_)
SetSize(S) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE x \in S : TRUE
         IN  1 + SetSize(S \ {x})

RECURSIVE SetSum(_, _)
SetSum(f, S) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE x \in S : TRUE
         IN  f[x] + SetSum(f, S \ {x})

RECURSIVE SetMax(_, _)
SetMax(f, S) ==
    IF Cardinality(S) = 1
    THEN f[CHOOSE x \in S : TRUE]
    ELSE LET x == CHOOSE x \in S : TRUE
             rest == S \ {x}
         IN  IF f[x] >= SetMax(f, rest) THEN f[x]
             ELSE SetMax(f, rest)

RECURSIVE SetMin(_, _)
SetMin(f, S) ==
    IF Cardinality(S) = 1
    THEN f[CHOOSE x \in S : TRUE]
    ELSE LET x == CHOOSE x \in S : TRUE
             rest == S \ {x}
         IN  IF f[x] <= SetMin(f, rest) THEN f[x]
             ELSE SetMin(f, rest)

RECURSIVE CountMatching(_, _, _)
CountMatching(f, S, v) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE x \in S : TRUE
         IN  (IF f[x] = v THEN 1 ELSE 0) + CountMatching(f, S \ {x}, v)

RECURSIVE AllDistinct(_, _)
AllDistinct(f, S) ==
    IF S = {} THEN TRUE
    ELSE LET x == CHOOSE x \in S : TRUE
             rest == S \ {x}
         IN  CountMatching(f, rest, f[x]) = 0 /\ AllDistinct(f, rest)

RECURSIVE MapApply(_, _, _)
MapApply(f, S, offset) ==
    IF S = {} THEN f
    ELSE LET x == CHOOSE x \in S : TRUE
             rest == S \ {x}
             newval == IF f[x] + offset > SetMax(f, ADDRS) THEN f[x] ELSE f[x] + offset
         IN  MapApply([f EXCEPT ![x] = newval], rest, offset)

RECURSIVE Hash(_, _)
Hash(f, S) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE x \in S : TRUE
         IN  (f[x] * 31 + 7) + Hash(f, S \ {x})

RECURSIVE Depth(_, _, _)
Depth(f, S, acc) ==
    IF S = {} THEN acc
    ELSE LET x == CHOOSE x \in S : TRUE
         IN  Depth(f, S \ {x}, acc + f[x])

\* --- State machine ---

Init == mem \in [ADDRS -> VALUES]

Swap(a1, a2) ==
    /\ a1 /= a2
    /\ mem' = [mem EXCEPT ![a1] = mem[a2], ![a2] = mem[a1]]

Next == \E a1, a2 \in ADDRS : Swap(a1, a2)

\* Invariants that exercise multiple RECURSIVE operators per state
SumPreserved == SetSum(mem, ADDRS) = SetSum(mem, ADDRS)
MaxBound == SetMax(mem, ADDRS) \in VALUES
MinBound == SetMin(mem, ADDRS) \in VALUES
HashDefined == Hash(mem, ADDRS) >= 0
DepthBound == Depth(mem, ADDRS, 0) >= 0

Spec == Init /\ [][Next]_<<mem>>

=============================================================================
