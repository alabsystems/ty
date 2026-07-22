------------------------ MODULE RecursiveOOM2566 ------------------------
\* Regression test for #2566: RECURSIVE operator evaluation causes OOM.
\* Models DllToBst pattern: multiple RECURSIVE operators that recurse
\* over set elements during model checking. With bind_all() env cloning,
\* this OOM'd on even small configurations. With bind_all_fast(), the
\* fast_bindings linked list avoids env growth.

EXTENDS Integers, FiniteSets

CONSTANT Addrs, Values

VARIABLE mem

\* --- RECURSIVE operators (mimicking DllToBst pattern) ---

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

RECURSIVE IsOrdered(_, _)
IsOrdered(seq, n) ==
    IF n <= 1 THEN TRUE
    ELSE seq[n-1] <= seq[n] /\ IsOrdered(seq, n - 1)

\* --- State machine ---

Init == mem = [a \in Addrs |-> CHOOSE v \in Values : TRUE]

Swap(a1, a2) ==
    /\ a1 /= a2
    /\ mem' = [mem EXCEPT ![a1] = mem[a2], ![a2] = mem[a1]]

Next == \E a1, a2 \in Addrs : Swap(a1, a2)

\* Invariants using RECURSIVE operators
SizeInv == SetSize(Addrs) = Cardinality(Addrs)
SumPreserved == SetSum(mem, Addrs) = SetSum(mem, Addrs)

Spec == Init /\ [][Next]_<<mem>>

=============================================================================
