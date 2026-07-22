---- MODULE RecFuncTest ----
EXTENDS Integers

\* Simple recursive function to compute set size
\* Size(S) = 0 if S = {}, else 1 + Size(S \ {first element})
SizeRec(S) ==
  LET Sz[T \in SUBSET S] ==
    IF T = {} THEN 0
    ELSE LET n == CHOOSE x \in T : TRUE
         IN 1 + Sz[T \ {n}]
  IN Sz[S]

\* Non-recursive size (using Cardinality would be simpler but let's test manually)
SizeNonRec(S) ==
  LET elems == S
  IN IF elems = {} THEN 0
     ELSE IF \E a \in elems : elems = {a} THEN 1
     ELSE IF \E a, b \in elems : elems = {a, b} THEN 2
     ELSE 3

TestSet == {0, 1, 2}

VARIABLE x

Init ==
  x \in {SizeRec({}), SizeRec({0}), SizeRec({0,1}), SizeRec({0,1,2})}

Next == UNCHANGED x

Spec == Init /\ [][Next]_x

\* Expected: {0, 1, 2, 3}
SizesCorrect == x \in {0, 1, 2, 3}

====
