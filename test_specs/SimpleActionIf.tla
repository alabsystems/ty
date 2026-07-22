---- MODULE SimpleActionIf ----
\* Minimal test case for IF with action predicate condition

VARIABLE x, y

Init == x = 0 /\ y = 0

\* Simple action that sets x to 1
ActionX == x' = 1

\* Y depends on whether ActionX was taken
\* y' = 5 if ActionX (x'=1), else y' = y
TNext == y' = IF ActionX THEN 5 ELSE y

\* Next with [ActionX]_x pattern (desugared) combined with TNext
Next == /\ (ActionX \/ UNCHANGED x)
        /\ TNext

\* Expected from {x=0, y=0}:
\*   - If ActionX taken: x'=1, y'=5 (ActionX is true)
\*   - If UNCHANGED x:   x'=0, y'=0 (ActionX is false, so y'=y=0)
\* Total: 2 distinct successors

====
