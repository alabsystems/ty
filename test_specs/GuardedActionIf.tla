---- MODULE GuardedActionIf ----
EXTENDS Integers
\* Test case with guarded action and IF condition

VARIABLE x, y

Init == x = 0 /\ y = 0

\* Guarded action that increments x (only when x < 1)
ActionX == x < 1 /\ x' = x + 1

\* Y depends on whether ActionX was taken
TNext == y' = IF ActionX THEN 5 ELSE y

\* Next with [ActionX]_x pattern combined with TNext
Next == /\ (ActionX \/ UNCHANGED x)
        /\ TNext

\* Expected states from {x=0, y=0}:
\*   1. {x=0, y=0} - initial
\*   2. {x=1, y=5} - ActionX taken (x<1 satisfied, x'=1, ActionX true so y'=5)
\*   3. {x=0, y=0} - UNCHANGED x (ActionX false since x'=x≠x+1, so y'=y=0)
\*
\* From {x=1, y=5}:
\*   - ActionX guard fails (x=1 not < 1)
\*   - UNCHANGED x: y' = IF ActionX THEN 5 ELSE y = IF FALSE THEN 5 ELSE 5 = 5
\*   - So {x=1, y=5} loops to itself
\*
\* Total distinct: 2 states

====
