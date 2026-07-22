---- MODULE ActionDependentIf ----
EXTENDS Integers
\* Test case for action-dependent IF conditions
\* This pattern comes from MCRealTimeHourClock where:
\*   TNext == t' = IF HCnxt THEN 0 ELSE t + 1
\*   BigNext == /\ [HCnxt]_hr /\ TNext
\*
\* The tricky part: t' depends on which branch of [HCnxt]_hr was taken.
\* HCnxt is an action predicate that is true when hr changes.

VARIABLE x, y

Init == x = 0 /\ y = 0

\* Action that changes x (bounded to max 2)
ActionX == /\ x < 2
           /\ x' = x + 1

\* Y depends on whether ActionX was taken (bounded)
\* y' = 0 if ActionX, else y' = y + 1 (up to max 2)
TNext == y' = IF ActionX THEN 0 ELSE IF y < 2 THEN y + 1 ELSE y

\* Next with [ActionX]_x pattern (desugared) combined with TNext
Next == /\ (ActionX \/ UNCHANGED x)
        /\ TNext

\* From {x=0, y=0}:
\*   - If ActionX taken: x'=1, y'=0 (because ActionX is true)
\*   - If UNCHANGED x:   x'=0, y'=1 (because ActionX is false)
\* Should produce 2 distinct successors

====
