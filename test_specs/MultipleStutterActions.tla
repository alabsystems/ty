---- MODULE MultipleStutterActions ----
\* Test case for disjunction distribution with multiple [A]_v patterns
\* This pattern comes from MCRealTimeHourClock and similar specs.
\*
\* The Next relation has:
\*   /\ (x' = 1 \/ UNCHANGED x)
\*   /\ (y' = 2 \/ UNCHANGED y)
\*
\* This should produce 4 successor states from {x=0, y=0}:
\*   1. {x=1, y=2}  -- both actions taken
\*   2. {x=1, y=0}  -- x action, y unchanged
\*   3. {x=0, y=2}  -- x unchanged, y action
\*   4. {x=0, y=0}  -- both unchanged (stuttering)

VARIABLE x, y

Init == x = 0 /\ y = 0

\* Action that sets x to 1
ActionX == x' = 1

\* Action that sets y to 2
ActionY == y' = 2

\* Next with multiple [A]_v patterns (desugared)
Next == /\ (ActionX \/ UNCHANGED x)
        /\ (ActionY \/ UNCHANGED y)

====
