---- MODULE AYSmokeTest ----
(* Copyright 2026 Andrew Yates.
   Author: Andrew Yates <andrewyates.name@gmail.com>

   ay Init enumeration smoke test (Part of #633, #634).
   Tests that ay-based Init enumeration produces correct state counts.

   Expected results (verified against TLC):
   - Initial states: 5 (x=1,2,3,4,5)
   - Total states: 6 (x=1,2,3,4,5,6)
*)
EXTENDS Integers

VARIABLE x

\* Init with multiple integer values - exercises ay enumeration
Init == x \in 1..5

\* Simple Next - increment x up to 6
Next == x' = IF x < 6 THEN x + 1 ELSE x

====
