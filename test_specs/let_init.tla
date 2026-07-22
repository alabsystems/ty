---- MODULE let_init ----
\* Test spec for LET with CHOOSE in Init (#580)
EXTENDS Naturals
VARIABLES x
Init == LET val == CHOOSE n \in 1..5 : n > 0 IN x = val
Next == x' = x
====
