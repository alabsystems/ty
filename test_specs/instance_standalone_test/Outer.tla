---- MODULE Outer ----
EXTENDS Naturals
XInit(v) == v \in {0}
VARIABLE x
INSTANCE Inner
====
