---- MODULE TLAPSTest ----
(*
 * Test spec for TLAPS operator support in TY.
 *
 * All TLAPS operators are proof backend pragmas that return TRUE
 * when used during model checking. This spec verifies that:
 * 1. Zero-arity operators (SMT, Zenon, Isa, etc.) work
 * 2. Parameterized operators (SMTT, ZenonT, IsaM, etc.) work
 * 3. These operators integrate correctly with spec invariants
 *)
EXTENDS TLAPS, Naturals

VARIABLE x

Init == x = 0
Next == x' = (x + 1) % 7

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..6

\* Test zero-arity SMT solver operators
TestSMT == SMT
TestCVC3 == CVC3
TestYices == Yices
TestveriT == veriT
TestZ3 == Z3

\* Test zero-arity Zenon operators
TestZenon == Zenon
TestSlowZenon == SlowZenon

\* Test zero-arity Isabelle operators
TestIsa == Isa
TestAuto == Auto
TestBlast == Blast

\* Test zero-arity temporal logic operators
TestLS4 == LS4
TestPTL == PTL

\* Test zero-arity multi-backend operators
TestAllProvers == AllProvers
TestAllSMT == AllSMT
TestAllIsa == AllIsa

\* Test zero-arity theorems
TestSetExtensionality == SetExtensionality
TestNoSetContainsEverything == NoSetContainsEverything

\* Test parameterized operators (with timeout)
TestSMTT == SMTT(10)
TestZenonT == ZenonT(20)
TestIsaT == IsaT(30)
TestIsaM == IsaM("auto")
TestIsaMT == IsaMT("blast", 60)

\* Combined invariant - all TLAPS operators return TRUE
TLAPSInvariant ==
    /\ TestSMT /\ TestCVC3 /\ TestYices /\ TestveriT /\ TestZ3
    /\ TestZenon /\ TestSlowZenon
    /\ TestIsa /\ TestAuto /\ TestBlast
    /\ TestLS4 /\ TestPTL
    /\ TestAllProvers /\ TestAllSMT /\ TestAllIsa
    /\ TestSetExtensionality /\ TestNoSetContainsEverything
    /\ TestSMTT /\ TestZenonT /\ TestIsaT /\ TestIsaM /\ TestIsaMT

\* Main invariant
Invariant == TypeOK /\ TLAPSInvariant

====
