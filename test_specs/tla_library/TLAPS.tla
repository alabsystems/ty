------------------------------- MODULE TLAPS --------------------------------
(***************************************************************************)
(* Minimal TLAPS stubs for TLC-based model checking.                       *)
(*                                                                         *)
(* Specs in tlaplus-examples often EXTEND TLAPS to attach proof pragmas    *)
(* (e.g. SMT/Zenon/Isa backends) in BY clauses. TLC does not run proofs,   *)
(* but it must still parse and semantically analyze these identifiers.     *)
(*                                                                         *)
(* For model checking, all of these are treated as TRUE.                   *)
(***************************************************************************)

\* SMT solver operators
SimpleArithmetic == TRUE
SMT == TRUE
SMTT(timeout) == TRUE
CVC3 == TRUE
CVC3T(timeout) == TRUE
Yices == TRUE
YicesT(timeout) == TRUE
veriT == TRUE
veriTT(timeout) == TRUE
Z3 == TRUE
Z3T(timeout) == TRUE
Spass == TRUE
SpassT(timeout) == TRUE

\* Zenon prover operators
Zenon == TRUE
ZenonT(timeout) == TRUE
SlowZenon == TRUE
SlowerZenon == TRUE
VerySlowZenon == TRUE
SlowestZenon == TRUE

\* Isabelle prover operators
Isa == TRUE
IsaT(timeout) == TRUE
IsaM(method) == TRUE
IsaMT(method, timeout) == TRUE
Auto == TRUE
Force == TRUE
Blast == TRUE
SimplifyAndSolve == TRUE
Simplification == TRUE
AutoBlast == TRUE

\* Temporal logic / proof backends
LS4 == TRUE
PTL == TRUE
PropositionalTemporalLogic == TRUE

\* Multi-backend selectors
AllProvers == TRUE
AllProversT(timeout) == TRUE
AllSMT == TRUE
AllSMTT(timeout) == TRUE
AllIsa == TRUE
AllIsaT(timeout) == TRUE

\* Proof-theorem hooks (kept as simple facts)
THEOREM SetExtensionality == TRUE
OBVIOUS

THEOREM NoSetContainsEverything == TRUE
OBVIOUS

\* Isabelle helper pragma
IsaWithSetExtensionality == TRUE

\* ENABLED expansion pragmas
ExpandENABLED == TRUE

\* Additional proof-level pragmas
NormalizeENABLED == TRUE
AutoUSE == TRUE

=============================================================================
