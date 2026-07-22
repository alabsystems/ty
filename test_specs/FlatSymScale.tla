---- MODULE FlatSymScale ----
\* WP-30 (wishlist item 9) differential fixture — the SCALED twin of
\* FlatSymMutex.
\*
\* FlatSymMutex proves flat-space symmetry canonicalization CORRECT (7 orbits
\* vs 20 unreduced states), but at 20 states it cannot measure anything: the
\* whole run is 14ms. This module keeps every admission-relevant property of
\* FlatSymMutex and scales the state space by ~5 orders of magnitude, so the
\* lexmin-over-a-flat-i64-buffer canonicalizer can be TIMED against the
\* interpreter's min-over-permutations on the `Value` tree.
\*
\* Every variable is deliberately of a flat-primary-admissible, provably
\* equivariant kind:
\*   st    : [Procs -> Phase]        model-value-keyed function with a proven
\*                                   fixed String range (key-window
\*                                   permutation, identity payload transform)
\*   flag  : [Procs -> {"lo","hi"}]  ditto
\*   mode  : [Procs -> Mode]         ditto — the free-cycling width knob
\*   owner : Procs \union {nobody}   FixedScalar model-value enum (NameId
\*                                   remap; `nobody` is fixed by every group
\*                                   element)
\*
\* DO NOT add an Int-valued keyed array here (e.g. [Procs -> 0..3]). Layout
\* inference classifies that as StringKeyedArray{value_types=[Int,..]}, which
\* is NOT flat-primary-admissible today, and it drops the WHOLE layout out of
\* flat-symmetry admission (`flat_primary_safe=false`). That single kind is
\* the narrowest gap between this fixture and, say, Disruptor's
\* `claimed_sequence`.
\*
\* TypeOK supplies the finite-universe proofs that upgrade all four vars to
\* flat-primary-admissible kinds. SYMM declares the FULL symmetric group on
\* Procs, so |G| = |Procs|! and per-state canonicalization cost is real.
\*
\* Differential bar (must be EXACT):
\*   - flat arm (TY_FLAT_SYMMETRY=1) == interpreter SymmetryCanonical arm
\*     (TY_FLAT_SYMMETRY=0) on states/transitions/verdict, AND
\*   - both STRICTLY SMALLER than the unreduced ground truth. The ground truth
\*     REQUIRES `--no-auto-symmetry` on the nosym cfg: ty's automatic symmetry
\*     detection is ON by default and independently rediscovers this same
\*     group, so a bare "no SYMMETRY in the cfg" twin reduces just as far and
\*     proves nothing. (That is exactly how the FlatSymMutex fixture came to
\*     look inert.)

EXTENDS Naturals, TLC

CONSTANTS Procs, nobody

VARIABLES st, owner, flag, mode

SYMM == Permutations(Procs)

Phase == {"idle", "want", "hold", "done"}
Mode  == {"m1", "m2", "m3", "m4", "m5"}

\* NOTE (WP-30, measured): `Cycle` below inlines this rotation instead of
\* calling an operator. Writing it as
\*     Cycle(p) == mode' = [mode EXCEPT ![p] = NextMode(mode[p])]
\* costs `mode` its fixed-String RANGE PROOF: layout inference then classifies
\* it `StringKeyedArray{..., encoding=ScalarSlots}` instead of
\* `encoding=FixedScalar`, which drops `supports_flat_primary()` for the WHOLE
\* layout and declines flat-symmetry admission (`flat_primary_safe=false`).
\* The range proof is what makes the key-window permutation a legal
\* re-encoding at the destination slot, so this is a real precondition, not an
\* accident — but it is also the second-narrowest admission gap after
\* Int-valued keyed arrays.
NextMode(m) == IF      m = "m1" THEN "m2"
               ELSE IF m = "m2" THEN "m3"
               ELSE IF m = "m3" THEN "m4"
               ELSE IF m = "m4" THEN "m5"
               ELSE                  "m1"

TypeOK == /\ st    \in [Procs -> Phase]
          /\ owner \in Procs \union {nobody}
          /\ flag  \in [Procs -> {"lo", "hi"}]
          /\ mode  \in [Procs -> Mode]

Init == /\ st    = [p \in Procs |-> "idle"]
        /\ owner = nobody
        /\ flag  = [p \in Procs |-> "lo"]
        /\ mode  = [p \in Procs |-> "m1"]

\* Width knob: every process may cycle its mode at any time, independently.
Cycle(p) == /\ mode' = [mode EXCEPT ![p] =
                          IF      @ = "m1" THEN "m2"
                          ELSE IF @ = "m2" THEN "m3"
                          ELSE IF @ = "m3" THEN "m4"
                          ELSE IF @ = "m4" THEN "m5"
                          ELSE                  "m1"]
            /\ UNCHANGED <<st, owner, flag>>

Want(p) == /\ st[p] = "idle"
           /\ st' = [st EXCEPT ![p] = "want"]
           /\ UNCHANGED <<owner, flag, mode>>

Acquire(p) == /\ st[p] = "want"
              /\ owner = nobody
              /\ st' = [st EXCEPT ![p] = "hold"]
              /\ owner' = p
              /\ UNCHANGED <<flag, mode>>

Toggle(p) == /\ st[p] = "hold"
             /\ owner = p
             /\ flag' = [flag EXCEPT ![p] = IF @ = "lo" THEN "hi" ELSE "lo"]
             /\ UNCHANGED <<st, owner, mode>>

Release(p) == /\ st[p] = "hold"
              /\ owner = p
              /\ st' = [st EXCEPT ![p] = "done"]
              /\ owner' = nobody
              /\ UNCHANGED <<flag, mode>>

Reset(p) == /\ st[p] = "done"
            /\ st' = [st EXCEPT ![p] = "idle"]
            /\ UNCHANGED <<owner, flag, mode>>

Flip(p) == /\ st[p] # "hold"
           /\ flag' = [flag EXCEPT ![p] = IF @ = "lo" THEN "hi" ELSE "lo"]
           /\ UNCHANGED <<st, owner, mode>>

Next == \E p \in Procs :
          \/ Cycle(p) \/ Want(p) \/ Acquire(p)
          \/ Toggle(p) \/ Release(p) \/ Reset(p) \/ Flip(p)

\* Mutual exclusion: at most one process may hold, and the holder is `owner`.
Mutex == \A p, q \in Procs :
           (st[p] = "hold" /\ st[q] = "hold") => p = q

OwnerAgrees == \A p \in Procs : (st[p] = "hold") => (owner = p)

====
