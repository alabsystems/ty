---- MODULE FlatSymMutex ----
\* WP-11 slice 2 (wishlist item 9) differential fixture: a small mutex over a
\* symmetric process set whose ENTIRE layout passes flat-symmetry admission:
\*   st    : [Procs -> {"idle","trying","cs"}]  — model-value-keyed function
\*           with a proven fixed String range (key-window permutation, identity
\*           payload transform),
\*   owner : Procs \union {nobody}              — proven FixedScalar model-value
\*           enum (NameId remap; `nobody` is fixed by every group element).
\* TypeOK supplies the G2 finite-universe proofs that upgrade both vars to
\* flat-primary-admissible kinds. SYMM declares the full symmetric group on
\* Procs.
\*
\* Differential bar (must be EXACT):
\*   - interpreter SymmetryCanonical arm  == flat arm (TY_FLAT_SYMMETRY=1)
\*     on states/transitions/verdict, and
\*   - orbit-count equivalence vs the --no-reduction ground truth.
EXTENDS Naturals, TLC

CONSTANTS Procs, nobody

VARIABLES st, owner

SYMM == Permutations(Procs)

TypeOK == /\ st \in [Procs -> {"idle", "trying", "cs"}]
          /\ owner \in Procs \union {nobody}

Init == /\ st = [p \in Procs |-> "idle"]
        /\ owner = nobody

Try(p) == /\ st[p] = "idle"
          /\ st' = [st EXCEPT ![p] = "trying"]
          /\ UNCHANGED owner

Enter(p) == /\ st[p] = "trying"
            /\ owner = nobody
            /\ st' = [st EXCEPT ![p] = "cs"]
            /\ owner' = p

Leave(p) == /\ st[p] = "cs"
            /\ owner = p
            /\ st' = [st EXCEPT ![p] = "idle"]
            /\ owner' = nobody

Next == \E p \in Procs : Try(p) \/ Enter(p) \/ Leave(p)

Mutex == \A p, q \in Procs : (st[p] = "cs" /\ st[q] = "cs") => p = q
====
