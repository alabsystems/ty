---- MODULE SelfLivenessJIT ----
(* Finite progress MIRROR of TY's trust-cg JIT engine-selection control flow  *)
(* (NOT its TLA+ semantics). Each action sets `span` to the literal Rust       *)
(* source location it abstracts, so a temporal-property lasso IS the           *)
(* Rust-span counterexample. Control points (crates/tla-check/src/check/       *)
(* model_checker/): trust_cg_dispatch.rs:271 (work threshold u64::MAX, dark),  *)
(* :321 (OR-gate), run_helpers.rs:6738/6762 (lazy trigger + work-arm consult), *)
(* :6827/7145/7168 (artifact build), :6810 (hot-swap predicate),               *)
(* run_bfs_notrace.rs:781 (startup interp), :861 (hot-swap call site).          *)
(*                                                                             *)
(* Synthesized from the engine control flow, then CORRECTED by the verification *)
(* loop on first run: (1) NativeEligible is a STATIC eligibility (TRUE here),   *)
(* not `artifactBuilt`, else P_hotness is vacuous; (2) Drain is blocked while a *)
(* *viable* (wired) lazy compile is still pending, else Drain races WorkArmFires *)
(* and the fixed cfg spuriously fails. Both faithfully mirror the engine: the   *)
(* run does not "finish on the interpreter" while a native compile that WILL    *)
(* fire is still pending.                                                       *)
EXTENDS Naturals
CONSTANTS WorkArmWired, HotSwapWired
VARIABLES engine, lazyPending, artifactBuilt, work, frontier, runDone, span
vars == <<engine, lazyPending, artifactBuilt, work, frontier, runDone, span>>

WorkCap == 3
NativeEligible == TRUE                 \* this spec class is native-eligible (static)
HotWork == work >= WorkCap
NativeEngaged == engine \in {"NativePerAction","CompiledBfsLoop"}

Init ==
    /\ engine = "InterpLoop"
    /\ lazyPending = TRUE
    /\ artifactBuilt = FALSE
    /\ work = 0
    /\ frontier = TRUE
    /\ runDone = FALSE
    /\ span = "run_bfs_notrace.rs:781:startup_interp"

AccumulateWork ==
    /\ ~runDone /\ frontier /\ work < WorkCap
    /\ work' = work + 1
    /\ span' = "bfs/transport_seq.rs:236:accumulate_work"
    /\ UNCHANGED <<engine, lazyPending, artifactBuilt, frontier, runDone>>

WorkArmFires ==
    /\ WorkArmWired            \* trust_cg_dispatch.rs:271 flipped off u64::MAX
    /\ ~runDone /\ lazyPending /\ HotWork
    /\ lazyPending' = FALSE
    /\ artifactBuilt' = TRUE   \* run_helpers.rs:6827/7145 initialize_trust_cg_cache
    /\ engine' = "NativePerAction"
    /\ span' = "run_helpers.rs:6765:work_arm_fires"
    /\ UNCHANGED <<work, frontier, runDone>>

HotSwap ==
    /\ HotSwapWired
    /\ ~runDone /\ frontier /\ artifactBuilt /\ NativeEligible
    /\ engine' = "CompiledBfsLoop"  \* run_bfs_notrace.rs:861
    /\ span' = "run_bfs_notrace.rs:861:hot_swap_to_compiled"
    /\ UNCHANGED <<lazyPending, artifactBuilt, work, frontier, runDone>>

(* Drain (run completes). Models that the engine does not finish the check     *)
(* while a viable native compile is still pending: enabled once work is hot AND *)
(* either the lazy decision is resolved (~lazyPending) or the work arm is dark  *)
(* (~WorkArmWired) so the compile would never fire anyway (the bug case).       *)
Drain ==
    /\ ~runDone /\ frontier /\ work >= WorkCap
    /\ (~lazyPending \/ ~WorkArmWired)                                  \* lazy decision resolved, or dark
    /\ (~artifactBuilt \/ ~HotSwapWired \/ engine = "CompiledBfsLoop")  \* don't finish owing a built hot-swap
    /\ frontier' = FALSE /\ runDone' = TRUE
    /\ span' = IF NativeEngaged THEN "run_compiled_bfs_loop:drain_native"
                                ELSE "run_bfs_loop:drain_interp"
    /\ UNCHANGED <<engine, lazyPending, artifactBuilt, work>>

Next == AccumulateWork \/ WorkArmFires \/ HotSwap \/ Drain
Spec == Init /\ [][Next]_vars
Fairness == WF_vars(AccumulateWork) /\ WF_vars(WorkArmFires)
            /\ WF_vars(HotSwap) /\ WF_vars(Drain)
FairSpec == Spec /\ Fairness

(* "When hot + eligible, native eventually engages." *)
P_hotness == [] ((HotWork /\ frontier /\ ~runDone /\ NativeEligible) => <> NativeEngaged)
(* "A built artifact is eventually executed by the compiled loop." *)
P_artifact_handoff ==
    [] ((artifactBuilt /\ frontier /\ ~runDone /\ NativeEligible)
          => <> (engine = "CompiledBfsLoop"))
(* Non-vacuity: native is reached at all. *)
P_reaches_native == <> NativeEngaged
====
