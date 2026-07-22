-------------------------- MODULE WorkerDeadlineProtocol --------------------------
\* TLA+ specification for TY's WORKER / DEADLINE / DETACH protocol.
\*
\* Purpose: verify the concurrency pattern TY uses to bound a non-cooperative
\* "seeding" worker (DD / PDR / CHC / AIGER) under a wall-clock deadline without
\* (a) blocking the main thread forever, (b) losing or double-emitting a verdict,
\* or (c) starving the sound exhaustive-BFS oracle of its reserved budget.
\*
\* Faithful to the real Rust pattern (read, not guessed):
\*   crates/tla-petri/src/examinations/reachability/dd_fastpath.rs
\*       run_dd_reachability_seeding : spawn worker on a CLONED net, send result
\*       over an mpsc channel, rx.recv_timeout(DD_BUDGET) -> on Timeout/Disconnect
\*       resolve nothing and DETACH the worker (handle dropped, thread leaked).
\*   crates/tla-petri/src/examinations/reachability/pipeline.rs
\*       run_pdr_seeding_wall_capped / run_symbolic_seeding_wall_capped /
\*       run_aiger_seeding_wall_capped : sync_channel(1) + `let _ = tx.send(..)`
\*       (buffered, SendError ignored); symbolic_seed_deadline /
\*       deadline_preserving_reserve_at / compute_fallback_reserve reserve a BFS
\*       tail so the seeding lane cannot eat the exhaustive pass's budget.
\*   crates/tla-petri/src/examinations/reachability/types.rs
\*       PropertyTracker.flushed + flush_resolved (skip_flushed) : partial-flush
\*       already-decided formulas at the deadline; first-writer-wins so a value is
\*       never emitted twice (flush + final).
\*   crates/tla-check/src/parallel/checker/seeding.rs:42  drop(result_tx) and
\*   crates/tla-check/src/parallel/checker/finalize/collect.rs:117 recv_timeout
\*       : the in-scope mirror of the same join/detach-with-deadline contract.
\*
\* The four real hazards this models (and the guard that defuses each):
\*   (h1) worker sends AFTER main stopped receiving -> lost result / BLOCKED send.
\*        guard: buffered sync_channel(1) + IgnoreSendError (let _ = tx.send).
\*   (h2) main joins forever because the worker never signals (no deadline).
\*        guard: recv_timeout, NOT recv()  (UseDeadline).
\*   (h3) a verdict emitted twice (partial flush + final).
\*        guard: PropertyTracker.flushed first-writer-wins (FirstWriterWins).
\*   (h4) main-thread starvation: the seeding worker eats the whole budget, the
\*        sound BFS oracle gets none (the StateSpace regression class).
\*        guard: deadline_preserving_reserve_at reserves a BFS tail (ReserveBfsTail).
\*
\* Each guard is a CONSTANT switch so TY can prove the guard is load-bearing:
\* the shipped (safe) .cfg holds all four invariants deadlock-free; each
\* *.<hazard>.cfg flips exactly one guard off and TY produces the matching
\* counterexample.

EXTENDS Integers, FiniteSets, TLC

\* ============================================================================
\* CONSTANTS  (all config-settable; booleans + small Nats)
\* ============================================================================

CONSTANTS
    Budget,            \* total wall ticks the main thread will wait (DD_BUDGET)
    Reserve,           \* ticks held back for the exhaustive BFS oracle
    BfsCost,           \* ticks the BFS oracle needs to resolve the formula
    WorkerCost,        \* ticks the seeding worker needs to finish computing
    ChannelCap,        \* mpsc buffer slots (1 = sync_channel(1); 0 = rendezvous)
    WorkerMayDiverge,  \* worker can spin forever (non-polling solve) -> never sends
    UseDeadline,       \* TRUE: recv_timeout ; FALSE: recv() forever            (h2)
    IgnoreSendError,   \* TRUE: `let _ = tx.send` ; FALSE: blocking send        (h1)
    FirstWriterWins,   \* TRUE: flushed/verdict.is_some() guard ; FALSE: re-emit (h3)
    ReserveBfsTail     \* TRUE: reserve a BFS tail ; FALSE: seeding eats budget  (h4)

\* Seeding lane gives up this many ticks early so the BFS oracle keeps `Reserve`.
TimeoutThreshold == IF ReserveBfsTail THEN Reserve ELSE 0

\* ============================================================================
\* VARIABLES
\* ============================================================================

VARIABLES
    main,          \* main thread PC: spawn|join|flush|bfs|done
    worker,        \* worker PC: idle|compute|ready|sent|blocked|diverged|gone
    budget,        \* remaining wall ticks of the main thread's deadline
    progress,      \* worker compute progress (0..WorkerCost)
    recvOpen,      \* main is still listening on the channel
    chan,          \* buffered, undelivered worker messages (0..ChannelCap)
    seedResolved,  \* formula resolved via the worker/channel (seeding lane)
    bfsResolved,   \* formula resolved via the main BFS oracle
    flushedOut,    \* verdict already emitted to stdout (PropertyTracker.flushed)
    emits,         \* number of times the verdict was emitted/committed
    detached       \* main timed out and detached the worker (leaked thread)

vars == <<main, worker, budget, progress, recvOpen, chan,
          seedResolved, bfsResolved, flushedOut, emits, detached>>

Resolved == seedResolved \/ bfsResolved

\* ============================================================================
\* TYPE INVARIANT
\* ============================================================================

TypeOK ==
    /\ main \in {"spawn", "join", "flush", "bfs", "done"}
    /\ worker \in {"idle", "compute", "ready", "sent", "blocked", "diverged", "gone"}
    /\ budget \in 0..Budget
    /\ progress \in 0..WorkerCost
    /\ recvOpen \in BOOLEAN
    /\ chan \in 0..ChannelCap
    /\ seedResolved \in BOOLEAN
    /\ bfsResolved \in BOOLEAN
    /\ flushedOut \in BOOLEAN
    /\ emits \in 0..3
    /\ detached \in BOOLEAN

\* ============================================================================
\* INITIAL STATE
\* ============================================================================

Init ==
    /\ main = "spawn"
    /\ worker = "idle"
    /\ budget = Budget
    /\ progress = 0
    /\ recvOpen = TRUE
    /\ chan = 0
    /\ seedResolved = FALSE
    /\ bfsResolved = FALSE
    /\ flushedOut = FALSE
    /\ emits = 0
    /\ detached = FALSE

\* ============================================================================
\* MAIN THREAD spawns the worker on the cloned net and begins the join.
\* ============================================================================

Spawn ==
    /\ main = "spawn"
    /\ main' = "join"
    /\ worker' = "compute"
    /\ UNCHANGED <<budget, progress, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* ============================================================================
\* WORKER: compute on the cloned net, then send (or diverge / block / drop).
\* The worker keeps running after a detach (leaked thread) -> WorkerStep is NOT
\* gated on the main thread's budget, exactly like the real leaked worker.
\* ============================================================================

WorkerStep ==
    /\ worker = "compute"
    /\ progress < WorkerCost
    /\ progress' = progress + 1
    /\ UNCHANGED <<main, worker, budget, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

WorkerReady ==
    /\ worker = "compute"
    /\ progress = WorkerCost
    /\ worker' = "ready"
    /\ UNCHANGED <<main, budget, progress, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* The non-polling solver pathology (compute_p_invariants etc.): the worker can
\* spin forever and NEVER send. With UseDeadline this is harmless; without it the
\* main thread joins forever (h2).
WorkerDiverge ==
    /\ worker = "compute"
    /\ WorkerMayDiverge
    /\ progress < WorkerCost
    /\ worker' = "diverged"
    /\ UNCHANGED <<main, budget, progress, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* Buffer has room: send succeeds (sync_channel(1) accepts one message even if
\* the receiver is already gone -> may be a lost result if main detached).
WorkerSendOK ==
    /\ worker = "ready"
    /\ chan < ChannelCap
    /\ chan' = chan + 1
    /\ worker' = "sent"
    /\ UNCHANGED <<main, budget, progress, recvOpen,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* Buffer full / rendezvous + blocking send: the worker BLOCKS forever waiting
\* for a receive that (after detach) will never come (h1).
WorkerSendBlock ==
    /\ worker = "ready"
    /\ chan >= ChannelCap
    /\ ~IgnoreSendError
    /\ worker' = "blocked"
    /\ UNCHANGED <<main, budget, progress, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* Buffer full but `let _ = tx.send(..)` swallows the SendError: result dropped,
\* worker exits cleanly (no block).
WorkerSendDrop ==
    /\ worker = "ready"
    /\ chan >= ChannelCap
    /\ IgnoreSendError
    /\ worker' = "gone"
    /\ UNCHANGED <<main, budget, progress, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* ============================================================================
\* MAIN THREAD: join (recv_timeout), partial-flush, BFS oracle, done.
\* ============================================================================

\* Wall time elapses while the main thread waits; the reserve stops the clock
\* `Reserve` ticks early so the BFS tail is preserved.
JoinTick ==
    /\ main = "join"
    /\ budget > TimeoutThreshold
    /\ budget' = budget - 1
    /\ UNCHANGED <<main, worker, progress, recvOpen, chan,
                   seedResolved, bfsResolved, flushedOut, emits, detached>>

\* recv_timeout delivers a buffered message (commit the verdict; first-writer
\* records it; emission happens at the flush phase).
MainRecv ==
    /\ main = "join"
    /\ recvOpen
    /\ chan > 0
    /\ chan' = chan - 1
    /\ seedResolved' = TRUE
    /\ main' = "flush"
    /\ UNCHANGED <<worker, budget, progress, recvOpen,
                   bfsResolved, flushedOut, emits, detached>>

\* recv_timeout expires with nothing buffered: DETACH the worker (leak it) and
\* move on. Only enabled with a deadline; gated chan=0 so a delivered message is
\* received rather than dropped.
MainTimeout ==
    /\ main = "join"
    /\ UseDeadline
    /\ budget <= TimeoutThreshold
    /\ chan = 0
    /\ main' = "flush"
    /\ recvOpen' = FALSE
    /\ detached' = TRUE
    /\ UNCHANGED <<worker, budget, progress, chan,
                   seedResolved, bfsResolved, flushedOut, emits>>

\* Partial flush of an already-decided formula at the deadline. flushedOut is the
\* PropertyTracker.flushed first-writer guard: emit at most once here.
MainFlush ==
    /\ main = "flush"
    /\ main' = "bfs"
    /\ IF Resolved /\ ~flushedOut
       THEN /\ flushedOut' = TRUE
            /\ emits' = emits + 1
       ELSE /\ flushedOut' = flushedOut
            /\ emits' = emits
    /\ UNCHANGED <<worker, budget, progress, recvOpen, chan,
                   seedResolved, bfsResolved, detached>>

\* The sound exhaustive-BFS oracle runs over whatever the seeding lane left
\* unresolved, IF the reserve preserved enough budget. First-writer-wins: if the
\* formula is already resolved+flushed, BFS must NOT re-emit it (h3).
MainBfs ==
    /\ main = "bfs"
    /\ main' = "done"
    /\ IF ~Resolved /\ budget >= BfsCost
       THEN /\ bfsResolved' = TRUE            \* BFS resolves and emits
            /\ flushedOut' = TRUE
            /\ emits' = emits + 1
       ELSE IF ~Resolved /\ budget < BfsCost
       THEN /\ bfsResolved' = bfsResolved     \* STARVED: stays unresolved (h4)
            /\ flushedOut' = flushedOut
            /\ emits' = emits
       ELSE IF Resolved /\ flushedOut /\ ~FirstWriterWins
       THEN /\ bfsResolved' = TRUE            \* DOUBLE emit: flush + final (h3)
            /\ flushedOut' = flushedOut
            /\ emits' = emits + 1
       ELSE IF Resolved /\ ~flushedOut
       THEN /\ bfsResolved' = bfsResolved     \* resolved but unflushed: emit once
            /\ flushedOut' = TRUE
            /\ emits' = emits + 1
       ELSE /\ bfsResolved' = bfsResolved     \* resolved+flushed+first-writer: skip
            /\ flushedOut' = flushedOut
            /\ emits' = emits
    /\ UNCHANGED <<worker, budget, progress, recvOpen, chan, seedResolved, detached>>

\* Absorbing terminal stutter so deadlock detection only fires on a genuinely
\* stuck NON-terminal state (e.g. the h2 join-forever), never on clean shutdown.
Done ==
    /\ main = "done"
    /\ UNCHANGED vars

\* ============================================================================
\* NEXT / SPEC
\* ============================================================================

Next ==
    \/ Spawn
    \/ WorkerStep
    \/ WorkerReady
    \/ WorkerDiverge
    \/ WorkerSendOK
    \/ WorkerSendBlock
    \/ WorkerSendDrop
    \/ JoinTick
    \/ MainRecv
    \/ MainTimeout
    \/ MainFlush
    \/ MainBfs
    \/ Done

Spec == Init /\ [][Next]_vars

\* Weak fairness on every progress action lets TY discharge the liveness
\* properties; the worker may still legitimately diverge (no WF on WorkerDiverge).
Fairness ==
    /\ WF_vars(Spawn)
    /\ WF_vars(WorkerStep)
    /\ WF_vars(WorkerReady)
    /\ WF_vars(WorkerSendOK)
    /\ WF_vars(WorkerSendDrop)
    /\ WF_vars(JoinTick)
    /\ WF_vars(MainRecv)
    /\ WF_vars(MainTimeout)
    /\ WF_vars(MainFlush)
    /\ WF_vars(MainBfs)

FairSpec == Spec /\ Fairness

\* ============================================================================
\* SAFETY INVARIANTS
\* ============================================================================

\* (h1/h4) When the main thread finishes, the formula has a verdict. A detached
\* worker's discarded result is sound ONLY because the reserved BFS oracle
\* re-derives it; if the seeding lane starves the BFS (h4) the result is lost.
NoLostResult ==
    (main = "done") => Resolved

\* (h3) A verdict is emitted at most once across the partial flush and the final
\* output (PropertyTracker.flushed first-writer-wins).
NoDoubleResolve ==
    emits <= 1

\* (h1) The worker never wedges on a blocking send into a buffer no one drains.
NoBlockedSend ==
    worker # "blocked"

\* ============================================================================
\* LIVENESS PROPERTIES  (checked under FairSpec)
\* ============================================================================

\* (h2 / MainNeverBlocksForever) the main thread always leaves the join state.
MainNeverBlocksForever ==
    [](main = "join" => <>(main # "join"))

\* (TerminatesByDeadline) the protocol always reaches clean shutdown.
TerminatesByDeadline ==
    <>(main = "done")

===================================================================================
