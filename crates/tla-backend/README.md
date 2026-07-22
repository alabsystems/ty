# tla-backend

TY's unified backend / engine-selection layer (leaf crate).

This crate owns the one typed decision that maps *(user request × spec
structural signals × policy)* → the engine that runs, plus a single auditable
admission gate registry and a typed `AdmissionDecision` for evidence. It
replaces the stringly-typed `TY_*` env-var handoff and the scattered
compiled-BFS admission predicates with one source of truth, while the
interpreter remains the permanent correctness oracle and universal fallback.

The design of record is the ty unified-backend architecture note
(2026-06-05) in the internal design docs.

Migration is staged (strangler-fig); Stage 1 leaves `ty check` byte-identical.
