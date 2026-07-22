# TLC → ty migration

`ty check Spec.tla` runs your existing model unchanged: it picks up `Spec.cfg`
automatically and honors the standard TLC config statements (SPECIFICATION,
INIT/NEXT, INVARIANT(S), PROPERTY(IES), CONSTANT(S), CONSTRAINT(S),
ACTION_CONSTRAINT(S), SYMMETRY, VIEW, ALIAS, POSTCONDITION, CHECK_DEADLOCK).

| TLC flag | ty equivalent | Notes |
|---|---|---|
| `-config <file>` | `ty check --config <FILE>` | Only needed when the name differs from `<Spec>.cfg`. |
| `-workers <N\|auto>` | `--workers <N>` | Default `0` = auto (TLC defaults to 1 worker). |
| `-deadlock` | `--no-deadlock` | Both tools check deadlock by default; this turns it off. |
| `-coverage <min>` | `--coverage` | Per-action statistics at the end of the run, not on a timer. Sequential mode only. |
| `-difftrace` | `--difftrace` | Identical behavior: traces show only changed variables. |
| `-tool` | `--tool` | Same tagged output (Toolbox-compatible). Equivalent to `--output tlc-tool`. |
| `-continue` | `--continue-on-error` | Keep exploring after a violation; stable, TLC-comparable state counts. |
| `-checkpoint <min>` | `--checkpoint <DIR>` | Interval set by `--checkpoint-interval <SECONDS>` (default 300). |
| `-recover <id>` | `--resume <DIR>` | Resume from a checkpoint directory. |
| `-simulate` | `ty simulate Spec.tla` | Dedicated command (`ty check --simulate` also works). Trace count: `--num-traces`. |
| `-depth <N>` | `ty simulate --max-trace-length <N>` | Max steps per trace; default 100, same as TLC. |
| `-seed <N>` | `ty simulate --seed <N>` | Deterministic replay (`0` = random). |
| `-dump <file>` | No equivalent | ty does not write the reachable-state set to a file. `--output json` captures results, not states. |

## Behavioral differences

Two defaults differ from TLC; each has a one-flag opt-out.

1. **Engine.** `ty check` defaults to a fused explicit-state BFS + symbolic
   engine. Pass `--bfs-only` for pure explicit-state BFS matching TLC.
2. **Reductions.** Automatic symmetry and partial-order reduction are on by
   default, so distinct-state counts can differ from TLC. Pass
   `--no-reduction` for TLC-identical counts.
