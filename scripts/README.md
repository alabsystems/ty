# Scripts

Andrew Yates <andrewyates.name@gmail.com>

## Build configurations

`tla-cli`'s default features are `["mimalloc", "ay", "clean-cic"]`: a plain
`cargo build --release --bin ty` is the full verification build (symbolic +
hardware backends and the certifying commands included). The lean build is
`cargo build --release --bin ty --no-default-features --features mimalloc`.
`check_ay_build_gate.sh` asserts the feature configurations all compile (this
repo has no CI runner — gates here are run manually).

## Benchmark corpus prerequisite

Scripts that compare against TLC (`compare_with_tlc.sh`, `mass_tlc_compare.sh`,
`perf_loser_scan.sh`, `ty_vs_tlc_memtime.sh`) and the `spec_regression` test
need the TLA+ benchmark corpus at `~/tlaplus-examples` (override with
`TLAPLUS_EXAMPLES`). It is not in git — fetch it first with `ty corpus fetch`
(downloads the sha256-verified release asset pinned to the commit in
`tests/tlc_comparison/spec_baseline.json`). See `ty corpus --help` for details.

Use the Rust-owned `ty supremacy compare`, `matrix`, and `matrix-full-suite`
commands for current TLC evidence. For a multi-day corpus run, use the
plan-bound `matrix-campaign-plan`, `matrix-segment`,
`matrix-merge-inventory`, and `matrix-merge` workflow. Its full operating
contract lives in the internal strict Linux launcher document, which is not
part of this snapshot; run each command with `--help` for the authoritative
flag contract.
Create the plan with a fresh absolute `--artifact-root`, the exact
`<artifact-root>/campaign-plan.json` `--output`, and an explicit nonzero
`--runtime-timeout`. The documented production protocol conservatively uses
`14400`; the command does not enforce that as an exact value or minimum. The
operator's selected timeout is fixed by the plan and applies separately to
each TLC or TY subprocess.

For bounded observation storage, only the root attestor's
`quotactl(Q_GETQUOTA)` records authorize project limits and current usage.
Directory `statvfs`/`fstatvfs` values are filesystem-global reserve telemetry
on the supported Linux/ext4 baseline and must not be interpreted as project
headroom. The kernel-enforced E/P hard byte and inode quotas remain the exact
transient project upper bounds. The P soft thresholds are checked against the
stable post-quiescence payload commitment before prune; exceeding either
threshold makes the observation nonqualifying.

The plan creates its root, `segments`, and `attempts` directories with mode
`0700` and `campaign-plan.json` as a single-link regular file with mode `0600`;
strict launcher admission requires them to remain canonical, caller-owned,
non-symlink paths with those exact modes.

The plan digest uses project-versioned TY canonical JSON v1, not RFC 8785.
It binds every segment output/report path plus separate
`merge-inventory` and `merge-superiority` outputs. It also binds
O_EXCL-created, mode-`0600`, write-once-by-protocol attempt claims under
`<artifact-root>/attempts`: `segment-NNNN.json`,
`merge-inventory.json`, and `merge-superiority.json`. Each command durably
creates its `ty.supremacy.campaign-attempt-claim.v1` marker with `O_EXCL`, so
an intact retained tree refuses reuse even if output is absent. A successful
receipt hashes the marker as an `input_dependencies` entry with role
`attempt_marker`, plus the campaign plan and, for a merge, every exact segment
report.

The retained-artifact protocol forbids deleting a marker, failed output,
launcher evidence, or failed campaign, and forbids redirecting or retrying a
claimed command. A failure invalidates that campaign for publication.
`matrix-merge-inventory` can receipt-finalize a complete loser inventory,
always with `corpus_claim_pass=false`; it is not public superiority evidence.
`matrix-merge` remains the gate and may reuse the same finalized segments.

These controls operate within an explicit local-filesystem trust boundary.
They make reuse fail closed only while artifacts are retained; local evidence
cannot cryptographically prove that a writer deliberately did not delete an
earlier marker or campaign. Publication therefore assumes honest retained
custody unless an external append-only/WORM record or equivalent independent
custodian supplies stronger non-deletion assurance.

Cross-segment admission binds hashed stable guest identity (not `boot_id`),
the selected logical CPU and its actual processor model/part, stable machine,
and a direct guest-local
output-mount/device/partition/queue/filesystem contract, and a closed child
environment with validated inherited `HOME`/`PATH`, `C` locale, `UTC`
timezone, and narrow toolchain variables. Path-valued mount-option values are
hashed; output paths, inodes, timestamps, capacity counters, and attempt-local
scratch paths are excluded. Parent-device size, hashed
model/vendor/revision/serial/WWID sources, and the deterministic full
bounded-text queue configuration are included. Numeric storage-device changes
conservatively invalidate compatibility. Production binds the structured
`balanced_steady_guest_page_cache.v1` policy and its
TY-canonical-JSON-v1 digest. Each row performs the policy's fixed eight
retained, unscored crossover warmups before scored pairs; strict validation
rejects missing, reordered, failed, drifted, or artifact-reused warmups. A
fresh strict Linux campaign is still required to satisfy the measurement
gate.

`ty_vs_tlc_memtime.sh` is retained as a legacy local runtime+memory diagnostic;
it is not an acceptance gate and does not implement the repeated,
complete-process-tree strict protocol. The current claim status and the
remaining measurement blockers are summarized in
[the benchmarking guide](../benchmarking.md); the full strict-superiority
burndown is internal and not part of this snapshot.

## GitHub CLI rate-limit workarounds

When GitHub GraphQL quota is exhausted, `gh issue list` can fail with:

`GraphQL: API rate limit already exceeded ...`

Use the Rust REST fallback binary (`ty-gh-issue-list-rest`):

- List open issues with a label: `cargo run --release --bin ty-gh-issue-list-rest -- --label needs-review --limit 200`
- List do-audit queue: `cargo run --release --bin ty-gh-issue-list-rest -- --label do-audit --limit 200`
- List blocked issues: `cargo run --release --bin ty-gh-issue-list-rest -- --label blocked --state open --limit 200`
- Query a small subset of GitHub search syntax (still GraphQL-free): `cargo run --release --bin ty-gh-issue-list-rest -- --search "repo:alabsystems/ty state:open label:bug"`

## Validation

CoffeeCan codegen/AOT validation: run `ty-validate-coffecan-codegen-poc`.

Example:
```bash
cargo run --release --bin ty-validate-coffecan-codegen-poc -- --beans 10 --output-json target/coffecan_codegen_poc_10.json
```

Prereqs: `~/tlaplus/tytools.jar` and
`~/tlaplus-examples/specifications/CoffeeCan/CoffeeCan.tla`.

Current-report routing validation is now built into
`cargo run --release --bin ty -- system-health-gate`, which calls the
native Rust port of the legacy `check_current_doc_routing.py`.

HWMCC `/unsafe/` full canary sweep: run `hwmcc_unsafe_full_sweep.sh` after
building `ty` and installing HWMCC fixtures under
`~/hwmcc/benchmarks/bitlevel/safety`. Per
`reports/2026-04-20-r22-hwmcc-soundness-audit.md`, the 34
`2019/mann/data-integrity/unsafe/` benchmarks are a SAT oracle: any `unsat`
verdict is a P0 false-UNSAT signal.

Example:
```bash
cargo build --release --bin ty   # ay is a default feature — no flag needed
./scripts/hwmcc_unsafe_full_sweep.sh target/release/ty /tmp/hwmcc_unsafe_full_sweep.csv
```
