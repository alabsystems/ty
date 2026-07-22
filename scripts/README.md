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

`ty_vs_tlc_memtime.sh` is the runtime+memory differential: it measures BOTH
wall-clock and peak RSS single-threaded and flags any spec where TY loses on
either metric (gating win/loss on verdict parity).

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
