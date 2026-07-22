//! End-to-end engine test: DijkstraMutex N=4 over TY's 17-slot flat layout,
//! with hand-written action device functions implementing exactly the contract
//! the trust-ir -> CUDA emitter generates (one candidate successor per action,
//! return 1 if enabled).
//!
//! The expected distinct-state count (33,288,512) is the symmetry-off count
//! independently agreed by the TY CPU engine and TLC.
//!
//! Environment-tolerant: on hosts without a usable CUDA device the test passes
//! trivially after asserting the probe returns a typed `Unavailable` error.

use tla_gpu::{probe, run_bfs, GpuBfsConfig, GpuBfsSpec, GpuError};

// Layout: slots 0..3 b[i], 4..7 c[i], 8 k, 9..12 pc[i], 13..16 temp[i].
// pc labels 0..12 = Li0,Li1,Li2,Li3a,Li3b,Li3c,Li3d,Li4a,Li4b,cs,Li5,Li6,ncs.
// temp: 0=defaultInitValue, 1+p=scalar proc p, 5+mask=subset of Proc.
//
// Actions are split per (label, self) plus per-binding specializations of the
// Li4b inner \E j (Route-A binding-specialization style): 4 procs x
// (12 single-successor labels + 4 j-bindings + 1 empty-set case) = 68.
const N: usize = 4;
const SLOTS: usize = 17;

fn dijkstra_actions_src() -> (String, usize) {
    let mut src = String::new();
    let mut k = 0usize;
    let action = |body: String, src: &mut String, k: &mut usize| {
        src.push_str(&format!(
            "static __device__ int ty_gpu_action_{k}(const long long* s, long long* t) {{\n{body}}}\n\n",
            k = *k,
        ));
        *k += 1;
    };

    for p in 0..N {
        // Li0: b[self] := FALSE; pc := Li1
        action(
            format!(
                "  if (s[{pc}] != 0) return 0;\n  t[{p}] = 0;\n  t[{pc}] = 1;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li1: pc := (k != self) ? Li2 : Li4a
        action(
            format!(
                "  if (s[{pc}] != 1) return 0;\n  t[{pc}] = (s[8] != {p}) ? 2 : 7;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li2: c[self] := TRUE; pc := Li3a
        action(
            format!(
                "  if (s[{pc}] != 2) return 0;\n  t[{c}] = 1;\n  t[{pc}] = 3;\n  return 1;\n",
                pc = 9 + p,
                c = 4 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li3a: temp[self] := k (scalar); pc := Li3b
        action(
            format!(
                "  if (s[{pc}] != 3) return 0;\n  t[{tmp}] = 1 + s[8];\n  t[{pc}] = 4;\n  return 1;\n",
                pc = 9 + p,
                tmp = 13 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li3b: pc := b[temp[self]] ? Li3c : Li3d
        action(
            format!(
                "  if (s[{pc}] != 4) return 0;\n  long long tp = s[{tmp}] - 1;\n  t[{pc}] = s[tp] ? 5 : 6;\n  return 1;\n",
                pc = 9 + p,
                tmp = 13 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li3c: k := self; pc := Li3d
        action(
            format!(
                "  if (s[{pc}] != 5) return 0;\n  t[8] = {p};\n  t[{pc}] = 6;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li3d: pc := Li1
        action(
            format!(
                "  if (s[{pc}] != 6) return 0;\n  t[{pc}] = 1;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li4a: c[self] := FALSE; temp[self] := Proc \ {self}; pc := Li4b
        action(
            format!(
                "  if (s[{pc}] != 7) return 0;\n  t[{c}] = 0;\n  t[{tmp}] = 5 + (0xF & ~(1 << {p}));\n  t[{pc}] = 8;\n  return 1;\n",
                pc = 9 + p,
                c = 4 + p,
                tmp = 13 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li4b, binding j: requires temp[self] a non-empty set containing j.
        for j in 0..N {
            if j == p {
                continue; // j == self is never in temp[self] = Proc \ {self} descendants
            }
            action(
                format!(
                    "  if (s[{pc}] != 8) return 0;\n  long long mask = s[{tmp}] - 5;\n  if (mask < 0 || !((mask >> {j}) & 1)) return 0;\n  t[{tmp}] = 5 + (mask & ~(1LL << {j}));\n  t[{pc}] = s[{cj}] ? 8 : 1;\n  return 1;\n",
                    pc = 9 + p,
                    tmp = 13 + p,
                    cj = 4 + j,
                ),
                &mut src,
                &mut k,
            );
        }
        // Li4b, empty set: pc := cs
        action(
            format!(
                "  if (s[{pc}] != 8) return 0;\n  if (s[{tmp}] != 5) return 0;\n  t[{pc}] = 9;\n  return 1;\n",
                pc = 9 + p,
                tmp = 13 + p,
            ),
            &mut src,
            &mut k,
        );
        // cs: pc := Li5
        action(
            format!(
                "  if (s[{pc}] != 9) return 0;\n  t[{pc}] = 10;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li5: c[self] := TRUE; pc := Li6
        action(
            format!(
                "  if (s[{pc}] != 10) return 0;\n  t[{c}] = 1;\n  t[{pc}] = 11;\n  return 1;\n",
                pc = 9 + p,
                c = 4 + p,
            ),
            &mut src,
            &mut k,
        );
        // Li6: b[self] := TRUE; pc := ncs
        action(
            format!(
                "  if (s[{pc}] != 11) return 0;\n  t[{p}] = 1;\n  t[{pc}] = 12;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
        // ncs: pc := Li0
        action(
            format!(
                "  if (s[{pc}] != 12) return 0;\n  t[{pc}] = 0;\n  return 1;\n",
                pc = 9 + p,
            ),
            &mut src,
            &mut k,
        );
    }

    // MutualExclusion; MCTypeOK is enforced by the encoding.
    src.push_str(
        "static __device__ int ty_gpu_invariants_ok(const long long* s) {\n\
         \x20 int in_cs = 0;\n\
         \x20 for (int i = 0; i < 4; i++) in_cs += (s[9 + i] == 9);\n\
         \x20 return in_cs <= 1;\n}\n",
    );
    (src, k)
}

fn init_rows() -> Vec<i64> {
    let mut rows = Vec::new();
    for k in 0..N as i64 {
        let mut row = vec![0i64; SLOTS];
        for i in 0..N {
            row[i] = 1; // b
            row[4 + i] = 1; // c
            row[9 + i] = 0; // pc = Li0
            row[13 + i] = 0; // temp = defaultInitValue
        }
        row[8] = k;
        rows.extend_from_slice(&row);
    }
    rows
}

#[test]
fn dijkstra4_exhaustive_state_exact() {
    match probe() {
        Err(GpuError::Unavailable(reason)) => {
            eprintln!("skipping GPU engine test: {reason}");
            return;
        }
        Err(other) => panic!("probe failed with non-availability error: {other}"),
        Ok(info) => eprintln!(
            "GPU: {} cc {}.{} ({} SMs)",
            info.device_name, info.cc_major, info.cc_minor, info.multiprocessors
        ),
    }

    let (actions_src, action_count) = dijkstra_actions_src();
    let spec = GpuBfsSpec {
        slots: SLOTS,
        action_count,
        actions_src,
        init_rows: init_rows(),
        track_slot_stats: false,
    };
    let config = GpuBfsConfig {
        table_bits: 27,
        frontier_cap_rows: 4 << 20,
        ..Default::default()
    };
    let outcome = run_bfs(&spec, &config).expect("GPU BFS should complete");

    eprintln!(
        "distinct={} transitions={} levels={} deadlocks={} wall={:?} compile={:?}",
        outcome.distinct_states,
        outcome.transitions,
        outcome.levels,
        outcome.deadlock_states,
        outcome.wall,
        outcome.compile_wall,
    );
    assert_eq!(outcome.distinct_states, 33_288_512, "state-exactness");
    assert_eq!(outcome.transitions, 146_157_712, "transition count");
    assert_eq!(outcome.levels, 89, "BFS levels");
    assert_eq!(outcome.deadlock_states, 0, "no deadlocks in DijkstraMutex");
    assert!(outcome.violation.is_none(), "MutualExclusion holds");
}
