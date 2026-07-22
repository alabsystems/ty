// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Ackermann array elimination — a BTOR2 → BTOR2 pre-pass that removes arrays so
//! large arrays with few distinct accesses bit-blast (fast bit-level IC3) instead
//! of routing to the word-level CHC portfolio. See
//! `docs/perf/array-elimination-ackermann-design.md`.
//!
//! MINIMAL SLICE (this file): reads on `input` (nondeterministic) arrays only —
//! no `write` chains and no `state` arrays yet. This is the simplest complete
//! case that exercises the SOUNDNESS CRUX (functional-consistency axioms), and it
//! is validated end-to-end against the sound direct 2^index expansion by the
//! solver-free differential harness (`bitblast::bad_reachable`). Anything outside
//! the slice makes the pass DECLINE (`None`), so the caller keeps the array /
//! routes to CHC exactly as before — fail-closed.

use std::collections::{HashMap, HashSet};

use crate::types::{Btor2Line, Btor2Node, Btor2Program, Btor2Sort};

/// Read budget for Ackermann eligibility: above this the `O(reads²)`
/// consistency axioms make the eliminated net constraint-dense (slow to
/// bit-blast, worse for ay-sat), so such nets stay on the CHC path.
const ACKERMANN_MAX_READS: usize = 64;

/// Eliminate arrays via Ackermann reduction, or `None` if the net is outside the
/// supported slice. On success the result is array-free and EQUISATISFIABLE.
///
/// Two phases: (1) resolve read-over-write chains to `ite` trees over reads on
/// base INPUT arrays (McCarthy axiom); (2) replace those base reads with fresh
/// vars + functional-consistency constraints. Both validated solver-free against
/// the direct 2^index expansion.
pub fn ackermann_eliminate(program: &Btor2Program) -> Option<Btor2Program> {
    let resolved = resolve_read_over_write(program)?;
    eliminate_base_reads(&resolved)
}

/// Phase 1: rewrite every `read(write(base, wi, wv), j)` to
/// `ite(eq(wi, j), wv, read(base, j))` (McCarthy read-over-write) to a FIXPOINT,
/// so write CHAINS (`write(write(...))`) fully unfold — the produced
/// `read(base, j)` is itself resolved next iteration if `base` is another write.
/// Result: a WRITE-FREE program in which every read is on a base input array.
/// Returns the input unchanged if it has no writes.
fn resolve_read_over_write(program: &Btor2Program) -> Option<Btor2Program> {
    let is_array_sort = |sid: i64| matches!(program.sorts.get(&sid), Some(Btor2Sort::Array { .. }));
    let has_array_op = program.lines.iter().any(|l| {
        matches!(l.node, Btor2Node::Write)
            || (matches!(l.node, Btor2Node::Ite) && is_array_sort(l.sort_id))
    });
    if !has_array_op {
        return Some(program.clone());
    }

    let mut lines = program.lines.clone();
    let mut sorts = program.sorts.clone();
    let mut next_id = lines.iter().map(|l| l.id).max().unwrap_or(0) + 1;
    let bool_sort = sorts
        .iter()
        .find(|(_, s)| matches!(s, Btor2Sort::BitVec(1)))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| {
            let id = next_id;
            next_id += 1;
            sorts.insert(id, Btor2Sort::BitVec(1));
            id
        });

    // Fixpoint: resolve one read-over-write per pass until none remain. Each pass
    // strictly reduces the total write-depth of reads, so it terminates; the cap
    // is a pure backstop.
    let cap = lines.len() * lines.len() + 16;
    for _ in 0..cap {
        let by_id: HashMap<i64, &Btor2Line> = lines.iter().map(|l| (l.id, l)).collect();
        let is_write = |id: i64| matches!(by_id.get(&id).map(|l| &l.node), Some(Btor2Node::Write));
        let is_array_ite = |id: i64| {
            by_id.get(&id).is_some_and(|l| {
                matches!(l.node, Btor2Node::Ite)
                    && matches!(program.sorts.get(&l.sort_id), Some(Btor2Sort::Array { .. }))
            })
        };

        let Some(pos) = lines.iter().position(|l| {
            matches!(l.node, Btor2Node::Read)
                && l.args
                    .first()
                    .is_some_and(|&a| a > 0 && (is_write(a) || is_array_ite(a)))
        }) else {
            break; // no more reads over writes / array-ites.
        };

        let read = lines[pos].clone();
        let arr = read.args[0];
        let j = *read.args.get(1)?;
        let src = *by_id.get(&arr)?;

        // Two peeled reads/eq are prepended before the rewritten node so they are
        // defined first (bit-blast processes in line order).
        let (rewritten, prepend): (Btor2Line, Vec<Btor2Line>) = if is_write(arr) {
            // read(write(base, wi, wv), j) → ite(eq(wi,j), wv, read(base,j)).
            let base = *src.args.first()?;
            let wi = *src.args.get(1)?;
            let wv = *src.args.get(2)?;
            if base < 0 {
                return None;
            }
            let eq_id = next_id;
            next_id += 1;
            let read_id = next_id;
            next_id += 1;
            (
                Btor2Line {
                    id: read.id,
                    sort_id: read.sort_id,
                    node: Btor2Node::Ite,
                    args: vec![eq_id, wv, read_id],
                },
                vec![
                    Btor2Line {
                        id: eq_id,
                        sort_id: bool_sort,
                        node: Btor2Node::Eq,
                        args: vec![wi, j],
                    },
                    Btor2Line {
                        id: read_id,
                        sort_id: read.sort_id,
                        node: Btor2Node::Read,
                        args: vec![base, j],
                    },
                ],
            )
        } else {
            // read(ite(c, a1, a2), j) → ite(c, read(a1,j), read(a2,j)).
            let c = *src.args.first()?;
            let a1 = *src.args.get(1)?;
            let a2 = *src.args.get(2)?;
            if a1 < 0 || a2 < 0 {
                return None;
            }
            let r1 = next_id;
            next_id += 1;
            let r2 = next_id;
            next_id += 1;
            (
                Btor2Line {
                    id: read.id,
                    sort_id: read.sort_id,
                    node: Btor2Node::Ite,
                    args: vec![c, r1, r2],
                },
                vec![
                    Btor2Line {
                        id: r1,
                        sort_id: read.sort_id,
                        node: Btor2Node::Read,
                        args: vec![a1, j],
                    },
                    Btor2Line {
                        id: r2,
                        sort_id: read.sort_id,
                        node: Btor2Node::Read,
                        args: vec![a2, j],
                    },
                ],
            )
        };
        lines[pos] = rewritten;
        for extra in prepend.into_iter().rev() {
            lines.insert(pos, extra);
        }
    }

    // Every read should now be on a base array. If any read is still over a
    // write or array-ite, the fixpoint hit its cap (malformed) ⇒ decline.
    let unresolved = lines.iter().any(|l| {
        matches!(l.node, Btor2Node::Read)
            && l.args.first().is_some_and(|&arr| {
                arr > 0
                    && lines.iter().any(|s| {
                        s.id == arr
                            && (matches!(s.node, Btor2Node::Write)
                                || (matches!(s.node, Btor2Node::Ite) && is_array_sort(s.sort_id)))
                    })
            })
    });
    if unresolved {
        return None;
    }
    // Writes and array-ites are now unreferenced by any read ⇒ drop them.
    lines.retain(|l| {
        !matches!(l.node, Btor2Node::Write)
            && !(matches!(l.node, Btor2Node::Ite) && is_array_sort(l.sort_id))
    });

    Some(Btor2Program {
        lines,
        sorts,
        num_inputs: program.num_inputs,
        num_states: program.num_states,
        bad_properties: program.bad_properties.clone(),
        constraints: program.constraints.clone(),
        fairness: program.fairness.clone(),
        justice: program.justice.clone(),
    })
}

/// Phase 2: replace reads on base INPUT arrays with fresh vars + consistency, or
/// `None` outside the slice (writes, state arrays, array-escape).
fn eliminate_base_reads(program: &Btor2Program) -> Option<Btor2Program> {
    let by_id: HashMap<i64, &Btor2Line> = program.lines.iter().map(|l| (l.id, l)).collect();

    let sort_of = |id: i64| by_id.get(&id).and_then(|l| program.sorts.get(&l.sort_id));
    let is_input_array = |id: i64| -> bool {
        matches!(
            by_id.get(&id).map(|l| &l.node),
            Some(Btor2Node::Input(_, _))
        ) && matches!(sort_of(id), Some(Btor2Sort::Array { .. }))
    };

    // Decline immediately on writes or state arrays (outside the slice).
    for line in &program.lines {
        match &line.node {
            Btor2Node::Write => return None,
            Btor2Node::State(_, _)
                if matches!(
                    program.sorts.get(&line.sort_id),
                    Some(Btor2Sort::Array { .. })
                ) =>
            {
                return None
            }
            _ => {}
        }
    }

    // The set of input-array ids present.
    let array_ids: HashSet<i64> = program
        .lines
        .iter()
        .filter(|l| is_input_array(l.id))
        .map(|l| l.id)
        .collect();
    if array_ids.is_empty() {
        return None; // no input arrays — nothing for this pass to do.
    }

    // Collect reads on those arrays, and enforce the array-ESCAPE rule: an array
    // id may be referenced ONLY as arg 0 of a `read`. If it flows anywhere else
    // (eq/ite/next/...), the fresh-var model would be unsound ⇒ decline.
    struct BaseRead {
        read_id: i64,
        array_id: i64,
        index_ref: i64,
        elem_sort: i64,
    }
    let mut reads: Vec<BaseRead> = Vec::new();
    for line in &program.lines {
        for (pos, &arg) in line.args.iter().enumerate() {
            if array_ids.contains(&arg.abs()) {
                // Only legal use: read(arr, idx) with arr at position 0, positive.
                let ok = matches!(line.node, Btor2Node::Read) && pos == 0 && arg > 0;
                if !ok {
                    return None; // array escapes into a non-read context.
                }
            }
        }
        if let Btor2Node::Read = line.node {
            let arr = *line.args.first()?;
            let idx = *line.args.get(1)?;
            if array_ids.contains(&arr) {
                reads.push(BaseRead {
                    read_id: line.id,
                    array_id: arr,
                    index_ref: idx,
                    elem_sort: line.sort_id,
                });
            } else {
                return None; // read over a non-input-array (e.g. would-be write) — decline.
            }
        }
    }
    if reads.is_empty() {
        return None;
    }
    // Ackermann-eligibility: the functional-consistency axioms are O(reads²).
    // Beyond a modest read budget the eliminated net becomes constraint-dense
    // (slow to bit-blast and worse for ay-sat), so DECLINE and keep the array on
    // the CHC path — exactly the "Ackermann-small ⇒ eliminate, else CHC" gate.
    if reads.len() > ACKERMANN_MAX_READS {
        return None;
    }

    // Fresh id allocator + a bitvector-1 (bool) sort for eq/implies results.
    let mut next_id = program.lines.iter().map(|l| l.id).max().unwrap_or(0) + 1;
    let mut sorts = program.sorts.clone();
    let bool_sort = program
        .sorts
        .iter()
        .find(|(_, s)| matches!(s, Btor2Sort::BitVec(1)))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| {
            let id = next_id;
            next_id += 1;
            sorts.insert(id, Btor2Sort::BitVec(1));
            id
        });

    // Rebuild lines: each `read` becomes a fresh Input of the element sort (SAME
    // id, so every consumer keeps working with no arg rewiring); array Input
    // declarations are dropped. Everything else is copied verbatim.
    let read_ids: HashSet<i64> = reads.iter().map(|r| r.read_id).collect();
    let mut out_lines: Vec<Btor2Line> = Vec::new();
    for line in &program.lines {
        if array_ids.contains(&line.id) {
            continue; // drop the array Input declaration.
        }
        if matches!(line.node, Btor2Node::SortArray(_, _)) {
            continue; // drop the array sort declaration.
        }
        if read_ids.contains(&line.id) {
            // Replace the read with a fresh nondeterministic Input (same id).
            out_lines.push(Btor2Line {
                id: line.id,
                sort_id: line.sort_id,
                node: Btor2Node::Input(line.sort_id, Some(format!("ackermann_read_{}", line.id))),
                args: vec![],
            });
            continue;
        }
        out_lines.push(line.clone());
    }

    // Consistency axioms: for every same-array read pair, add
    // `(index_j == index_k) => (read_j == read_k)` as a constraint. Missing a
    // pair is UNSOUND — this is the crux (see consistency_pairs test).
    let mut new_constraints: Vec<i64> = Vec::new();
    for j in 0..reads.len() {
        for k in (j + 1)..reads.len() {
            if reads[j].array_id != reads[k].array_id {
                continue;
            }
            let mut fresh = || {
                let id = next_id;
                next_id += 1;
                id
            };
            let eq_i = fresh();
            out_lines.push(Btor2Line {
                id: eq_i,
                sort_id: bool_sort,
                node: Btor2Node::Eq,
                args: vec![reads[j].index_ref, reads[k].index_ref],
            });
            let eq_v = fresh();
            out_lines.push(Btor2Line {
                id: eq_v,
                sort_id: bool_sort,
                node: Btor2Node::Eq,
                args: vec![reads[j].read_id, reads[k].read_id],
            });
            let imp = fresh();
            out_lines.push(Btor2Line {
                id: imp,
                sort_id: bool_sort,
                node: Btor2Node::Implies,
                args: vec![eq_i, eq_v],
            });
            let cid = fresh();
            out_lines.push(Btor2Line {
                id: cid,
                sort_id: 0,
                node: Btor2Node::Constraint(imp),
                args: vec![],
            });
            new_constraints.push(cid);
        }
    }

    // Drop array sorts from the sort table (the net is now array-free).
    sorts.retain(|_, s| !matches!(s, Btor2Sort::Array { .. }));

    let num_inputs = out_lines
        .iter()
        .filter(|l| matches!(l.node, Btor2Node::Input(_, _)))
        .count();
    let num_states = out_lines
        .iter()
        .filter(|l| matches!(l.node, Btor2Node::State(_, _)))
        .count();

    Some(Btor2Program {
        lines: out_lines,
        sorts,
        num_inputs,
        num_states,
        bad_properties: program.bad_properties.clone(),
        constraints: {
            let mut c = program.constraints.clone();
            c.extend(new_constraints);
            c
        },
        fairness: program.fairness.clone(),
        justice: program.justice.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitblast::{bad_reachable, bitblast};
    use crate::parser::parse;

    fn elim_matches_direct(net: &str, expect_reachable: bool) {
        let orig = parse(net).expect("parse");
        let direct = bitblast(&orig, 32).expect("direct blast");
        let elim = ackermann_eliminate(&orig).expect("eliminable");
        let acker = bitblast(&elim, 32).expect("ackermann blast");
        assert_eq!(
            bad_reachable(&direct),
            bad_reachable(&acker),
            "Ackermann elimination must agree with direct expansion on bad-reachability"
        );
        assert_eq!(
            bad_reachable(&direct),
            expect_reachable,
            "expected reachability"
        );
    }

    #[test]
    fn single_read_reachable() {
        // bad = mem[i] over a 2-element 1-bit input array — reachable.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 i
5 read 1 3 4
6 bad 5
";
        elim_matches_direct(net, true);
    }

    #[test]
    fn two_reads_same_index_consistency() {
        // bad = (mem[i] != mem[j]) AND (i == j) — UNREACHABLE, but only because
        // of functional consistency: without the axiom the eliminated net's two
        // fresh read vars could differ at i==j and bad would be (wrongly)
        // reachable. This is the test with teeth for the consistency crux.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 i
5 input 1 j
6 read 1 3 4
7 read 1 3 5
8 neq 1 6 7
9 eq 1 4 5
10 and 1 8 9
11 bad 10
";
        let orig = parse(net).expect("parse");
        let elim = ackermann_eliminate(&orig).expect("eliminable");
        assert!(!elim.constraints.is_empty(), "consistency constraint added");
        elim_matches_direct(net, false);
    }

    #[test]
    fn read_over_write_same_index() {
        // read(write(mem, wi, wv), wi) MUST equal wv (McCarthy) ⇒ (read != wv)
        // is unreachable. Exercises the read-over-write resolution.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 wi
5 input 1 wv
6 write 2 3 4 5
7 read 1 6 4
8 neq 1 7 5
9 bad 8
";
        elim_matches_direct(net, false);
    }

    #[test]
    fn read_over_write_symbolic_index() {
        // read(write(mem, wi, wv), ri) with ri==wi must equal wv; the else branch
        // reads the base array (a fresh var). Unreachable.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 wi
5 input 1 wv
6 input 1 ri
7 write 2 3 4 5
8 read 1 7 6
9 neq 1 8 5
10 eq 1 6 4
11 and 1 9 10
12 bad 11
";
        elim_matches_direct(net, false);
    }

    #[test]
    fn read_over_write_reachable_bad() {
        // bad = read(write(mem, wi, wv), wi) == wv-negated? Simpler: bad = the
        // read itself (= wv), reachable. Confirms resolution isn't trivially
        // making everything unreachable.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 wi
5 input 1 wv
6 write 2 3 4 5
7 read 1 6 4
8 bad 7
";
        elim_matches_direct(net, true);
    }

    #[test]
    fn nested_write_chain_resolves() {
        // write(write(mem, i, v), i, v) then read at i ⇒ v (McCarthy to a
        // FIXPOINT unfolds the chain). bad = the read = v, reachable.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 i
5 input 1 v
6 write 2 3 4 5
7 write 2 6 4 5
8 read 1 7 4
9 bad 8
";
        elim_matches_direct(net, true);
    }

    #[test]
    fn nested_write_distinct_indices() {
        // write(write(mem, a, va), b, vb); read at a. If a!=b the outer write
        // doesn't touch a, so read = va; if a==b, read = vb. bad = (read != va)
        // AND (a != b) ⇒ unreachable (a!=b ⇒ read=va ⇒ read!=va false).
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 a
5 input 1 b
6 input 1 va
7 input 1 vb
8 write 2 3 4 6
9 write 2 8 5 7
10 read 1 9 4
11 neq 1 10 6
12 neq 1 4 5
13 and 1 11 12
14 bad 13
";
        elim_matches_direct(net, false);
    }

    #[test]
    fn array_ite_read_distributes() {
        // read(ite(c, mem1, mem2), j) = ite(c, mem1[j], mem2[j]) — conditional
        // memory selection. bad = the read, reachable. The differential
        // (direct == ackermann) validates the distribution is correct.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem1
4 input 2 mem2
5 input 1 c
6 input 1 j
7 ite 2 5 3 4
8 read 1 7 6
9 bad 8
";
        elim_matches_direct(net, true);
    }

    #[test]
    fn array_ite_over_write() {
        // read(ite(c, write(mem,wi,wv), mem), wi) — mixes array-ite AND
        // read-over-write. If c: read = wv (McCarthy); else read = mem[wi].
        // bad = (read != wv) AND c ⇒ unreachable (c ⇒ read=wv).
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 input 1 wi
5 input 1 wv
6 input 1 c
7 write 2 3 4 5
8 ite 2 6 7 3
9 read 1 8 4
10 neq 1 9 5
11 and 1 10 6
12 bad 11
";
        elim_matches_direct(net, false);
    }

    #[test]
    fn declines_too_many_reads() {
        // More than ACKERMANN_MAX_READS reads on one array ⇒ decline (the
        // O(reads²) consistency would be too dense), keeping it on CHC.
        let n = ACKERMANN_MAX_READS + 1;
        let mut net =
            String::from("1 sort bitvec 8\n2 sort bitvec 1\n3 sort array 1 2\n4 input 3 mem\n");
        // index inputs 5..5+n, reads at ids 5+n..5+2n, OR them into a 1-bit bad.
        for k in 0..n {
            net.push_str(&format!("{} input 1 idx{k}\n", 5 + k));
        }
        for k in 0..n {
            net.push_str(&format!("{} read 2 4 {}\n", 5 + n + k, 5 + k));
        }
        // bad = read_0 (a single bit) — enough to keep the reads live.
        net.push_str(&format!("{} bad {}\n", 5 + 2 * n, 5 + n));
        let orig = parse(&net).expect("parse");
        assert!(
            ackermann_eliminate(&orig).is_none(),
            "{n} reads exceeds the Ackermann budget ⇒ decline"
        );
    }

    #[test]
    fn distinct_arrays_get_no_cross_consistency() {
        // Reads on DIFFERENT arrays at the SAME index are INDEPENDENT — mem1[i]
        // and mem2[i] may differ. bad = (mem1[i] != mem2[i]) is REACHABLE. A bug
        // that added cross-array consistency would wrongly make it unreachable,
        // and the differential (direct == ackermann) would catch the mismatch.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem1
4 input 2 mem2
5 input 1 i
6 read 1 3 5
7 read 1 4 5
8 neq 1 6 7
9 bad 8
";
        elim_matches_direct(net, true);
    }

    #[test]
    fn declines_array_equality() {
        // Array EQUALITY is extensional (a1==a2 iff ∀i a1[i]==a2[i]); it is NOT
        // soundly recoverable from the accessed indices alone, so an array
        // flowing into `eq` must trip the escape check and DECLINE.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem1
4 input 2 mem2
5 eq 1 3 4
6 bad 5
";
        let orig = parse(net).expect("parse");
        assert!(ackermann_eliminate(&orig).is_none(), "array-eq ⇒ decline");
    }

    #[test]
    fn declines_state_array() {
        // A `state` array (memory-as-state) is a later slice ⇒ decline.
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 mem
4 input 1 i
5 read 1 3 4
6 bad 5
";
        let orig = parse(net).expect("parse");
        assert!(
            ackermann_eliminate(&orig).is_none(),
            "state array ⇒ decline"
        );
    }
}
