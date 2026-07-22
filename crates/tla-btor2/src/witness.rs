// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BTOR2 counterexample witness projection.
//!
//! When a bit-blasted net is found UNSAFE, the IC3/PDR + BMC lane produces a
//! *bit-level* counterexample: a per-frame assignment to the AIGER latch and
//! input variables. HWMCC requires every `sat` verdict to be accompanied by a
//! **witness** — a replayable counterexample in the BTOR2 witness format. This
//! module projects the bit-level trace back to the word/array level (each
//! bitvector state's value, each array state's per-cell value, and the input
//! stimulus per frame) and serializes it in the standard btorsim-compatible
//! format:
//!
//! ```text
//! sat
//! b0                         # violated bad property/ies
//! #0                         # initial-state frame
//! 0 00000000 mem             # bitvector state: <ordinal> <binvalue> <symbol>
//! 0 [0] 00000000 mem         # array cell: <ordinal> [<binindex>] <binvalue> <symbol>
//! @0                         # inputs applied at frame 0
//! 0 1 req                    # <ordinal> <binvalue> <symbol>
//! @1
//! ...
//! .
//! ```
//!
//! ## Soundness
//!
//! The projector is *fail-closed*: it re-simulates the reconstructed
//! initial-state + per-frame inputs forward through the bit-blasted circuit
//! (btorsim's exact replay semantics) and only emits a witness if some `bad`
//! literal is actually true at the final frame with every `constraint`
//! satisfied at every frame. A trace that does not genuinely reach a bad state
//! yields `None` — never a bogus witness claiming `sat` without a real
//! counterexample.

use rustc_hash::FxHashMap;

use crate::bitblast::{eval_vals, lit_val, BitblastedCircuit};
use crate::types::{Btor2Node, Btor2Program, Btor2Sort};

/// A bitvector state's value (MSB-first binary string) or an array state's
/// per-cell values (`(index_bin, value_bin)`, both MSB-first).
#[derive(Debug)]
pub(crate) enum StateValue {
    BitVec(String),
    Array(Vec<(String, String)>),
}

/// A single state's assignment at the initial (`#0`) frame.
#[derive(Debug)]
struct StateAssignment {
    /// Ordinal among states, in declaration order — the leading witness column.
    ordinal: usize,
    /// State symbol, or a synthesized `s{k}`.
    name: String,
    value: StateValue,
}

/// A single input's assignment at one frame (`@k`).
#[derive(Debug)]
struct InputAssignment {
    ordinal: usize,
    name: String,
    /// MSB-first binary string.
    value: String,
}

/// A projected BTOR2 counterexample witness, ready to serialize.
#[derive(Debug)]
pub struct Btor2Witness {
    /// Violated bad-property ordinals (the `b<j>` header line).
    bad_props: Vec<usize>,
    /// Initial-state assignment (`#0` frame).
    init_state: Vec<StateAssignment>,
    /// Per-frame input assignments (`@k`), one entry per frame `0..num_frames`.
    input_frames: Vec<Vec<InputAssignment>>,
}

impl Btor2Witness {
    /// Assemble a witness from already-rendered primitive parts (the word-level
    /// lane, [`crate::word_replay`], builds these from a concrete `SmtValue`
    /// model instead of a bit-level trace). `init_state` is one
    /// `(ordinal, symbol, value)` per state in declaration order;
    /// `input_frames` is one `Vec<(ordinal, symbol, msb_first_bits)>` per frame.
    ///
    /// This keeps the private [`StateAssignment`]/[`InputAssignment`] shapes
    /// encapsulated so both projection lanes serialize through the identical
    /// [`Self::to_btor2_string`] (DRY).
    pub(crate) fn from_parts(
        bad_props: Vec<usize>,
        init_state: Vec<(usize, String, StateValue)>,
        input_frames: Vec<Vec<(usize, String, String)>>,
    ) -> Self {
        Btor2Witness {
            bad_props,
            init_state: init_state
                .into_iter()
                .map(|(ordinal, name, value)| StateAssignment {
                    ordinal,
                    name,
                    value,
                })
                .collect(),
            input_frames: input_frames
                .into_iter()
                .map(|frame| {
                    frame
                        .into_iter()
                        .map(|(ordinal, name, value)| InputAssignment {
                            ordinal,
                            name,
                            value,
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// Number of bad properties this witness reports as violated.
    pub fn bad_property_count(&self) -> usize {
        self.bad_props.len()
    }

    /// Number of frames (transition steps + 1) in the witness.
    pub fn frame_count(&self) -> usize {
        self.input_frames.len()
    }

    /// Serialize to the standard BTOR2 witness format (btorsim-compatible).
    pub fn to_btor2_string(&self) -> String {
        let mut out = String::new();
        out.push_str("sat\n");
        // Property header: the violated bad properties, space-separated.
        let props: Vec<String> = self.bad_props.iter().map(|j| format!("b{j}")).collect();
        out.push_str(&props.join(" "));
        out.push('\n');

        // Initial-state frame.
        out.push_str("#0\n");
        for st in &self.init_state {
            match &st.value {
                StateValue::BitVec(v) => {
                    out.push_str(&format!("{} {} {}\n", st.ordinal, v, st.name));
                }
                StateValue::Array(cells) => {
                    for (idx_bin, val_bin) in cells {
                        out.push_str(&format!(
                            "{} [{}] {} {}\n",
                            st.ordinal, idx_bin, val_bin, st.name
                        ));
                    }
                }
            }
        }

        // Per-frame input sections (empty header lines for input-free nets).
        for (k, frame) in self.input_frames.iter().enumerate() {
            out.push_str(&format!("@{k}\n"));
            for inp in frame {
                out.push_str(&format!("{} {} {}\n", inp.ordinal, inp.value, inp.name));
            }
        }

        out.push_str(".\n");
        out
    }
}

/// Bitvector width of a sort, or `None` for arrays / exotic sorts.
fn bitvec_width(sort: &Btor2Sort) -> Option<u32> {
    match sort {
        Btor2Sort::BitVec(w) => Some(*w),
        Btor2Sort::Array { .. } => None,
    }
}

/// Per-state array shape `(index_width, element_width)`, in state-declaration
/// order (aligned with [`BitblastedCircuit::state_bits`]). `None` = the state
/// is a plain bitvector (or an array over exotic element/index sorts we do not
/// split into cells; it is then emitted as a flat bitvector value).
fn array_shapes(program: &Btor2Program) -> Vec<Option<(u32, u32)>> {
    let mut shapes = Vec::new();
    for line in &program.lines {
        if let Btor2Node::State(sort_id, _) = &line.node {
            let shape = match program.sorts.get(sort_id) {
                Some(Btor2Sort::Array { index, element }) => {
                    match (bitvec_width(index), bitvec_width(element)) {
                        (Some(iw), Some(ew)) if ew > 0 && iw < u32::BITS => Some((iw, ew)),
                        _ => None,
                    }
                }
                _ => None,
            };
            shapes.push(shape);
        }
    }
    shapes
}

/// MSB-first binary string of the LSB-first `bits` literals evaluated in `val`.
fn binary_msb_first(val: &[bool], bits: &[u64]) -> String {
    (0..bits.len())
        .rev()
        .map(|b| if lit_val(val, bits[b]) { '1' } else { '0' })
        .collect()
}

/// MSB-first binary string of `value` rendered in `width` bits.
pub(crate) fn binary_msb_first_from_int(value: u128, width: u32) -> String {
    (0..width)
        .rev()
        .map(|b| if (value >> b) & 1 == 1 { '1' } else { '0' })
        .collect()
}

/// Project a bit-level counterexample trace onto the word/array level and build
/// a serializable BTOR2 witness.
///
/// `trace` is the per-frame assignment produced by
/// [`tla_aiger::extract_original_cex_trace`] over the SAME bit-blasted circuit
/// (`circuit`): keys `l{idx}` / `i{idx}` index `circuit.latches` /
/// `circuit.inputs` in declaration order, values are the latch/input bits at
/// that frame (frame 0 = initial state). `program` is the BTOR2 program the
/// circuit was bit-blasted from (used to recover per-state array dimensions);
/// its `State` nodes must be in the same order as `circuit.state_bits`.
///
/// Returns `None` (fail-closed) if the trace is empty, or if replaying the
/// reconstructed initial-state + inputs does not actually reach a bad state
/// with all constraints satisfied — never a witness that is not a real
/// counterexample.
pub fn project_bitblast_witness(
    program: &Btor2Program,
    circuit: &BitblastedCircuit,
    trace: &[FxHashMap<String, bool>],
) -> Option<Btor2Witness> {
    if trace.is_empty() {
        return None;
    }
    let num_frames = trace.len();
    let num_latches = circuit.latches.len();
    let num_inputs = circuit.inputs.len();

    // Per-frame input vectors (circuit.inputs order) from the `i{idx}` keys.
    // The BMC extractor assigns every latch/input at every frame; a missing key
    // defaults to false so the emitted witness is still fully concrete.
    let input_vals_per_frame: Vec<Vec<bool>> = (0..num_frames)
        .map(|k| {
            (0..num_inputs)
                .map(|idx| *trace[k].get(&format!("i{idx}")).unwrap_or(&false))
                .collect()
        })
        .collect();

    // Initial (frame-0) latch values from the `l{idx}` keys.
    let init_latch_vals: Vec<bool> = (0..num_latches)
        .map(|idx| *trace[0].get(&format!("l{idx}")).unwrap_or(&false))
        .collect();

    // Replay forward from the initial state + per-frame inputs (btorsim's
    // semantics) building the full variable table at each frame. This is the
    // honest self-check: it derives intermediate states itself rather than
    // trusting the trace, and confirms a bad state is reached.
    let mut per_frame_val: Vec<Vec<bool>> = Vec::with_capacity(num_frames);
    let mut latch_vals = init_latch_vals;
    for input_vals in &input_vals_per_frame {
        let val = eval_vals(circuit, input_vals, &latch_vals);
        // Every constraint must hold at every frame for a valid witness.
        if !circuit.constraints.iter().all(|&c| lit_val(&val, c)) {
            return None;
        }
        latch_vals = circuit
            .latches
            .iter()
            .map(|&(_, next, _)| lit_val(&val, next))
            .collect();
        per_frame_val.push(val);
    }

    // Which bad properties fire at the final frame.
    let last_val = &per_frame_val[num_frames - 1];
    let bad_props: Vec<usize> = circuit
        .bad
        .iter()
        .enumerate()
        .filter(|(_, &b)| lit_val(last_val, b))
        .map(|(j, _)| j)
        .collect();
    if bad_props.is_empty() {
        // Not actually a counterexample — withhold rather than emit a bogus one.
        return None;
    }

    // Word/array projection of the initial state (frame 0).
    let frame0_val = &per_frame_val[0];
    let shapes = array_shapes(program);
    let mut init_state = Vec::with_capacity(circuit.state_bits.len());
    for (ordinal, (name, bits)) in circuit.state_bits.iter().enumerate() {
        let shape = shapes.get(ordinal).copied().flatten();
        let value = match shape {
            Some((iw, ew)) if bits.len() == (1usize << iw) * ew as usize => {
                let num_cells = 1usize << iw;
                let ew = ew as usize;
                let mut cells = Vec::with_capacity(num_cells);
                for c in 0..num_cells {
                    let idx_bin = binary_msb_first_from_int(c as u128, iw);
                    let cell_bits = &bits[c * ew..(c + 1) * ew];
                    let val_bin = binary_msb_first(frame0_val, cell_bits);
                    cells.push((idx_bin, val_bin));
                }
                StateValue::Array(cells)
            }
            // Plain bitvector, or an array whose flat width did not match the
            // recovered dimensions (defensive) — emit as one flat value.
            _ => StateValue::BitVec(binary_msb_first(frame0_val, bits)),
        };
        init_state.push(StateAssignment {
            ordinal,
            name: name.clone(),
            value,
        });
    }

    // Per-frame input stimulus.
    let mut input_frames = Vec::with_capacity(num_frames);
    for val in &per_frame_val {
        let mut frame = Vec::with_capacity(circuit.input_bits.len());
        for (ordinal, (name, bits)) in circuit.input_bits.iter().enumerate() {
            frame.push(InputAssignment {
                ordinal,
                name: name.clone(),
                value: binary_msb_first(val, bits),
            });
        }
        input_frames.push(frame);
    }

    Some(Btor2Witness {
        bad_props,
        init_state,
        input_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitblast::bitblast;
    use crate::parser::parse;

    /// The genuinely-UNSAFE array net from the HWMCC witness-gap repro: mem is
    /// init to all-zero, `next` writes 5 into mem[0], and `bad = (mem[0] == 5)`
    /// fires one step later.
    const REPRO: &str = "\
1 sort bitvec 1
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
10 constd 2 5
13 write 3 4 8 10
7 next 3 4 13
9 read 2 4 8
11 eq 1 9 10
12 bad 11
";

    /// A 2-frame all-zero-initial bit-level trace over `circuit` (frame 0 sets
    /// every latch to false; later frames carry no inputs). The projector
    /// replays the transition relation itself, so this is a faithful stand-in
    /// for the concrete BMC trace on this input-free net.
    fn zero_init_trace(circuit: &BitblastedCircuit, frames: usize) -> Vec<FxHashMap<String, bool>> {
        (0..frames)
            .map(|k| {
                let mut m = FxHashMap::default();
                if k == 0 {
                    for idx in 0..circuit.latches.len() {
                        m.insert(format!("l{idx}"), false);
                    }
                }
                m
            })
            .collect()
    }

    #[test]
    fn projects_array_counterexample_to_btor2_witness() {
        let prog = parse(REPRO).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");
        // 2 cells x 8 bits.
        assert_eq!(circuit.latches.len(), 16);

        // Depth-1 counterexample: initial state + one transition.
        let trace = zero_init_trace(&circuit, 2);
        let w = project_bitblast_witness(&prog, &circuit, &trace)
            .expect("a real counterexample must project to a witness");

        // The replay reached a bad state (mem[0] == 5 after the write): the
        // projector only returns `Some` when a bad literal is true at the final
        // frame, so this asserts the trace genuinely fires the property.
        assert_eq!(w.bad_props, vec![0]);
        assert_eq!(w.frame_count(), 2);

        let text = w.to_btor2_string();
        assert!(text.starts_with("sat\nb0\n#0\n"), "header:\n{text}");
        // Array state emitted per cell, both zero at the initial frame.
        assert!(text.contains("0 [0] 00000000 mem\n"), "cell0:\n{text}");
        assert!(text.contains("0 [1] 00000000 mem\n"), "cell1:\n{text}");
        // Two input-free frames, then the terminator.
        assert!(text.contains("@0\n@1\n.\n"), "frames:\n{text}");
    }

    #[test]
    fn fail_closed_when_trace_does_not_reach_bad() {
        let prog = parse(REPRO).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        // Only frame 0 (mem all zero): bad = (mem[0] == 5) is FALSE here, and
        // no transition is taken, so no bad state is reached. The projector
        // must withhold the witness rather than emit a bogus `sat`.
        let trace = zero_init_trace(&circuit, 1);
        assert!(project_bitblast_witness(&prog, &circuit, &trace).is_none());

        // An empty trace likewise yields nothing.
        assert!(project_bitblast_witness(&prog, &circuit, &[]).is_none());
    }
}
