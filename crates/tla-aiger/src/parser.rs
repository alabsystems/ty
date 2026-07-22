// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AIGER parser for both ASCII (`.aag`) and binary (`.aig`) formats.
//!
//! Supports the extended HWMCC header `aig M I L O A [B C J F]`
//! where `B`=bad, `C`=constraints, `J`=justice, `F`=fairness. The public entry
//! points are [`parse_file`] (auto-detecting), [`parse_aag`], and [`parse_aig`];
//! each returns a fully-resolved [`AigerCircuit`].
//!
//! Reference: "The AIGER And-Inverter Graph (AIG) Format Version 20071012"
//! by Armin Biere, Johannes Kepler University.

use std::path::Path;

use crate::error::AigerError;
use crate::types::*;

// ---------------------------------------------------------------------------
// Header parsing (shared between ASCII and binary)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct AigerHeader {
    is_binary: bool,
    maxvar: u64,
    num_inputs: u64,
    num_latches: u64,
    num_outputs: u64,
    num_ands: u64,
    num_bad: u64,
    num_constraints: u64,
    num_justice: u64,
    num_fairness: u64,
}

fn parse_header(line: &str) -> Result<AigerHeader, AigerError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 6 {
        return Err(AigerError::InvalidHeader(
            "header must have at least 6 fields: aag/aig M I L O A".into(),
        ));
    }

    let is_binary = match parts[0] {
        "aag" => false,
        "aig" => true,
        other => {
            return Err(AigerError::InvalidHeader(format!(
                "expected 'aag' or 'aig', got '{other}'"
            )));
        }
    };

    let parse_u64 = |s: &str, name: &str| -> Result<u64, AigerError> {
        s.parse::<u64>()
            .map_err(|_| AigerError::InvalidHeader(format!("invalid {name}: '{s}'")))
    };

    let maxvar = parse_u64(parts[1], "M (maxvar)")?;
    let num_inputs = parse_u64(parts[2], "I (inputs)")?;
    let num_latches = parse_u64(parts[3], "L (latches)")?;
    let num_outputs = parse_u64(parts[4], "O (outputs)")?;
    let num_ands = parse_u64(parts[5], "A (ands)")?;
    let num_bad = if parts.len() > 6 {
        parse_u64(parts[6], "B (bad)")?
    } else {
        0
    };
    let num_constraints = if parts.len() > 7 {
        parse_u64(parts[7], "C (constraints)")?
    } else {
        0
    };
    let num_justice = if parts.len() > 8 {
        parse_u64(parts[8], "J (justice)")?
    } else {
        0
    };
    let num_fairness = if parts.len() > 9 {
        parse_u64(parts[9], "F (fairness)")?
    } else {
        0
    };

    Ok(AigerHeader {
        is_binary,
        maxvar,
        num_inputs,
        num_latches,
        num_outputs,
        num_ands,
        num_bad,
        num_constraints,
        num_justice,
        num_fairness,
    })
}

// ---------------------------------------------------------------------------
// Header validation (fail-closed against malicious/oversized counts)
// ---------------------------------------------------------------------------

/// Hard ceiling on the maximum variable index (M). The transition-system and
/// CNF layers size dense `Vec`s of length `maxvar + 1` (and cast `maxvar` to
/// `u32`), so an unbounded `maxvar` is both an OOM vector and a silent-truncation
/// soundness hazard. 256M variables (~2GB of `Option<Var>` in the CNF map) is far
/// beyond any real benchmark while staying well under `u32::MAX`.
const MAX_VARS: u64 = 256 * 1024 * 1024;

/// Upper bound on any single section count (I, L, O, A, B, C, J, F) and on a
/// justice subcount. Sections cannot have more entries than there are variables,
/// plus this absolute cap guards counts that are not otherwise bounded by maxvar
/// (outputs, bad, constraints, justice, fairness can legitimately exceed maxvar
/// because they reference, not define, literals).
const MAX_SECTION_ENTRIES: u64 = MAX_VARS;

/// Cap the capacity passed to `Vec::with_capacity` so a large-but-not-yet-rejected
/// count (e.g. an output count that survives the per-section body check) cannot
/// pre-allocate gigabytes. The vector still grows on demand for legitimate files.
const SANE_CAP: usize = 1 << 16;

/// Hard ceiling on the number of implicit inputs in a *binary* file. Binary
/// inputs consume zero body bytes, so the per-section body check cannot bound
/// them; only maxvar (`I + L + A <= M <= MAX_VARS`) does, and that still permits
/// synthesizing hundreds of millions of `AigerSymbol`s. 16M inputs realizes to a
/// few hundred MB at most, far above any real benchmark.
const MAX_INPUTS: u64 = 16 * 1024 * 1024;

/// Capacity helper: never pre-reserve more than `SANE_CAP` entries, regardless of
/// the (already body-validated) count claimed by the header.
#[inline]
fn capped_capacity(count: u64) -> usize {
    count.min(SANE_CAP as u64) as usize
}

/// Validate a parsed header against the actual number of body bytes available.
///
/// Fail-closed: any count whose section cannot physically fit in the remaining
/// body bytes, any count above the absolute section cap, a `maxvar` above the
/// hard ceiling, or a `maxvar` smaller than `I + L + A` (which would mean more
/// defined variables than the header claims exist) is rejected *before* any
/// allocation sized off those counts.
///
/// `body_len` is the number of bytes after the header line. For each section we
/// require `count * min_bytes_per_entry <= body_len`; the minimum is a
/// conservative lower bound (smallest legal encoding of one entry) so legitimate
/// files are never rejected.
fn validate_header(h: &AigerHeader, body_len: usize) -> Result<(), AigerError> {
    // Absolute caps on individual section counts.
    let section_caps: [(u64, &str); 8] = [
        (h.num_inputs, "I (inputs)"),
        (h.num_latches, "L (latches)"),
        (h.num_outputs, "O (outputs)"),
        (h.num_ands, "A (ands)"),
        (h.num_bad, "B (bad)"),
        (h.num_constraints, "C (constraints)"),
        (h.num_justice, "J (justice)"),
        (h.num_fairness, "F (fairness)"),
    ];
    for (count, name) in section_caps {
        if count > MAX_SECTION_ENTRIES {
            return Err(AigerError::InvalidHeader(format!(
                "section {name} count {count} exceeds maximum {MAX_SECTION_ENTRIES}"
            )));
        }
    }

    // maxvar hard ceiling (guards transys/cnf dense-Vec sizing and u32 cast).
    if h.maxvar > MAX_VARS {
        return Err(AigerError::InvalidHeader(format!(
            "M (maxvar) {} exceeds maximum {MAX_VARS}",
            h.maxvar
        )));
    }

    // The number of *defined* variables is exactly I + L + A; the header's M must
    // be at least that, or the file claims more definitions than variables.
    let defined = h
        .num_inputs
        .checked_add(h.num_latches)
        .and_then(|x| x.checked_add(h.num_ands))
        .ok_or_else(|| AigerError::InvalidHeader("I + L + A overflows u64".into()))?;
    if defined > h.maxvar {
        return Err(AigerError::InvalidHeader(format!(
            "I + L + A = {defined} exceeds M (maxvar) {}",
            h.maxvar
        )));
    }

    // Per-section body-size check: each section must physically fit in body_len.
    // Minimum bytes per entry are conservative lower bounds for the legal encoding.
    //
    // ASCII (.aag): every section is line-based; smallest entry is one digit plus
    //   a newline = 2 bytes (latch/and lines are larger, so 2 is a safe floor).
    // Binary (.aig): inputs are implicit (0 body bytes); latches/outputs/bad/
    //   constraints/fairness are 1 ASCII line each (>= 2 bytes incl. newline);
    //   AND gates are >= 2 delta bytes; justice header is >= 2 bytes per record.
    let (in_min, latch_min, line_min, and_min, just_min): (u64, u64, u64, u64, u64) = if h.is_binary
    {
        (0, 2, 2, 2, 2)
    } else {
        (2, 2, 2, 2, 2)
    };

    let body = body_len as u64;
    let mut required: u64 = 0;
    let add = |required: &mut u64, count: u64, per: u64, name: &str| -> Result<(), AigerError> {
        let need = count.checked_mul(per).ok_or_else(|| {
            AigerError::InvalidHeader(format!("section {name} size overflows u64"))
        })?;
        *required = required
            .checked_add(need)
            .ok_or_else(|| AigerError::InvalidHeader("total body size overflows u64".into()))?;
        Ok(())
    };
    add(&mut required, h.num_inputs, in_min, "I (inputs)")?;
    add(&mut required, h.num_latches, latch_min, "L (latches)")?;
    add(&mut required, h.num_outputs, line_min, "O (outputs)")?;
    add(&mut required, h.num_bad, line_min, "B (bad)")?;
    add(
        &mut required,
        h.num_constraints,
        line_min,
        "C (constraints)",
    )?;
    // Each justice record is at least its count line (>= 2 bytes); the inner
    // literals are validated per-record at parse time against remaining bytes.
    add(&mut required, h.num_justice, just_min, "J (justice)")?;
    add(&mut required, h.num_fairness, line_min, "F (fairness)")?;
    add(&mut required, h.num_ands, and_min, "A (ands)")?;

    if required > body {
        return Err(AigerError::InvalidHeader(format!(
            "header declares sections needing at least {required} body bytes \
             but only {body} bytes follow the header"
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Circuit body validation (shared between ASCII and binary)
// ---------------------------------------------------------------------------

/// Validate a fully-parsed circuit body against its declared `maxvar`.
///
/// Enforces the two invariants the doc contracts of [`parse_aag`] /
/// [`parse_aig`] promise but that per-line parsing cannot check:
///
/// - **Literal range**: every literal in the file (input/latch/latch-next/
///   latch-reset/output/bad/constraint/justice/fairness/AND lhs+rhs) must
///   reference a variable `<= maxvar` ([`AigerError::InvalidLiteral`]).
/// - **Definition uniqueness**: every variable definition (input, latch, or
///   AND-gate output) must occur at most once, and a variable cannot be
///   defined as more than one kind ([`AigerError::DuplicateDefinition`]).
///   Without this, a duplicate AND gate with contradictory functions makes
///   the Tseitin transition relation UNSAT, and every downstream engine
///   vacuously reports `unsat` (a bogus SAFE) for a malformed file.
///
/// Called at the end of both [`parse_aag`] and [`parse_aig`] (fail-closed:
/// malformed files are rejected at parse time). `maxvar` is already capped by
/// `validate_header` (`MAX_VARS`), so the `defined` bitset allocation is safe.
fn validate_circuit(c: &AigerCircuit) -> Result<(), AigerError> {
    let maxvar = c.maxvar;
    let check_lit = |lit: u64| -> Result<(), AigerError> {
        if lit / 2 > maxvar {
            Err(AigerError::InvalidLiteral {
                literal: lit,
                maxvar,
            })
        } else {
            Ok(())
        }
    };

    for s in &c.inputs {
        check_lit(s.lit)?;
    }
    for l in &c.latches {
        check_lit(l.lit)?;
        check_lit(l.next)?;
        check_lit(l.reset)?;
    }
    for s in &c.outputs {
        check_lit(s.lit)?;
    }
    for s in &c.bad {
        check_lit(s.lit)?;
    }
    for s in &c.constraints {
        check_lit(s.lit)?;
    }
    for j in &c.justice {
        for &lit in &j.lits {
            check_lit(lit)?;
        }
    }
    for s in &c.fairness {
        check_lit(s.lit)?;
    }
    for a in &c.ands {
        check_lit(a.lhs)?;
        check_lit(a.rhs0)?;
        check_lit(a.rhs1)?;
    }

    // Definition uniqueness. All lits were range-checked above, so indexing
    // the (maxvar + 1)-sized bitset by lit/2 cannot go out of bounds.
    fn define(defined: &mut [bool], lit: u64) -> Result<(), AigerError> {
        let var = lit / 2;
        if defined[var as usize] {
            return Err(AigerError::DuplicateDefinition(var));
        }
        defined[var as usize] = true;
        Ok(())
    }
    let mut defined = vec![false; (maxvar + 1) as usize];
    for s in &c.inputs {
        define(&mut defined, s.lit)?;
    }
    for l in &c.latches {
        define(&mut defined, l.lit)?;
    }
    for a in &c.ands {
        define(&mut defined, a.lhs)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Binary delta encoding
// ---------------------------------------------------------------------------

fn decode_delta(data: &[u8], pos: &mut usize) -> Result<u64, AigerError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= data.len() {
            return Err(AigerError::UnexpectedEof);
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift > 63 {
            return Err(AigerError::InvalidHeader("delta encoding overflow".into()));
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_lit_str(s: &str, line: usize) -> Result<Literal, AigerError> {
    s.trim().parse::<u64>().map_err(|_| AigerError::Parse {
        line,
        message: format!("invalid literal: '{}'", s.trim()),
    })
}

fn parse_symbols(
    text: &str,
    inputs: &mut [AigerSymbol],
    latches: &mut [AigerLatch],
    outputs: &mut [AigerSymbol],
    bad: &mut [AigerSymbol],
    fairness: &mut [AigerSymbol],
    comments: &mut Vec<String>,
) {
    let mut in_comments = false;
    for line in text.lines() {
        let line = line.trim_end();
        if in_comments {
            comments.push(line.to_string());
            continue;
        }
        if line == "c" {
            in_comments = true;
            continue;
        }
        // Symbol table: [ilobf]<pos> <name>
        let (prefix, rest) = if line.len() >= 2 {
            (line.as_bytes()[0], &line[1..])
        } else {
            continue;
        };
        if let Some((pos_str, name)) = rest.split_once(' ') {
            if let Ok(idx) = pos_str.parse::<usize>() {
                match prefix {
                    b'i' if idx < inputs.len() => inputs[idx].name = Some(name.to_string()),
                    b'l' if idx < latches.len() => latches[idx].name = Some(name.to_string()),
                    b'o' if idx < outputs.len() => outputs[idx].name = Some(name.to_string()),
                    b'b' if idx < bad.len() => bad[idx].name = Some(name.to_string()),
                    b'f' if idx < fairness.len() => fairness[idx].name = Some(name.to_string()),
                    _ => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ASCII parser (.aag)
// ---------------------------------------------------------------------------

/// Parse an AIGER file in ASCII format (`.aag`).
///
/// # Errors
///
/// Returns [`AigerError::InvalidHeader`] if the file is empty or its header is
/// malformed, [`AigerError::Parse`] for a malformed body line,
/// [`AigerError::InvalidLiteral`] for a literal exceeding the declared maxvar,
/// and [`AigerError::DuplicateDefinition`] if a variable is defined twice.
pub fn parse_aag(source: &str) -> Result<AigerCircuit, AigerError> {
    let all_lines: Vec<&str> = source.lines().collect();
    if all_lines.is_empty() {
        return Err(AigerError::InvalidHeader("empty file".into()));
    }

    let header = parse_header(all_lines[0])?;
    if header.is_binary {
        return Err(AigerError::InvalidHeader(
            "expected ASCII format (aag), got binary (aig)".into(),
        ));
    }

    // Fail-closed: reject malicious/oversized header counts BEFORE any allocation
    // sized off those counts. body_len is the number of bytes after the header
    // line (a conservative bound on how many entries can actually be encoded).
    let body_len = source.len().saturating_sub(all_lines[0].len());
    validate_header(&header, body_len)?;

    let mut idx = 1usize; // Current line index (0-based, 0 = header)

    let take_line = |idx: &mut usize| -> Result<&str, AigerError> {
        if *idx >= all_lines.len() {
            return Err(AigerError::Parse {
                line: *idx + 1,
                message: "unexpected end of file".into(),
            });
        }
        let line = all_lines[*idx];
        *idx += 1;
        Ok(line)
    };

    // Inputs
    let mut inputs = Vec::with_capacity(capped_capacity(header.num_inputs));
    for _ in 0..header.num_inputs {
        let ln = idx + 1;
        let lit = parse_lit_str(take_line(&mut idx)?, ln)?;
        inputs.push(AigerSymbol { lit, name: None });
    }

    // Latches
    let mut latches = Vec::with_capacity(capped_capacity(header.num_latches));
    for _ in 0..header.num_latches {
        let ln = idx + 1;
        let line = take_line(&mut idx)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(AigerError::Parse {
                line: ln,
                message: "latch needs at least 2 fields: lit next".into(),
            });
        }
        let lit = parse_lit_str(parts[0], ln)?;
        let next = parse_lit_str(parts[1], ln)?;
        let reset = if parts.len() > 2 {
            parse_lit_str(parts[2], ln)?
        } else {
            0
        };
        latches.push(AigerLatch {
            lit,
            next,
            reset,
            name: None,
        });
    }

    // Outputs
    let mut outputs = Vec::with_capacity(capped_capacity(header.num_outputs));
    for _ in 0..header.num_outputs {
        let ln = idx + 1;
        let lit = parse_lit_str(take_line(&mut idx)?, ln)?;
        outputs.push(AigerSymbol { lit, name: None });
    }

    // Bad properties
    let mut bad = Vec::with_capacity(capped_capacity(header.num_bad));
    for _ in 0..header.num_bad {
        let ln = idx + 1;
        let lit = parse_lit_str(take_line(&mut idx)?, ln)?;
        bad.push(AigerSymbol { lit, name: None });
    }

    // Constraints
    let mut constraints = Vec::with_capacity(capped_capacity(header.num_constraints));
    for _ in 0..header.num_constraints {
        let ln = idx + 1;
        let lit = parse_lit_str(take_line(&mut idx)?, ln)?;
        constraints.push(AigerSymbol { lit, name: None });
    }

    // Justice
    let mut justice = Vec::with_capacity(capped_capacity(header.num_justice));
    for _ in 0..header.num_justice {
        let ln = idx + 1;
        let count = parse_lit_str(take_line(&mut idx)?, ln)?;
        // Fail-closed: a justice subcount must be physically representable in the
        // remaining lines (>= 1 line per literal) and within the absolute cap,
        // before reserving any capacity for it.
        if count > MAX_SECTION_ENTRIES {
            return Err(AigerError::InvalidHeader(format!(
                "justice subcount {count} exceeds maximum {MAX_SECTION_ENTRIES}"
            )));
        }
        let remaining_lines = (all_lines.len() as u64).saturating_sub(idx as u64);
        if count > remaining_lines {
            return Err(AigerError::Parse {
                line: ln,
                message: format!(
                    "justice subcount {count} exceeds {remaining_lines} remaining lines"
                ),
            });
        }
        let mut lits = Vec::with_capacity(capped_capacity(count));
        for _ in 0..count {
            let ln2 = idx + 1;
            let lit = parse_lit_str(take_line(&mut idx)?, ln2)?;
            lits.push(lit);
        }
        justice.push(AigerJustice { lits });
    }

    // Fairness
    let mut fairness = Vec::with_capacity(capped_capacity(header.num_fairness));
    for _ in 0..header.num_fairness {
        let ln = idx + 1;
        let lit = parse_lit_str(take_line(&mut idx)?, ln)?;
        fairness.push(AigerSymbol { lit, name: None });
    }

    // AND gates
    let mut ands = Vec::with_capacity(capped_capacity(header.num_ands));
    for _ in 0..header.num_ands {
        let ln = idx + 1;
        let line = take_line(&mut idx)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(AigerError::Parse {
                line: ln,
                message: "AND gate needs 3 fields: lhs rhs0 rhs1".into(),
            });
        }
        let lhs = parse_lit_str(parts[0], ln)?;
        let rhs0 = parse_lit_str(parts[1], ln)?;
        let rhs1 = parse_lit_str(parts[2], ln)?;
        ands.push(AigerAnd { lhs, rhs0, rhs1 });
    }

    // Symbol table and comments from remaining lines
    let mut comments = Vec::new();
    let remaining = all_lines[idx..].join("\n");
    parse_symbols(
        &remaining,
        &mut inputs,
        &mut latches,
        &mut outputs,
        &mut bad,
        &mut fairness,
        &mut comments,
    );

    let circuit = AigerCircuit {
        maxvar: header.maxvar,
        inputs,
        latches,
        outputs,
        ands,
        bad,
        constraints,
        justice,
        fairness,
        comments,
    };
    validate_circuit(&circuit)?;
    Ok(circuit)
}

// ---------------------------------------------------------------------------
// Binary parser (.aig)
// ---------------------------------------------------------------------------

/// Parse an AIGER file in binary format (`.aig`).
///
/// # Errors
///
/// Returns [`AigerError::InvalidHeader`] if the header is missing or malformed,
/// [`AigerError::UnexpectedEof`] if the delta-encoded AND-gate section ends
/// prematurely, [`AigerError::InvalidLiteral`] for a literal exceeding the
/// declared maxvar, and [`AigerError::DuplicateDefinition`] if a variable is
/// defined twice.
pub fn parse_aig(data: &[u8]) -> Result<AigerCircuit, AigerError> {
    let header_end = data
        .iter()
        .position(|&b| b == b'\n')
        .ok_or(AigerError::InvalidHeader("no newline after header".into()))?;

    let header_str = std::str::from_utf8(&data[..header_end])
        .map_err(|_| AigerError::InvalidHeader("header is not valid UTF-8".into()))?;
    let header = parse_header(header_str)?;
    if !header.is_binary {
        return Err(AigerError::InvalidHeader(
            "expected binary format (aig), got ASCII (aag)".into(),
        ));
    }

    // Fail-closed: reject malicious/oversized header counts BEFORE any allocation.
    // body_len bounds how many entries can physically be encoded after the header.
    let body_len = data.len().saturating_sub(header_end + 1);
    validate_header(&header, body_len)?;

    let mut pos = header_end + 1;

    // Read one ASCII line from the byte stream
    let read_line = |data: &[u8], pos: &mut usize| -> Result<String, AigerError> {
        let start = *pos;
        while *pos < data.len() && data[*pos] != b'\n' {
            *pos += 1;
        }
        if *pos >= data.len() && *pos == start {
            return Err(AigerError::UnexpectedEof);
        }
        let line = std::str::from_utf8(&data[start..*pos])
            .map_err(|_| AigerError::Parse {
                line: 0,
                message: "non-UTF-8 in ASCII section".into(),
            })?
            .to_string();
        if *pos < data.len() {
            *pos += 1; // Skip newline
        }
        Ok(line)
    };

    // Inputs are implicit in binary format: they consume NO body bytes, so the
    // per-section body check cannot bound them. Bound the synthesized-inputs loop
    // explicitly by both maxvar (already enforced by validate_header via
    // I + L + A <= M and M <= MAX_VARS) and the absolute capacity ceiling so a
    // header claiming billions of inputs cannot allocate gigabytes here.
    if header.num_inputs > MAX_INPUTS {
        return Err(AigerError::InvalidHeader(format!(
            "I (inputs) {} exceeds maximum {MAX_INPUTS} for binary format",
            header.num_inputs
        )));
    }
    let mut inputs: Vec<AigerSymbol> = Vec::new();
    inputs
        .try_reserve(capped_capacity(header.num_inputs))
        .map_err(|_| AigerError::InvalidHeader("input allocation failed".into()))?;
    for i in 0..header.num_inputs {
        inputs.push(AigerSymbol {
            lit: aiger_var2lit(i + 1),
            name: None,
        });
    }

    // Latches: one line per latch
    let mut latches = Vec::with_capacity(capped_capacity(header.num_latches));
    for i in 0..header.num_latches {
        let line = read_line(data, &mut pos)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        // Fail-closed: a blank latch line yields no fields; index parts[0] only
        // after confirming there is at least one field.
        if parts.is_empty() {
            return Err(AigerError::Parse {
                line: 0,
                message: "binary latch line is empty".into(),
            });
        }
        let next = parse_lit_str(parts[0], 0)?;
        let reset = if parts.len() > 1 {
            parse_lit_str(parts[1], 0)?
        } else {
            0
        };
        let lit = aiger_var2lit(header.num_inputs + i + 1);
        latches.push(AigerLatch {
            lit,
            next,
            reset,
            name: None,
        });
    }

    // Outputs
    let mut outputs = Vec::with_capacity(capped_capacity(header.num_outputs));
    for _ in 0..header.num_outputs {
        let line = read_line(data, &mut pos)?;
        outputs.push(AigerSymbol {
            lit: parse_lit_str(&line, 0)?,
            name: None,
        });
    }

    // Bad
    let mut bad = Vec::with_capacity(capped_capacity(header.num_bad));
    for _ in 0..header.num_bad {
        let line = read_line(data, &mut pos)?;
        bad.push(AigerSymbol {
            lit: parse_lit_str(&line, 0)?,
            name: None,
        });
    }

    // Constraints
    let mut constraints = Vec::with_capacity(capped_capacity(header.num_constraints));
    for _ in 0..header.num_constraints {
        let line = read_line(data, &mut pos)?;
        constraints.push(AigerSymbol {
            lit: parse_lit_str(&line, 0)?,
            name: None,
        });
    }

    // Justice
    let mut justice = Vec::with_capacity(capped_capacity(header.num_justice));
    for _ in 0..header.num_justice {
        let count_line = read_line(data, &mut pos)?;
        let count = parse_lit_str(&count_line, 0)?;
        // Fail-closed: bound the justice subcount before reserving. Each literal
        // is at least one byte in the remaining stream, so it cannot exceed the
        // remaining bytes; also enforce the absolute cap.
        if count > MAX_SECTION_ENTRIES {
            return Err(AigerError::InvalidHeader(format!(
                "justice subcount {count} exceeds maximum {MAX_SECTION_ENTRIES}"
            )));
        }
        let remaining_bytes = (data.len() as u64).saturating_sub(pos as u64);
        if count > remaining_bytes {
            return Err(AigerError::Parse {
                line: 0,
                message: format!(
                    "justice subcount {count} exceeds {remaining_bytes} remaining bytes"
                ),
            });
        }
        let mut lits = Vec::with_capacity(capped_capacity(count));
        for _ in 0..count {
            let jline = read_line(data, &mut pos)?;
            lits.push(parse_lit_str(&jline, 0)?);
        }
        justice.push(AigerJustice { lits });
    }

    // Fairness
    let mut fairness = Vec::with_capacity(capped_capacity(header.num_fairness));
    for _ in 0..header.num_fairness {
        let line = read_line(data, &mut pos)?;
        fairness.push(AigerSymbol {
            lit: parse_lit_str(&line, 0)?,
            name: None,
        });
    }

    // AND gates: binary delta encoding
    let mut ands = Vec::with_capacity(capped_capacity(header.num_ands));
    for i in 0..header.num_ands {
        // lhs index = I + L + i + 1; validate_header guarantees I + L + A <= M <=
        // MAX_VARS, so these additions cannot overflow u64, but guard anyway.
        let lhs_var = header
            .num_inputs
            .checked_add(header.num_latches)
            .and_then(|x| x.checked_add(i + 1))
            .ok_or_else(|| AigerError::InvalidHeader("AND-gate index overflow".into()))?;
        let lhs = aiger_var2lit(lhs_var);
        let delta0 = decode_delta(data, &mut pos)?;
        let delta1 = decode_delta(data, &mut pos)?;
        // Fail-closed: a delta larger than its base underflows. With
        // overflow-checks this panics; in release it wraps to a bogus huge
        // literal. Reject either way. (Spec bug #6.)
        let rhs0 = lhs.checked_sub(delta0).ok_or_else(|| AigerError::Parse {
            line: 0,
            message: format!("AND-gate delta0 {delta0} exceeds lhs {lhs}"),
        })?;
        let rhs1 = rhs0.checked_sub(delta1).ok_or_else(|| AigerError::Parse {
            line: 0,
            message: format!("AND-gate delta1 {delta1} exceeds rhs0 {rhs0}"),
        })?;
        ands.push(AigerAnd { lhs, rhs0, rhs1 });
    }

    // Symbol table and comments
    let mut comments = Vec::new();
    if pos < data.len() {
        if let Ok(remaining) = std::str::from_utf8(&data[pos..]) {
            parse_symbols(
                remaining,
                &mut inputs,
                &mut latches,
                &mut outputs,
                &mut bad,
                &mut fairness,
                &mut comments,
            );
        }
    }

    let circuit = AigerCircuit {
        maxvar: header.maxvar,
        inputs,
        latches,
        outputs,
        ands,
        bad,
        constraints,
        justice,
        fairness,
        comments,
    };
    validate_circuit(&circuit)?;
    Ok(circuit)
}

// ---------------------------------------------------------------------------
// Auto-detect format from file
// ---------------------------------------------------------------------------

/// Parse an AIGER file, auto-detecting ASCII (`.aag`) vs binary (`.aig`) format.
///
/// The format is chosen by the leading `aag`/`aig` tag, not the file extension.
///
/// # Errors
///
/// Returns [`AigerError::Io`] if the file cannot be read,
/// [`AigerError::InvalidHeader`] if it is too short or has no recognized tag,
/// and otherwise propagates the errors of [`parse_aag`] / [`parse_aig`].
pub fn parse_file(path: &Path) -> Result<AigerCircuit, AigerError> {
    let data = std::fs::read(path)?;
    if data.len() < 3 {
        return Err(AigerError::InvalidHeader("file too short".into()));
    }
    if data.starts_with(b"aag") {
        let source = std::str::from_utf8(&data)
            .map_err(|_| AigerError::InvalidHeader("ASCII AIGER file is not valid UTF-8".into()))?;
        parse_aag(source)
    } else if data.starts_with(b"aig") {
        parse_aig(&data)
    } else {
        Err(AigerError::InvalidHeader(
            "file does not start with 'aag' or 'aig'".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_circuit() {
        let c = parse_aag("aag 0 0 0 0 0\n").unwrap();
        assert_eq!(c.maxvar, 0);
        assert!(c.inputs.is_empty());
        assert!(c.ands.is_empty());
    }

    #[test]
    fn test_constant_false() {
        let c = parse_aag("aag 0 0 0 1 0\n0\n").unwrap();
        assert_eq!(c.outputs[0].lit, 0);
    }

    #[test]
    fn test_constant_true() {
        let c = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
        assert_eq!(c.outputs[0].lit, 1);
    }

    #[test]
    fn test_buffer() {
        let c = parse_aag("aag 1 1 0 1 0\n2\n2\n").unwrap();
        assert_eq!(c.inputs[0].lit, 2);
        assert_eq!(c.outputs[0].lit, 2);
    }

    #[test]
    fn test_inverter() {
        let c = parse_aag("aag 1 1 0 1 0\n2\n3\n").unwrap();
        assert_eq!(c.outputs[0].lit, 3);
        assert!(aiger_is_negated(3));
    }

    #[test]
    fn test_and_gate() {
        let c = parse_aag("aag 3 2 0 1 1\n2\n4\n6\n6 2 4\n").unwrap();
        assert_eq!(c.ands.len(), 1);
        assert_eq!(
            c.ands[0],
            AigerAnd {
                lhs: 6,
                rhs0: 2,
                rhs1: 4
            }
        );
    }

    #[test]
    fn test_half_adder_with_symbols() {
        let src = "aag 7 2 0 2 3\n2\n4\n6\n12\n6 13 15\n12 2 4\n14 3 5\ni0 x\ni1 y\no0 s\no1 c\nc\nhalf adder\n";
        let c = parse_aag(src).unwrap();
        assert_eq!(c.inputs.len(), 2);
        assert_eq!(c.outputs.len(), 2);
        assert_eq!(c.ands.len(), 3);
        assert_eq!(c.inputs[0].name.as_deref(), Some("x"));
        assert_eq!(c.inputs[1].name.as_deref(), Some("y"));
        assert_eq!(c.outputs[0].name.as_deref(), Some("s"));
        assert_eq!(c.outputs[1].name.as_deref(), Some("c"));
        assert_eq!(c.comments, vec!["half adder"]);
    }

    #[test]
    fn test_toggle_flip_flop() {
        let c = parse_aag("aag 1 0 1 2 0\n2 3\n2\n3\n").unwrap();
        assert_eq!(c.latches.len(), 1);
        assert_eq!(c.latches[0].lit, 2);
        assert_eq!(c.latches[0].next, 3);
    }

    #[test]
    fn test_latch_with_reset() {
        let c = parse_aag("aag 1 0 1 1 0\n2 3 1\n2\n").unwrap();
        assert_eq!(c.latches[0].reset, 1);
    }

    #[test]
    fn test_extended_bad() {
        let c = parse_aag("aag 3 2 1 0 0 1 0 0 0\n2\n4\n6 7\n6\n").unwrap();
        assert_eq!(c.bad.len(), 1);
        assert_eq!(c.bad[0].lit, 6);
    }

    #[test]
    fn test_binary_delta_decode() {
        let mut p = 0;
        assert_eq!(decode_delta(&[0x00], &mut p).unwrap(), 0);
        p = 0;
        assert_eq!(decode_delta(&[0x01], &mut p).unwrap(), 1);
        p = 0;
        assert_eq!(decode_delta(&[0x7f], &mut p).unwrap(), 127);
        p = 0;
        assert_eq!(decode_delta(&[0x80, 0x01], &mut p).unwrap(), 128);
        p = 0;
        assert_eq!(decode_delta(&[0x82, 0x02], &mut p).unwrap(), 258);
    }

    #[test]
    fn test_binary_and_gate() {
        // aig 3 2 0 1 1: two inputs, one output, one AND
        // AND: var3(lit6) = lit4 AND lit2, delta0=6-4=2, delta1=4-2=2
        let mut data = Vec::new();
        data.extend_from_slice(b"aig 3 2 0 1 1\n");
        data.extend_from_slice(b"6\n"); // output
        data.push(0x02); // delta0
        data.push(0x02); // delta1
        let c = parse_aig(&data).unwrap();
        assert_eq!(c.inputs.len(), 2);
        assert_eq!(c.ands.len(), 1);
        assert_eq!(
            c.ands[0],
            AigerAnd {
                lhs: 6,
                rhs0: 4,
                rhs1: 2
            }
        );
    }

    #[test]
    fn test_binary_with_latch() {
        let mut data = Vec::new();
        data.extend_from_slice(b"aig 1 0 1 2 0\n");
        data.extend_from_slice(b"3\n"); // latch next=3
        data.extend_from_slice(b"2\n"); // output 0
        data.extend_from_slice(b"3\n"); // output 1
        let c = parse_aig(&data).unwrap();
        assert_eq!(c.latches.len(), 1);
        assert_eq!(c.latches[0].lit, 2);
        assert_eq!(c.latches[0].next, 3);
        assert_eq!(c.outputs.len(), 2);
    }

    #[test]
    fn test_literal_helpers() {
        assert_eq!(aiger_var(6), 3);
        assert!(!aiger_is_negated(2));
        assert!(aiger_is_negated(3));
        assert_eq!(aiger_not(2), 3);
        assert_eq!(aiger_strip(3), 2);
        assert_eq!(aiger_var2lit(3), 6);
    }

    // -----------------------------------------------------------------------
    // Regression tests for fail-closed header validation (audit bugs #2/#6/#8/#13)
    //
    // Each asserts the parser DECLINES (returns Err) for a malformed/oversized
    // input BEFORE allocating, and runs instantly (no gigabyte allocation).
    // -----------------------------------------------------------------------

    /// Bug #2 (OOM, ASCII): a 24-byte file claiming ~4e9 inputs must be rejected
    /// by the per-section body-size check, not drive a 128GB Vec::with_capacity.
    #[test]
    fn reject_huge_input_count_ascii() {
        let src = "aag 4000000000 4000000000 0 0 0\n";
        let err = parse_aag(src).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_)),
            "expected InvalidHeader, got {err:?}"
        );
    }

    /// Bug #2 (OOM, binary): synthesized-inputs loop reads 0 body bytes; the
    /// MAX_INPUTS ceiling + maxvar bound must reject before allocating.
    #[test]
    fn reject_huge_input_count_binary() {
        let data = b"aig 4000000000 4000000000 0 0 0\n";
        let err = parse_aig(data).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_)),
            "expected InvalidHeader, got {err:?}"
        );
    }

    /// Bug #2 (OOM): huge AND-gate count with an empty body must be rejected by
    /// the per-section body check (each gate needs >= 2 delta bytes).
    #[test]
    fn reject_huge_and_count_binary() {
        // maxvar must cover I + L + A; both A and M are huge, body is empty.
        let data = b"aig 4000000000 0 0 0 4000000000\n";
        let err = parse_aig(data).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_)),
            "expected InvalidHeader, got {err:?}"
        );
    }

    /// Bug #8: maxvar above the hard ceiling must be rejected.
    #[test]
    fn reject_maxvar_above_ceiling() {
        let src = "aag 4000000000 0 0 0 0\n";
        let err = parse_aag(src).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_)),
            "expected InvalidHeader, got {err:?}"
        );
    }

    /// Bug #8: more defined variables (I + L + A) than maxvar must be rejected.
    #[test]
    fn reject_defined_exceeds_maxvar() {
        // M = 2 but I + L + A = 3 + 0 + 0 = 3 > 2.
        let src = "aag 2 3 0 0 0\n";
        let err = parse_aag(src).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_)),
            "expected InvalidHeader, got {err:?}"
        );
    }

    /// Bug #2 (OOM, justice): a justice subcount of ~4e9 with no body lines must
    /// be rejected before reserving the inner lits Vec.
    #[test]
    fn reject_huge_justice_subcount_ascii() {
        // 1 justice property; its first (and only) body line is the subcount.
        let src = "aag 0 0 0 0 0 0 0 1 0\n4000000000\n";
        let err = parse_aag(src).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_) | AigerError::Parse { .. }),
            "expected InvalidHeader/Parse, got {err:?}"
        );
    }

    /// Bug #2 (OOM, justice binary): same, via the binary path.
    #[test]
    fn reject_huge_justice_subcount_binary() {
        let mut data = Vec::new();
        data.extend_from_slice(b"aig 0 0 0 0 0 0 0 1 0\n");
        data.extend_from_slice(b"4000000000\n");
        let err = parse_aig(&data).unwrap_err();
        assert!(
            matches!(err, AigerError::InvalidHeader(_) | AigerError::Parse { .. }),
            "expected InvalidHeader/Parse, got {err:?}"
        );
    }

    /// Bug #13: a binary latch line with no fields (whitespace only) must not
    /// panic on `parts[0]`. The line is 2 bytes (" \n") so it passes the body-size
    /// check and reaches the latch-parse loop, where `split_whitespace()` yields
    /// an empty `parts`; the `is_empty()` guard must return Err, not index-panic.
    #[test]
    fn reject_blank_binary_latch_line() {
        let mut data = Vec::new();
        // aig 1 0 1 0 0: one latch, no inputs/outputs/ands.
        data.extend_from_slice(b"aig 1 0 1 0 0\n");
        data.extend_from_slice(b" \n"); // whitespace-only latch line (2 bytes)
        let err = parse_aig(&data).unwrap_err();
        assert!(
            matches!(err, AigerError::Parse { .. } | AigerError::UnexpectedEof),
            "expected Parse/UnexpectedEof, got {err:?}"
        );
    }

    /// Bug #6: an AND-gate delta larger than its base must not underflow/panic;
    /// it must be reported as a Parse error.
    #[test]
    fn reject_and_delta_underflow_binary() {
        // aig 1 1 0 0 1: I=1, L=0, A=1, M=1 -> wait, I + A = 2 > M=1 invalid.
        // Use M=2, I=1, A=1: lhs var = I + L + 1 = 2 -> lhs lit = 4.
        // delta0 = 5 (> lhs 4) triggers checked_sub underflow.
        let mut data = Vec::new();
        data.extend_from_slice(b"aig 2 1 0 0 1\n");
        data.push(0x05); // delta0 = 5 > lhs (4)
        data.push(0x00); // delta1 = 0
        let err = parse_aig(&data).unwrap_err();
        assert!(
            matches!(err, AigerError::Parse { .. }),
            "expected Parse, got {err:?}"
        );
    }

    /// Sanity: a legitimate small circuit still parses after validation is added.
    #[test]
    fn valid_circuit_still_parses() {
        let c = parse_aag("aag 3 2 0 1 1\n2\n4\n6\n6 2 4\n").unwrap();
        assert_eq!(c.ands.len(), 1);
        assert_eq!(c.inputs.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Regression tests for circuit-body validation (audit: literal-range +
    // duplicate-definition checks were documented but never enforced).
    // -----------------------------------------------------------------------

    /// Duplicate AND-gate definition: var 3 (lit 6) is defined twice with
    /// contradictory functions (6 = 2&2 and 6 = 3&3). Before validation this
    /// made the Tseitin transition relation UNSAT, so every engine vacuously
    /// reported 'unsat' (bogus SAFE) for a malformed file. Must be a parse
    /// error instead.
    #[test]
    fn reject_duplicate_and_definition() {
        let src = "aag 4 1 1 0 2 1\n2\n4 6\n4\n6 2 2\n6 3 3\n";
        let err = parse_aag(src).unwrap_err();
        assert!(
            matches!(err, AigerError::DuplicateDefinition(3)),
            "expected DuplicateDefinition(3), got {err:?}"
        );
    }

    /// A variable defined both as a latch and as an AND-gate output is a
    /// duplicate definition.
    #[test]
    fn reject_latch_redefined_as_and() {
        // M=3, I=1 (lit 2), L=1 (lit 4), A=1 (lhs 4 again).
        let src = "aag 3 1 1 0 1 1\n2\n4 2\n4\n4 2 2\n";
        let err = parse_aag(src).unwrap_err();
        assert!(
            matches!(err, AigerError::DuplicateDefinition(2)),
            "expected DuplicateDefinition(2), got {err:?}"
        );
    }

    /// A literal referencing a variable beyond the declared maxvar must be
    /// rejected (here: output lit 4 = var 2 with M = 1).
    #[test]
    fn reject_literal_out_of_range_ascii() {
        let err = parse_aag("aag 1 0 0 1 0\n4\n").unwrap_err();
        assert!(
            matches!(
                err,
                AigerError::InvalidLiteral {
                    literal: 4,
                    maxvar: 1
                }
            ),
            "expected InvalidLiteral, got {err:?}"
        );
    }

    /// Out-of-range latch NEXT literal in ASCII format (next lit 6 = var 3,
    /// M = 1).
    #[test]
    fn reject_latch_next_out_of_range_ascii() {
        let err = parse_aag("aag 1 0 1 0 0 1\n2 6\n2\n").unwrap_err();
        assert!(
            matches!(
                err,
                AigerError::InvalidLiteral {
                    literal: 6,
                    maxvar: 1
                }
            ),
            "expected InvalidLiteral, got {err:?}"
        );
    }

    /// Out-of-range latch next literal in BINARY format (next lit 6 = var 3,
    /// M = 1).
    #[test]
    fn reject_literal_out_of_range_binary() {
        let mut data = Vec::new();
        data.extend_from_slice(b"aig 1 0 1 1 0\n");
        data.extend_from_slice(b"6\n"); // latch next = lit 6 (var 3 > M)
        data.extend_from_slice(b"2\n"); // output
        let err = parse_aig(&data).unwrap_err();
        assert!(
            matches!(
                err,
                AigerError::InvalidLiteral {
                    literal: 6,
                    maxvar: 1
                }
            ),
            "expected InvalidLiteral, got {err:?}"
        );
    }
}
