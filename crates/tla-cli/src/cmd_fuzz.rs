// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty fuzz` — standing differential fuzzer.
//!
//! Generates well-formed random TLA+ specs (deterministically, from a seed) and runs
//! cross-mode verdict parity (`ty parity`) on each, hunting for an engine that diverges
//! — the soundness-bug signal. Any divergent spec is kept as a reproducible fixture.
//! A self-service version of the "differential testing as a standing process" the
//! roadmap calls for: a user points it at a seed/count and either gets silence (good)
//! or a minimal divergent spec to file.
//!
//! Exits 0 when no divergence is found, 1 when at least one spec diverged.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct ParityJson {
    status: String,
}

/// Run `ty fuzz`.
pub(crate) fn cmd_fuzz(seed: u64, count: usize, keep_dir: Option<&Path>) -> Result<()> {
    let exe = std::env::current_exe().context("could not resolve the `ty` executable path")?;
    let work = std::env::temp_dir().join(format!("ty-fuzz-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).with_context(|| "create fuzz workdir")?;
    if let Some(d) = keep_dir {
        std::fs::create_dir_all(d).with_context(|| "create --keep dir")?;
    }

    println!("Differential fuzz: seed={seed}, {count} spec(s)\n");
    let mut divergences = 0usize;
    let mut errors = 0usize;
    let mut parity = 0usize;

    for i in 0..count {
        let mut rng = splitmix64_seed(seed, i as u64);
        let (tla_src, cfg_src) = gen_spec(&mut rng);
        let tla = work.join(format!("Fuzz{i}.tla"));
        let cfg = work.join(format!("Fuzz{i}.cfg"));
        std::fs::write(&tla, &tla_src).ok();
        std::fs::write(&cfg, &cfg_src).ok();

        let status = run_parity_status(&exe, &tla, &cfg);
        match status.as_str() {
            "parity" => parity += 1,
            "disagreement" => {
                divergences += 1;
                let dest_dir = keep_dir.unwrap_or(&work);
                let kept_tla = dest_dir.join(format!("DIVERGENCE-seed{seed}-i{i}.tla"));
                let kept_cfg = dest_dir.join(format!("DIVERGENCE-seed{seed}-i{i}.cfg"));
                let _ = std::fs::write(&kept_tla, &tla_src);
                let _ = std::fs::write(&kept_cfg, &cfg_src);
                println!(
                    "  DIVERGENCE (spec #{i}, reproduce: `ty fuzz --seed {seed} --count {}` or run\n\
                     \x20  `ty parity {}`)\n  kept -> {}",
                    i + 1,
                    kept_tla.display(),
                    kept_tla.display()
                );
            }
            _ => errors += 1, // inconclusive / generation produced a non-comparable spec
        }
    }

    // Clean the scratch (kept fixtures live in --keep dir if requested).
    if keep_dir.is_some() {
        let _ = std::fs::remove_dir_all(&work);
    }

    println!(
        "\n{count} specs fuzzed: {parity} parity, {divergences} DIVERGENCE, {errors} \
         inconclusive/non-comparable."
    );
    if divergences > 0 {
        println!("SOUNDNESS ALERT — {divergences} spec(s) diverged across engines.");
        if keep_dir.is_none() {
            println!("(re-run with --keep <dir> to save the divergent specs.)");
        }
        std::process::exit(1);
    }
    println!("CLEAN — every generated spec agreed across all engines.");
    Ok(())
}

/// Run `ty parity <tla> --config <cfg> --format json` and return its `status`.
fn run_parity_status(exe: &Path, tla: &Path, cfg: &Path) -> String {
    let child = Command::new(exe)
        .arg("parity")
        .arg(tla)
        .arg("--config")
        .arg(cfg)
        .arg("--format")
        .arg("json")
        .arg("--timeout")
        .arg("20")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut out = String::new();
    if let Ok(mut c) = child {
        if let Some(mut so) = c.stdout.take() {
            let _ = so.read_to_string(&mut out);
        }
        let _ = c.wait();
    }
    serde_json::from_str::<ParityJson>(out.trim())
        .map(|p| p.status)
        .unwrap_or_else(|_| "error".to_string())
}

/// Deterministic splitmix64, seeded by (seed, index) so each spec is reproducible.
fn splitmix64_seed(seed: u64, index: u64) -> u64 {
    seed ^ index.wrapping_mul(0x9E3779B97F4A7C15)
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn range(state: &mut u64, lo: u64, hi: u64) -> u64 {
    debug_assert!(hi > lo);
    lo + next_u64(state) % (hi - lo)
}

/// Generate a well-formed bounded-counter spec + config. The structure is fixed so it
/// always parses + has reachable states; the numeric bounds, action set, and invariant
/// threshold vary, producing safe / violation / deadlock verdicts that the engines must
/// agree on.
fn gen_spec(state: &mut u64) -> (String, String) {
    let k = range(state, 1, 4) as usize; // 1..=3 variables
    let names: Vec<String> = (0..k).map(|i| format!("x{i}")).collect();
    let bounds: Vec<u64> = (0..k).map(|_| range(state, 2, 7)).collect(); // each var caps at 2..=6

    let mut s = String::from("---- MODULE Fuzz ----\nEXTENDS Naturals\n");
    if k == 1 {
        s.push_str(&format!("VARIABLE {}\n", names[0]));
    } else {
        s.push_str(&format!("VARIABLES {}\n", names.join(", ")));
    }
    // Init: all zero.
    let init: Vec<String> = names.iter().map(|n| format!("{n} = 0")).collect();
    s.push_str(&format!("Init == {}\n", init.join(" /\\ ")));

    // One increment action per variable.
    let mut action_names = Vec::new();
    for i in 0..k {
        let an = format!("A{i}");
        action_names.push(an.clone());
        let others: Vec<&str> = names
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, n)| n.as_str())
            .collect();
        let unchanged = if others.is_empty() {
            String::new()
        } else {
            format!(" /\\ UNCHANGED <<{}>>", others.join(", "))
        };
        s.push_str(&format!(
            "{an} == {n} < {b} /\\ {n}' = {n} + 1{unchanged}\n",
            n = names[i],
            b = bounds[i],
        ));
    }
    s.push_str(&format!("Next == {}\n", action_names.join(" \\/ ")));

    // Invariant on x0: threshold around its bound so we get both safe and violation.
    let c = range(state, 1, bounds[0] + 2);
    s.push_str(&format!("Inv == {} <= {c}\n", names[0]));
    s.push_str("====\n");

    let cfg = "INIT Init\nNEXT Next\nINVARIANT Inv\n".to_string();
    (s, cfg)
}
