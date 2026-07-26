// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::time::{Duration, Instant};

use tempfile::TempDir;
use tla_mc_core::TransitionSystem;
use tla_petri::{parser, PetriNet, PetriNetSystem};

const TRUST_CG_PARITY_ENV: &str = "TY_MCC_TRUST_CG_PETRI_PARITY";
const LANES: usize = 48;
const SAMPLE_STATES: usize = 256;
const REPEATS: usize = 64;

#[derive(Debug)]
struct ProfileResult {
    label: &'static str,
    elapsed: Duration,
    successor_count: usize,
    fingerprint_accumulator: u128,
}

struct EnvGuard {
    previous: Option<String>,
}

impl EnvGuard {
    fn set_trust_cg_parity(enabled: bool) -> Self {
        let previous = env::var(TRUST_CG_PARITY_ENV).ok();
        if enabled {
            tla_petri::env_guard::set_var(TRUST_CG_PARITY_ENV, "1");
        } else {
            tla_petri::env_guard::remove_var(TRUST_CG_PARITY_ENV);
        }
        Self { previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => tla_petri::env_guard::set_var(TRUST_CG_PARITY_ENV, value),
            None => tla_petri::env_guard::remove_var(TRUST_CG_PARITY_ENV),
        }
    }
}

#[test]
fn profile_current_vs_trust_cg_checked_successors() {
    if !env::var_os("TY_RUN_PETRI_SUCCESSOR_PROFILE").is_some_and(|value| value == "1") {
        eprintln!(
            "SKIP profile_current_vs_trust_cg_checked_successors: set \
             TY_RUN_PETRI_SUCCESSOR_PROFILE=1 to authorize the profiling campaign"
        );
        return;
    }
    let model_dir = write_lane_model(LANES);
    let net = parser::parse_pnml_dir(model_dir.path()).expect("parse generated lane PNML");
    let sample_markings = sample_lane_markings(LANES, SAMPLE_STATES);

    let baseline = measure_successors("current", &net, &sample_markings, REPEATS, false);
    let per_transition_checked = measure_successors(
        "trust_cg-per-transition-checked",
        &net,
        &sample_markings,
        REPEATS,
        true,
    );
    let all_transition_checked = measure_all_transition_checked_successors(
        "trust_cg-all-transition-flat-checked",
        &net,
        &sample_markings,
        REPEATS,
    );

    assert_eq!(
        baseline.successor_count,
        per_transition_checked.successor_count
    );
    assert_eq!(
        baseline.successor_count,
        all_transition_checked.successor_count
    );
    assert_eq!(
        baseline.fingerprint_accumulator, per_transition_checked.fingerprint_accumulator,
        "per-transition checked parity must preserve successor identities",
    );
    assert_eq!(
        baseline.fingerprint_accumulator, all_transition_checked.fingerprint_accumulator,
        "all-transition flat checked parity must preserve successor identities",
    );
    assert!(baseline.successor_count > 0);

    println!("{}", format_result(&baseline));
    println!("{}", format_result(&per_transition_checked));
    println!("{}", format_result(&all_transition_checked));
    println!(
        "trust_cg-per-transition-checked/current elapsed ratio: {:.3}",
        per_transition_checked.elapsed.as_secs_f64() / baseline.elapsed.as_secs_f64()
    );
    println!(
        "trust_cg-all-transition-flat-checked/current elapsed ratio: {:.3}",
        all_transition_checked.elapsed.as_secs_f64() / baseline.elapsed.as_secs_f64()
    );
    println!(
        "trust_cg-all-transition-flat-checked/trust-cg-per-transition-checked elapsed ratio: {:.3}",
        all_transition_checked.elapsed.as_secs_f64() / per_transition_checked.elapsed.as_secs_f64()
    );
}

fn measure_successors(
    label: &'static str,
    net: &PetriNet,
    sample_markings: &[Vec<u64>],
    repeats: usize,
    trust_cg_parity_enabled: bool,
) -> ProfileResult {
    let _env_guard = EnvGuard::set_trust_cg_parity(trust_cg_parity_enabled);
    let system = PetriNetSystem::new(net.clone());
    let states: Vec<_> = sample_markings
        .iter()
        .map(|marking| system.pack_marking(marking))
        .collect();

    let start = Instant::now();
    let mut successor_count = 0usize;
    let mut fingerprint_accumulator = 0u128;

    for _ in 0..repeats {
        for state in &states {
            let successors = black_box(&system).successors(black_box(state));
            successor_count += successors.len();
            for (transition, successor) in successors {
                fingerprint_accumulator = fingerprint_accumulator.wrapping_add(
                    successor
                        .fingerprint()
                        .rotate_left(transition.0 % u128::BITS)
                        ^ u128::from(transition.0),
                );
            }
        }
    }

    ProfileResult {
        label,
        elapsed: start.elapsed(),
        successor_count,
        fingerprint_accumulator,
    }
}

fn measure_all_transition_checked_successors(
    label: &'static str,
    net: &PetriNet,
    sample_markings: &[Vec<u64>],
    repeats: usize,
) -> ProfileResult {
    let system = PetriNetSystem::new(net.clone());

    let start = Instant::now();
    let mut successor_count = 0usize;
    let mut fingerprint_accumulator = 0u128;

    black_box(net)
        .trust_cg_profile_all_transition_checked_successors(
            black_box(sample_markings),
            repeats,
            |transition, successor| {
                successor_count += 1;
                let successor = black_box(&system).pack_marking(black_box(successor));
                fingerprint_accumulator = fingerprint_accumulator.wrapping_add(
                    successor
                        .fingerprint()
                        .rotate_left(transition.0 % u128::BITS)
                        ^ u128::from(transition.0),
                );
            },
        )
        .expect("all-transition flat checked successor profile must pass parity");

    ProfileResult {
        label,
        elapsed: start.elapsed(),
        successor_count,
        fingerprint_accumulator,
    }
}

fn format_result(result: &ProfileResult) -> String {
    let ns_per_successor = result.elapsed.as_nanos() as f64 / result.successor_count as f64;
    format!(
        "{}: {:?}, successors={}, ns/successor={:.1}, fingerprint_accumulator={:#034x}",
        result.label,
        result.elapsed,
        result.successor_count,
        ns_per_successor,
        result.fingerprint_accumulator
    )
}

fn sample_lane_markings(lanes: usize, samples: usize) -> Vec<Vec<u64>> {
    (0..samples)
        .map(|sample| {
            let bits = splitmix64(sample as u64);
            let mut marking = vec![0; lanes * 2];
            for lane in 0..lanes {
                let right_side = (bits.rotate_left((lane % u64::BITS as usize) as u32) & 1) == 1;
                let place = lane * 2 + usize::from(right_side);
                marking[place] = 1;
            }
            marking
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn write_lane_model(lanes: usize) -> TempDir {
    let dir = TempDir::new().expect("create temporary PNML directory");
    let mut pnml = String::from(
        r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="trust_cg-successor-profile" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p0">
"#,
    );

    for lane in 0..lanes {
        writeln!(
            pnml,
            r#"      <place id="P{lane}L"><initialMarking><text>1</text></initialMarking></place>"#,
        )
        .expect("write left place");
        writeln!(pnml, r#"      <place id="P{lane}R"/>"#).expect("write right place");
    }

    for lane in 0..lanes {
        writeln!(pnml, r#"      <transition id="T{lane}LR"/>"#).expect("write lr transition");
        writeln!(pnml, r#"      <transition id="T{lane}RL"/>"#).expect("write rl transition");
        writeln!(
            pnml,
            r#"      <arc id="A{lane}LLR" source="P{lane}L" target="T{lane}LR"/>"#,
        )
        .expect("write lr input");
        writeln!(
            pnml,
            r#"      <arc id="A{lane}RLR" source="T{lane}LR" target="P{lane}R"/>"#,
        )
        .expect("write lr output");
        writeln!(
            pnml,
            r#"      <arc id="A{lane}RRL" source="P{lane}R" target="T{lane}RL"/>"#,
        )
        .expect("write rl input");
        writeln!(
            pnml,
            r#"      <arc id="A{lane}LRL" source="T{lane}RL" target="P{lane}L"/>"#,
        )
        .expect("write rl output");
    }

    pnml.push_str(
        r#"    </page>
  </net>
</pnml>
"#,
    );
    fs::write(dir.path().join("model.pnml"), pnml).expect("write generated model.pnml");
    dir
}
