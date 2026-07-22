// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-backend soundness for native TLA+ string concatenation (`\o` /
//! `StrConcat`) folding.
//!
//! The trust-ir->trust-IR lowering folds `\o` on two compile-time-known string
//! scalars to the interned `NameId` of the concatenated string (see
//! `tla_ir::lower`'s `lower_string_concat_const`), bit-identical to the
//! bytecode VM's `Value::string(a + b)`. These tests pin that the native
//! result agrees with the interpreter on a spec where `\o` drives a genuinely
//! VIOLATED invariant at a reachable state.
//!
//! The interpreter/compiled-BFS test exercises the bytecode-VM/interpreter and
//! the compiled-BFS path and asserts both find the violation. The native-parity
//! test additionally drives the trust-codegen native backend and asserts (a)
//! it finds the same violation and (b) every action instance compiled natively
//! (`actions_compiled == actions_total > 0`) -- i.e. the `Build` action that
//! uses `\o` lowered through the native string-concat fold rather than falling
//! back to the VM.

mod common;

use std::path::{Path, PathBuf};
use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

const SPEC_NAME: &str = "StrConcatNativeFold.tla";
const CFG_NAME: &str = "StrConcatNativeFold.cfg";
const VIOLATED_INVARIANT: &str = "LabelNeverKey42";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

fn read_fixture(name: &str) -> String {
    let path = repo_root().join("test_specs").join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn parse_config() -> Config {
    Config::parse(&read_fixture(CFG_NAME))
        .unwrap_or_else(|errors| panic!("failed to parse {CFG_NAME}: {errors:?}"))
}

/// The interpreter (compiled BFS force-disabled) and the auto-activated
/// compiled-BFS path must both find the `LabelNeverKey42` violation that
/// `\o` ("key_" \o "42" = "key_42") produces at the reachable phase-1 state.
///
/// This proves the fixture is genuinely violated and that `\o` on string
/// constants is handled end-to-end by the default pipeline. Verdict parity
/// here is the runnable parity proxy on an LLVM-less box (the trust-codegen
/// native backend that exercises the trust-ir fold requires LLVM; see the
/// `trust-cg`-gated test below).
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn test_strconcat_fold_interpreter_and_compiled_bfs_agree_on_violation() {
    let module = common::parse_module(&read_fixture(SPEC_NAME));
    let config = parse_config();

    // Interpreter path: force compiled BFS off.
    let interp = {
        let _no_compiled = common::EnvVarGuard::set("TY_NO_COMPILED_BFS", Some("1"));
        let _no_jit = common::EnvVarGuard::remove("TY_JIT");
        clear_for_test_reset();
        check_module(&module, &config)
    };

    // Compiled-BFS path: let it auto-activate.
    let compiled = {
        let _no_disable = common::EnvVarGuard::remove("TY_NO_COMPILED_BFS");
        let _no_jit = common::EnvVarGuard::remove("TY_JIT");
        clear_for_test_reset();
        check_module(&module, &config)
    };

    let interp_inv = match &interp {
        CheckResult::InvariantViolation { invariant, .. } => invariant.clone(),
        other => panic!("interpreter run should violate {VIOLATED_INVARIANT}, got {other:?}"),
    };
    let compiled_inv = match &compiled {
        CheckResult::InvariantViolation { invariant, .. } => invariant.clone(),
        other => panic!("compiled-BFS run should violate {VIOLATED_INVARIANT}, got {other:?}"),
    };

    assert_eq!(
        interp_inv, VIOLATED_INVARIANT,
        "interpreter should violate {VIOLATED_INVARIANT}"
    );
    assert_eq!(
        interp_inv, compiled_inv,
        "interpreter and compiled-BFS must agree on the violated invariant"
    );
}

/// Trust-codegen native parity: the default backend and the trust-codegen
/// native backend must agree that `\o` drives the `LabelNeverKey42`
/// violation, AND every action instance must compile natively
/// (`actions_compiled == actions_total > 0`). The `Build` action's
/// `"key_" \o "42"` lowers through the native string-concat fold; if that fold
/// were missing it would fall back to the VM and `actions_compiled` would drop
/// below `actions_total`, failing this assertion.
///
/// The trust-codegen native backend is opt-in at runtime via `TY_TRUST_CG=1`
/// (set by this test). It uses trust-codegen's own native backend rather than
/// shelling out to LLVM, so it runs in any environment.
#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn test_strconcat_fold_trust_cg_native_parity() {
    use common::EnvVarGuard;
    use tla_check::ModelChecker;

    fn run(trust_cg: bool) -> (String, Option<(usize, usize)>) {
        let _trust_cg = EnvVarGuard::set("TY_TRUST_CG", trust_cg.then_some("1"));
        let _no_compiled = EnvVarGuard::remove("TY_NO_COMPILED_BFS");
        let _auto_por = EnvVarGuard::set("TY_AUTO_POR", Some("0"));
        clear_for_test_reset();

        let module = common::parse_module(&read_fixture(SPEC_NAME));
        let config = parse_config();
        let mut checker = ModelChecker::new(&module, &config);
        let result = checker.check();
        let invariant = match result {
            CheckResult::InvariantViolation { invariant, .. } => invariant,
            other => panic!(
                "{} backend should violate {VIOLATED_INVARIANT}, got {other:?}",
                if trust_cg { "trust-cg" } else { "default" }
            ),
        };
        (invariant, checker.trust_cg_action_coverage_for_testing())
    }

    let (baseline_inv, _) = run(false);
    let (trust_cg_inv, coverage) = run(true);

    assert_eq!(
        baseline_inv, VIOLATED_INVARIANT,
        "default backend should violate {VIOLATED_INVARIANT}"
    );
    assert_eq!(
        baseline_inv, trust_cg_inv,
        "default and trust-cg backends must agree on the violated invariant"
    );

    let (compiled, total) = coverage.expect("trust-cg run should record action coverage");
    assert!(
        compiled > 0 && compiled == total,
        "every trust-codegen action instance should compile natively (the `\\o` fold must \
         not fall back to the VM), got {compiled}/{total}"
    );
}
