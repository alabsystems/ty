// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! End-to-end CLI test for certifying verification: `ty certify` then
//! `ty cert-check`, and that a forged certificate is rejected.

use std::path::PathBuf;
use std::process::Command;

const SPEC: &str = "---- MODULE Accumulator ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
const CFG: &str = "INIT Init\nNEXT Next\nINVARIANT Safety\n";

fn ty_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ty"))
}

#[test]
fn test_certify_then_cert_check_roundtrip() {
    let dir = std::env::temp_dir().join(format!("ty_cert_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("Accumulator.tla");
    let cfg = dir.join("Accumulator.cfg");
    let cert = dir.join("acc.ty.cert");
    std::fs::write(&spec, SPEC).unwrap();
    std::fs::write(&cfg, CFG).unwrap();

    // ty certify
    let certify = Command::new(ty_bin())
        .args(["certify"])
        .arg(&spec)
        .arg("--out")
        .arg(&cert)
        .output()
        .expect("run ty certify");

    // Skip on a non-`ay` build (certify exits 2 with a "requires ay" message).
    if !certify.status.success() {
        let stderr = String::from_utf8_lossy(&certify.stderr);
        if stderr.contains("requires the `ay` feature") {
            eprintln!("skipping: `ty` built without the `ay` feature");
            return;
        }
        panic!(
            "ty certify failed: status={:?} stderr={stderr}",
            certify.status
        );
    }
    assert!(cert.exists(), "certify must write the certificate");

    // ty cert-check on the genuine certificate -> VERIFIED, exit 0.
    let good = Command::new(ty_bin())
        .args(["cert-check"])
        .arg(&cert)
        .output()
        .expect("run ty cert-check");
    assert!(
        good.status.success(),
        "genuine certificate must verify; stdout={} stderr={}",
        String::from_utf8_lossy(&good.stdout),
        String::from_utf8_lossy(&good.stderr)
    );

    // Forge the certificate (flip the invariant) -> REJECTED, exit 1.
    let forged = dir.join("forged.ty.cert");
    let body = std::fs::read_to_string(&cert).unwrap();
    std::fs::write(&forged, body.replace("x >= 0", "x >= 5")).unwrap();
    let bad = Command::new(ty_bin())
        .args(["cert-check"])
        .arg(&forged)
        .output()
        .expect("run ty cert-check on forged");
    assert!(
        !bad.status.success(),
        "a forged certificate must be REJECTED (nonzero exit); stdout={}",
        String::from_utf8_lossy(&bad.stdout)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Skip helper: true if this `ty` was built without the `ay` feature (certify unavailable).
fn is_non_ay(out: &std::process::Output) -> bool {
    !out.status.success()
        && String::from_utf8_lossy(&out.stderr).contains("requires the `ay` feature")
}

/// The well-formedness gate (`cert_gate`) must DECLINE a spec `ty check` rejects as ill-formed
/// (here: a duplicate `Safety` definition — `find_op` would silently keep the first, certifying a
/// DIFFERENT spec than check reads). A decline is fail-closed, so never a false safe.
#[test]
fn test_certify_declines_duplicate_definition() {
    let dir = std::env::temp_dir().join(format!("ty_dupdef_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("Dup.tla");
    let cfg = dir.join("Dup.cfg");
    std::fs::write(
        &spec,
        "---- MODULE Dup ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\n\
         Next == x' = x\nSafety == x = 0\nSafety == x = 99\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Safety\n").unwrap();

    let out = Command::new(ty_bin())
        .args(["certify"])
        .arg(&spec)
        .arg("--out")
        .arg(dir.join("d.cert"))
        .output()
        .expect("run ty certify");
    if is_non_ay(&out) {
        eprintln!("skipping: non-`ay` build");
        return;
    }
    assert!(
        !out.status.success(),
        "certify must DECLINE a duplicate-definition spec; stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !dir.join("d.cert").exists(),
        "no certificate may be written for an ill-formed spec"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The SHARED gate must also cover `certify-liveness` — the entry point an earlier per-lane
/// version of this fix left ungated. A duplicate `Measure` definition must decline here too.
#[test]
fn test_certify_liveness_declines_duplicate_definition() {
    let dir = std::env::temp_dir().join(format!("ty_dupliv_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("DupL.tla");
    let cfg = dir.join("DupL.cfg");
    std::fs::write(
        &spec,
        "---- MODULE DupL ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\n\
         Next == x < 3 /\\ x' = x + 1\nReach == <>(x = 3)\nMeasure == 3 - x\nMeasure == 999\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\n").unwrap();

    let out = Command::new(ty_bin())
        .args(["certify-liveness"])
        .arg(&spec)
        .args(["--property", "Reach", "--measure", "Measure", "--out"])
        .arg(dir.join("l.cert"))
        .output()
        .expect("run ty certify-liveness");
    if is_non_ay(&out) {
        eprintln!("skipping: non-`ay` build");
        return;
    }
    assert!(
        !out.status.success(),
        "certify-liveness must DECLINE a duplicate-definition spec (shared gate); stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The config-operator gate must decline a `.cfg` naming a CONSTRAINT operator absent from the
/// module (a hard `ty check` failure).
#[test]
fn test_certify_declines_undefined_config_operator() {
    let dir = std::env::temp_dir().join(format!("ty_cfgop_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("S.tla");
    let cfg = dir.join("S.cfg");
    std::fs::write(
        &spec,
        "---- MODULE S ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\n\
         Next == x' = x\nSafety == x = 0\n====\n",
    )
    .unwrap();
    std::fs::write(
        &cfg,
        "INIT Init\nNEXT Next\nINVARIANT Safety\nCONSTRAINT Nope\n",
    )
    .unwrap();

    let out = Command::new(ty_bin())
        .args(["certify"])
        .arg(&spec)
        .arg("--out")
        .arg(dir.join("s.cert"))
        .output()
        .expect("run ty certify");
    if is_non_ay(&out) {
        eprintln!("skipping: non-`ay` build");
        return;
    }
    assert!(
        !out.status.success(),
        "certify must DECLINE a config naming an undefined CONSTRAINT operator; stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Module-binding parity (SOUNDNESS — the confirmed false safe): a file whose file-stem module
/// (what `ty check` verifies) is NOT the first module (what the certify lanes bind) must be
/// certified BY BINDING THE STEM MODULE — never by certifying a different (earlier, safe) module.
/// Here `Inner` (safe: x stays 0) is first, `Multi` (= filename, reachably VIOLATED: Init x=5 with
/// Inv x=0) is second. `ty check` sees Multi VIOLATED, so certify must bind Multi, find it violated,
/// and DECLINE with NO certificate — NOT certify the safe `Inner`. (Previously this DECLINED on the
/// binding mismatch itself; now it binds the right module and declines because that module fails.)
#[test]
fn test_certify_declines_multi_module_binding_mismatch() {
    let dir = std::env::temp_dir().join(format!("ty_modbind_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("Multi.tla");
    let cfg = dir.join("Multi.cfg");
    std::fs::write(
        &spec,
        "---- MODULE Inner ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\nNext == x' = x\n\
         Inv == x = 0\n====\n\
         ---- MODULE Multi ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 5\nNext == x' = x\n\
         Inv == x = 0\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Inv\n").unwrap();

    let out = Command::new(ty_bin())
        .args(["certify"])
        .arg(&spec)
        .arg("--out")
        .arg(dir.join("m.cert"))
        .output()
        .expect("run ty certify");
    if is_non_ay(&out) {
        eprintln!("skipping: non-`ay` build");
        return;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "certify must DECLINE when the stem module `Multi` is reachably violated (binding it, not \
         the safe `Inner`); stdout={stdout} stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The decisive NO-FALSE-SAFE assertion: it must NOT have certified the safe `Inner`.
    assert!(
        !stdout.contains("CERTIFIED:") && !stdout.contains("KERNEL-CERTIFIED"),
        "certify must NOT emit a CERTIFIED verdict for a violated stem module; stdout={stdout}"
    );
    assert!(
        !dir.join("m.cert").exists(),
        "no certificate may be written when the bound (stem) module is violated"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The OTHER direction of module-binding parity (the over-decline this change fixes): a file whose
/// file-stem module is NOT first but IS SAFE must now CERTIFY — by binding the stem module, and by
/// embedding a self-contained `spec_src` whose single module IS that stem module (so every re-check
/// re-lowers exactly what `ty check` verified). Here `Helper` (safe, 4-state) is first, `Main`
/// (= filename, safe, distinctive 6-state counter) is second.
#[test]
fn test_certify_binds_nonfirst_stem_module_when_safe() {
    let dir = std::env::temp_dir().join(format!("ty_modbind_ok_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("Main.tla");
    let cfg = dir.join("Main.cfg");
    std::fs::write(
        &spec,
        "---- MODULE Helper ----\nEXTENDS Naturals\nVARIABLE y\nInitH == y = 0\n\
         NextH == (y < 3 /\\ y' = y + 1) \\/ (y = 3 /\\ y' = 0)\nInvH == y >= 0\n====\n\
         ---- MODULE Main ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\n\
         Next == (x < 5 /\\ x' = x + 1) \\/ (x = 5 /\\ x' = 0)\nInv == x >= 0\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Inv\n").unwrap();
    let cert = dir.join("main.cert");
    let out = Command::new(ty_bin())
        .args(["certify"])
        .arg(&spec)
        .arg("--out")
        .arg(&cert)
        .output()
        .expect("run ty certify");
    if is_non_ay(&out) {
        eprintln!("skipping: non-`ay` build");
        return;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "certify must CERTIFY a safe non-first stem module (over-decline fixed); stdout={stdout} \
         stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        cert.exists(),
        "a certificate must be written for the safe stem module"
    );
    // Binding proof: the embedded self-contained `spec_src` is the STEM module `Main`, and the
    // dropped, unrelated `Helper` is absent. (A dependency the stem EXTENDS would be inlined; an
    // independent sibling is not.)
    let cert_json = std::fs::read_to_string(&cert).unwrap();
    assert!(
        cert_json.contains("MODULE Main"),
        "the certificate must embed the stem module `Main`; cert={cert_json}"
    );
    assert!(
        !cert_json.contains("MODULE Helper") && !cert_json.contains("InvH"),
        "the certificate must NOT embed the unrelated first module `Helper`; cert={cert_json}"
    );
    // And it re-checks independently.
    let recheck = Command::new(ty_bin())
        .args(["cert-check"])
        .arg(&cert)
        .output()
        .expect("run ty cert-check");
    assert!(
        recheck.status.success(),
        "the stem-module certificate must re-check; stdout={} stderr={}",
        String::from_utf8_lossy(&recheck.stdout),
        String::from_utf8_lossy(&recheck.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// SOUNDNESS: `ty prove` (the 5th certificate-minting entry point) must run the SAME shared
/// well-formedness gate as `ty certify` — it must DECLINE a duplicate-definition spec rather than
/// announcing PROVED and writing a re-checkable certificate for a spec `ty check` refuses.
#[test]
fn test_prove_declines_duplicate_definition() {
    let dir = std::env::temp_dir().join(format!("ty_provedup_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let spec = dir.join("Dup.tla");
    let cfg = dir.join("Dup.cfg");
    std::fs::write(
        &spec,
        "---- MODULE Dup ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\n\
         Next == x' = x\nSafety == x = 0\nSafety == x = 99\n====\n",
    )
    .unwrap();
    std::fs::write(&cfg, "INIT Init\nNEXT Next\nINVARIANT Safety\n").unwrap();
    let out = Command::new(ty_bin())
        .args(["prove"])
        .arg(&spec)
        .arg("--out")
        .arg(dir.join("d.cert"))
        .output()
        .expect("run ty prove");
    if is_non_ay(&out) {
        eprintln!("skipping: non-`ay` build");
        return;
    }
    assert!(
        !out.status.success(),
        "ty prove must DECLINE a duplicate-definition spec (shared gate); stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !dir.join("d.cert").exists(),
        "ty prove must not write a certificate for an ill-formed spec"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
