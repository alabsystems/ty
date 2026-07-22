#![cfg(feature = "clean-cic")]
use tla_check::explicit_fixpoint_cert::{certify_explicit_state_spec, verify_explicit_state_cert};
use tla_check::Config;
#[test]
fn probe_time_63() {
    let spec = "---- MODULE C ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\nNext == x' = x + 1 /\\ x < 63\nSafety == x >= 0\n====\n";
    let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Safety\n").unwrap();
    let t0 = std::time::Instant::now();
    let cert = certify_explicit_state_spec(spec, &config).unwrap();
    eprintln!("mint: {:?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    assert!(verify_explicit_state_cert(&cert));
    eprintln!("verify: {:?}", t1.elapsed());
}
