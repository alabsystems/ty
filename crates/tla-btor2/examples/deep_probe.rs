// One-off probe: deep_overflow_bug through the BMC lane with a raised read cap.
fn main() {
    let src = std::fs::read_to_string(
        std::env::var("HOME").unwrap()
            + "/hwmcc/benchmarks/wordlevel/array/2020/mann/deep_overflow_bug.btor2",
    )
    .unwrap();
    let prog = tla_btor2::parse_btor2(&src).unwrap();
    let t = std::time::Instant::now();
    let out = tla_btor2::check_array_bmc(
        &prog,
        &tla_btor2::ArrayBmcConfig {
            max_reads: 8192,
            time_budget: Some(std::time::Duration::from_secs(600)),
            ..Default::default()
        },
    );
    println!("[deep-bmc] {:?} in {:.1}s", out, t.elapsed().as_secs_f64());
}
