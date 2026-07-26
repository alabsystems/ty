// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Differential SOUNDNESS GATE for the symbolic-colored StateSpace engine
//! (`crate::symbolic_colored`).
//!
//! For every colored net SMALL enough that `unfold_to_pt` succeeds, this battery
//! builds the EXPLICITLY unfolded P/T net, runs the TRUSTED P/T MDD StateSpace
//! on it (`build_sound_dd_spec` → `MddNet::state_space_metrics`), and asserts the
//! symbolic-colored engine's four metrics EQUAL the unfolded engine's, EXACTLY.
//! A proptest over randomly-generated small colored nets in the v1 sub-class
//! gives breadth. Any disagreement ⇒ the engine is buggy ⇒ the sub-class
//! narrows; a symbolic count is NEVER published if it differs from the oracle.

#![cfg(feature = "dd-backend")]

use crate::hlpnml::{parse_hlpnml, ColoredNet};
use crate::symbolic_colored::{
    build_colored_mdd_net, colored_state_space_metrics, colored_state_space_metrics_quantified,
    SymbolicColoredError,
};

const HEAVY_CAMPAIGN_TOTAL_BUDGET: std::time::Duration = std::time::Duration::from_secs(15 * 60);
static HEAVY_CAMPAIGN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
static HEAVY_CAMPAIGN_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

struct HeavyCampaignGuard {
    _serial: std::sync::MutexGuard<'static, ()>,
    deadline: std::time::Instant,
    test_name: &'static str,
}

impl HeavyCampaignGuard {
    fn deadline_with_cap(&self, seconds: u64) -> std::time::Instant {
        let now = std::time::Instant::now();
        assert!(
            now < self.deadline,
            "{}: shared 15-minute symbolic campaign budget exhausted during setup",
            self.test_name
        );
        std::cmp::min(
            now.checked_add(std::time::Duration::from_secs(seconds))
                .expect("per-test symbolic deadline should fit in Instant"),
            self.deadline,
        )
    }
}

fn heavy_campaign_guard(test_name: &'static str) -> Option<HeavyCampaignGuard> {
    if !std::env::var_os("TY_RUN_HEAVY_SYMBOLIC_COLORED_TESTS").is_some_and(|value| value == "1") {
        eprintln!(
            "SKIP {test_name}: set TY_RUN_HEAVY_SYMBOLIC_COLORED_TESTS=1 to authorize \
             the serialized, 15-minute symbolic-colored campaign"
        );
        return None;
    }

    let serial = HEAVY_CAMPAIGN_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let start = *HEAVY_CAMPAIGN_START.get_or_init(std::time::Instant::now);
    let deadline = start
        .checked_add(HEAVY_CAMPAIGN_TOTAL_BUDGET)
        .expect("symbolic campaign deadline should fit in Instant");
    assert!(
        std::time::Instant::now() < deadline,
        "{test_name}: shared 15-minute symbolic campaign budget exhausted while waiting for \
         another heavy test"
    );

    Some(HeavyCampaignGuard {
        _serial: serial,
        deadline,
        test_name,
    })
}

/// The four StateSpace metrics, normalized for comparison across engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Metrics {
    states: u128,
    edges: u128,
    max_token_in_place: u64,
    max_token_sum: u64,
}

/// The TRUSTED ORACLE: explicitly unfold `colored` to a P/T net, then run an
/// EXPLICIT-BFS four-metric StateSpace over the unfolded net (the same firing
/// rule + metric semantics as the production `StateSpaceObserver` /
/// `tla_dd::bfs_full_metrics`). This is ground truth independent of the
/// symbolic engine and of any DD/LP admission gate, so it decides every net the
/// symbolic engine decides (as long as the reachable set is small enough to
/// enumerate, which the proptest sizing guarantees). Returns `None` only if the
/// net does not unfold at all.
fn oracle_metrics(colored: &ColoredNet) -> Option<Metrics> {
    let unfolded = crate::unfold::unfold_to_pt(colored).ok()?;
    Some(bfs_metrics(&unfolded.net))
}

/// Explicit-BFS four-metric StateSpace over a P/T net. Enumerates the reachable
/// set with the standard `is_enabled` / `fire` rule; `edges` = Σ enabled
/// firings over reachable markings; `max_token_in_place` / `max_token_sum` over
/// reachable markings. Bounded by the small reachable sets the proptest /
/// fixtures generate.
fn bfs_metrics(net: &crate::petri_net::PetriNet) -> Metrics {
    use crate::petri_net::TransitionIdx;
    use std::collections::HashSet;
    let init = net.initial_marking.clone();
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    seen.insert(init.clone());
    let mut frontier = vec![init.clone()];
    let mut edges: u128 = 0;
    let mut max_in_place: u64 = init.iter().copied().max().unwrap_or(0);
    let mut max_sum: u64 = init.iter().sum();
    while let Some(m) = frontier.pop() {
        for tid in 0..net.num_transitions() {
            let t = TransitionIdx(tid as u32);
            if !net.is_enabled(&m, t) {
                continue;
            }
            // Fire (P/T firing; cannot underflow since enabled). The unfolded
            // conserving nets stay bounded, so this enumerates a finite set.
            let next = net.fire(&m, t).expect("enabled transition fires");
            edges += 1;
            if seen.insert(next.clone()) {
                let s: u64 = next.iter().sum();
                let mxp = next.iter().copied().max().unwrap_or(0);
                max_sum = max_sum.max(s);
                max_in_place = max_in_place.max(mxp);
                frontier.push(next);
            }
        }
    }
    Metrics {
        states: seen.len() as u128,
        edges,
        max_token_in_place: max_in_place,
        max_token_sum: max_sum,
    }
}

/// The SYMBOLIC-COLORED engine's metrics (the engine under test). `Ok(None)`
/// means a fail-closed DECLINE (out-of-sub-class / overflow / budget); `Ok(Some)`
/// a metric bundle.
fn symbolic_metrics(colored: &ColoredNet) -> Result<Option<Metrics>, String> {
    metrics_of(colored_state_space_metrics(colored, None))
}

/// The BINDING-QUANTIFIED engine's metrics (the driver under test). Same
/// Ok(None)=DECLINE convention as [`symbolic_metrics`].
fn quantified_metrics(colored: &ColoredNet) -> Result<Option<Metrics>, String> {
    metrics_of(colored_state_space_metrics_quantified(colored, None))
}

/// The quantified metrics with a wall-clock cap, for the always-green battery so
/// a high-diameter committed fixture (whose breadth-first quantified relprod is
/// SLOW but correct) declines (vacuous) instead of blowing the CI budget. A
/// deadline DECLINE is `Ok(None)` (fail-closed), NOT a disagreement.
fn quantified_metrics_capped(colored: &ColoredNet, secs: u64) -> Result<Option<Metrics>, String> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    match colored_state_space_metrics_quantified(colored, Some(deadline)) {
        // A resource/deadline DECLINE is fail-closed (vacuous), NOT a
        // disagreement — the capped helper exists precisely so a slow
        // high-diameter fixture declines instead of failing the battery.
        Err(SymbolicColoredError::Mdd(tla_mdd::CountError::ResourceCap(_))) => Ok(None),
        other => metrics_of(other),
    }
}

fn metrics_of(
    r: Result<tla_mdd::MddStateSpaceMetrics, SymbolicColoredError>,
) -> Result<Option<Metrics>, String> {
    match r {
        Ok(m) => Ok(Some(Metrics {
            states: m.state_count_u128,
            edges: m.edge_count,
            max_token_in_place: m.max_token_in_place,
            max_token_sum: m.max_token_sum,
        })),
        Err(SymbolicColoredError::OutOfSubclass(_)) => Ok(None),
        Err(SymbolicColoredError::Mdd(e)) => Err(format!("MDD fail-closed: {e:?}")),
    }
}

/// Run `colored_state_space_metrics` on a worker thread with the big DD stack,
/// exactly like the production MDD lanes do (the MDD recursions — count,
/// saturate, max-sum — descend the per-`(place,color)`-level node chain, so a
/// net with very many levels needs more than the default 8 MiB test-thread
/// stack). The gate-only/test win demonstrations cross 100k levels, hence this.
fn metrics_on_big_stack(
    colored: &ColoredNet,
    deadline: std::time::Instant,
) -> Result<tla_mdd::MddStateSpaceMetrics, SymbolicColoredError> {
    let colored = colored.clone();
    std::thread::Builder::new()
        .name("tla-symbolic-colored-test".into())
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || colored_state_space_metrics(&colored, Some(deadline)))
        .expect("spawn big-stack worker")
        .join()
        .expect("worker did not panic")
}

/// The core differential assertion: if the symbolic engine produced a bundle, it
/// MUST equal the oracle exactly. ALSO exercises the BINDING-QUANTIFIED driver:
/// whenever it decides, its metrics must equal BOTH the v1 enumerate path AND the
/// oracle (the quantified == enumerated == oracle gate). Returns `true` when the
/// v1 symbolic engine actually decided (so callers can assert non-vacuity).
fn assert_agrees(label: &str, colored: &ColoredNet) -> bool {
    let sym = symbolic_metrics(colored).unwrap_or_else(|e| panic!("{label}: {e}"));

    // The binding-quantified driver, run on the SAME net (capped so a slow
    // high-diameter fixture declines vacuously rather than blowing the CI
    // budget). If it decides, it must agree with the oracle (and, when v1 also
    // decided, with v1 exactly — proving the quantified image == the enumerated
    // image as a SET, transitively exact).
    let quant = quantified_metrics_capped(colored, 20)
        .unwrap_or_else(|e| panic!("{label} (quantified): {e}"));
    if let Some(quant) = quant {
        let oracle = oracle_metrics(colored).unwrap_or_else(|| {
            panic!("{label}: quantified decided but oracle could not unfold/spec")
        });
        assert_eq!(
            quant, oracle,
            "{label}: binding-QUANTIFIED StateSpace != unfolded oracle"
        );
        if let Some(sym) = sym {
            assert_eq!(
                quant, sym,
                "{label}: binding-QUANTIFIED StateSpace != v1 ENUMERATED StateSpace"
            );
        }
    }

    let Some(sym) = sym else {
        return false; // v1 symbolic declined — vacuously fine
    };
    let oracle = oracle_metrics(colored)
        .unwrap_or_else(|| panic!("{label}: symbolic decided but oracle could not unfold/spec"));
    assert_eq!(
        sym, oracle,
        "{label}: symbolic-colored StateSpace != unfolded oracle"
    );
    true
}

/// Like [`assert_agrees`] but asserts the QUANTIFIED driver actually DECIDED (it
/// did not silently fall back / decline) AND agrees — the non-vacuity guard for
/// the quantified gate. Returns the agreed metrics.
fn assert_quantified_decides_and_agrees(label: &str, colored: &ColoredNet) -> Metrics {
    let quant = quantified_metrics(colored)
        .unwrap_or_else(|e| panic!("{label} (quantified): {e}"))
        .unwrap_or_else(|| panic!("{label}: quantified driver DECLINED (vacuous)"));
    let oracle = oracle_metrics(colored)
        .unwrap_or_else(|| panic!("{label}: quantified decided but oracle could not unfold"));
    assert_eq!(quant, oracle, "{label}: quantified StateSpace != oracle");
    quant
}

// ---------------------------------------------------------------------------
// Hand-written colored fixtures in the v1 sub-class.
// ---------------------------------------------------------------------------

/// A cyclic token ring over a 3-color enum: one token rotates `c0 -> c1 -> c2 ->
/// c0` via successor. Conserving (1 token), so the engine admits it. Reachable:
/// 3 markings.
const TOKEN_RING_3_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="ring3" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><useroperator declaration="c0"/></subterm>
          </numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="rot"/>
      <arc id="p2t" source="p" target="rot">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="t2p" source="rot" target="p">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><successor><subterm><variable refvariable="x"/></subterm></successor></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration>
          <feconstant id="c0" name="c0"/>
          <feconstant id="c1" name="c1"/>
          <feconstant id="c2" name="c2"/>
        </cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// A 2-place shuttle over a 3-color enum: `<all>` tokens start on `p`, each
/// color shuttles `p <-> q` independently. Conserving. Reachable: 2^3 = 8.
const SHUTTLE_COL_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="shuttle" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <all><usersort declaration="C"/></all>
        </structure></hlinitialMarking>
      </place>
      <place id="q">
        <type><structure><usersort declaration="C"/></structure></type>
      </place>
      <transition id="fwd"/>
      <transition id="bwd"/>
      <arc id="p2f" source="p" target="fwd">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
      <arc id="f2q" source="fwd" target="q">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
      <arc id="q2b" source="q" target="bwd">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
      <arc id="b2p" source="bwd" target="p">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration>
          <feconstant id="c0" name="c0"/>
          <feconstant id="c1" name="c1"/>
          <feconstant id="c2" name="c2"/>
        </cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// A guarded shuttle: only colors with `x != c1` may shuttle (guard prunes one
/// binding). Validates that the engine's bindings respect the guard exactly.
const GUARDED_SHUTTLE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="gshuttle" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <all><usersort declaration="C"/></all>
        </structure></hlinitialMarking>
      </place>
      <place id="q">
        <type><structure><usersort declaration="C"/></structure></type>
      </place>
      <transition id="fwd">
        <condition><structure>
          <inequality>
            <subterm><variable refvariable="x"/></subterm>
            <subterm><useroperator declaration="c1"/></subterm>
          </inequality>
        </structure></condition>
      </transition>
      <arc id="p2f" source="p" target="fwd">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
      <arc id="f2q" source="fwd" target="q">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration>
          <feconstant id="c0" name="c0"/>
          <feconstant id="c1" name="c1"/>
          <feconstant id="c2" name="c2"/>
        </cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// A product-sort shuttle: a place over `C × C` (9 colors), `<all>` start,
/// shuttling `p <-> q` per (a,b) binding. Conserving. Reachable: 2^9 = 512.
const PRODUCT_SHUTTLE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="prod" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="CC"/></structure></type>
        <hlinitialMarking><structure>
          <all><usersort declaration="CC"/></all>
        </structure></hlinitialMarking>
      </place>
      <place id="q">
        <type><structure><usersort declaration="CC"/></structure></type>
      </place>
      <transition id="fwd"/>
      <transition id="bwd"/>
      <arc id="p2f" source="p" target="fwd">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm>
            <subterm><tuple><subterm><variable refvariable="a"/></subterm><subterm><variable refvariable="b"/></subterm></tuple></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="f2q" source="fwd" target="q">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm>
            <subterm><tuple><subterm><variable refvariable="a"/></subterm><subterm><variable refvariable="b"/></subterm></tuple></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="q2b" source="q" target="bwd">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm>
            <subterm><tuple><subterm><variable refvariable="a"/></subterm><subterm><variable refvariable="b"/></subterm></tuple></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="b2p" source="bwd" target="p">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm>
            <subterm><tuple><subterm><variable refvariable="a"/></subterm><subterm><variable refvariable="b"/></subterm></tuple></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration>
          <feconstant id="c0" name="c0"/>
          <feconstant id="c1" name="c1"/>
          <feconstant id="c2" name="c2"/>
        </cyclicenumeration>
      </namedsort>
      <namedsort id="CC" name="CC">
        <productsort><usersort declaration="C"/><usersort declaration="C"/></productsort>
      </namedsort>
      <variabledecl id="a" name="a"><usersort declaration="C"/></variabledecl>
      <variabledecl id="b" name="b"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn differential_token_ring_3() {
    let colored = parse_hlpnml(TOKEN_RING_3_PNML).expect("ring3 parses");
    assert!(
        assert_agrees("token_ring_3", &colored),
        "symbolic engine must DECIDE the conserving ring"
    );
    // Pin the expected count directly so the oracle cannot silently drift.
    let m = colored_state_space_metrics(&colored, None).expect("decides");
    assert_eq!(m.state_count_u128, 3, "ring of 3 colors ⇒ 3 markings");
}

#[test]
fn differential_shuttle_col() {
    let colored = parse_hlpnml(SHUTTLE_COL_PNML).expect("shuttle parses");
    assert!(assert_agrees("shuttle_col", &colored));
    let m = colored_state_space_metrics(&colored, None).expect("decides");
    assert_eq!(m.state_count_u128, 8, "3 independent shuttles ⇒ 2^3 = 8");
}

#[test]
fn differential_guarded_shuttle() {
    let colored = parse_hlpnml(GUARDED_SHUTTLE_PNML).expect("guarded shuttle parses");
    assert!(assert_agrees("guarded_shuttle", &colored));
    // Guard x != c1 ⇒ only c0, c2 ever move ⇒ 2 mutually-independent shuttles
    // each {still, moved}, c1 never moves ⇒ 2^2 = 4 reachable markings.
    let m = colored_state_space_metrics(&colored, None).expect("decides");
    assert_eq!(m.state_count_u128, 4, "guard prunes c1 ⇒ 2^2 = 4");
}

#[test]
fn differential_product_shuttle() {
    let colored = parse_hlpnml(PRODUCT_SHUTTLE_PNML).expect("product shuttle parses");
    assert!(assert_agrees("product_shuttle", &colored));
    let m = colored_state_space_metrics(&colored, None).expect("decides");
    assert_eq!(
        m.state_count_u128, 512,
        "9 independent shuttles ⇒ 2^9 = 512"
    );
}

/// Build a single-place token-ring colored net over an enum of `m` colors: one
/// token starts on color `c0` of place `p`, the transition `rot` rotates it via
/// `successor(x)`. Unfolds to `m` places + `m` transitions; reachable set is `m`
/// markings (the token position). Scalable to cross the unfolder's place /
/// transition caps.
fn build_color_ring(m: usize) -> ColoredNet {
    let consts: String = (0..m)
        .map(|i| format!("<feconstant id=\"c{i}\" name=\"c{i}\"/>"))
        .collect();
    let pnml = format!(
        r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="cring" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><useroperator declaration="c0"/></subterm></numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="rot"/>
      <arc id="p2t" source="p" target="rot">
        <hlinscription><structure><numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof></structure></hlinscription>
      </arc>
      <arc id="t2p" source="rot" target="p">
        <hlinscription><structure><numberof><subterm><numberconstant value="1"/></subterm><subterm><successor><subterm><variable refvariable="x"/></subterm></successor></subterm></numberof></structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C"><cyclicenumeration>{consts}</cyclicenumeration></namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#
    );
    parse_hlpnml(&pnml).expect("generated color-ring PNML parses")
}

/// Build a colored net with a SINGLE place over an enum of `m` colors, ONE
/// token (of color `c0`), and NO transitions: it unfolds to `m` places (one per
/// color), so it crosses the unfolder's `MAX_UNFOLDED_PLACES` cap for large `m`.
/// The sort holds only ONE token, so the symbolic engine's sound per-sort bound
/// is 1 — every slot's MDD domain is width 2 — and it handles the net in O(m):
/// the reachable set is the single initial marking (|R| = 1). The fast win
/// demonstration of the place-cap cliff.
fn build_wide_color_place(m: usize) -> ColoredNet {
    let consts: String = (0..m)
        .map(|i| format!("<feconstant id=\"c{i}\" name=\"c{i}\"/>"))
        .collect();
    let pnml = format!(
        r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="wide" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><useroperator declaration="c0"/></subterm></numberof>
        </structure></hlinitialMarking>
      </place>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C"><cyclicenumeration>{consts}</cyclicenumeration></namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#
    );
    parse_hlpnml(&pnml).expect("generated wide-place PNML parses")
}

/// DELIVERABLE (3): a concrete colored net the v1 engine DECIDES symbolically
/// where `unfold_to_pt` EXCEEDS its place-materialization budget.
///
/// `build_wide_color_place(M)` unfolds to `M` places. For `M` past the
/// unfolder's `MAX_UNFOLDED_PLACES` (100_000) cap, `unfold_to_pt` DECLINES with
/// `ColoredUnfoldUnavailable`, while the symbolic engine — which never
/// materializes the P/T place / transition / alias tables — answers the
/// StateSpace directly. Soundness is pinned on a SMALLER member of the SAME
/// family that the oracle CAN enumerate, where the symbolic count == the oracle.
///
/// Crossing the 100k place cap is intrinsically O(100k) setup (parse + level
/// build), so this runs only when the caller explicitly authorizes the heavy
/// campaign. The always-green soundness gate is the differential battery + the
/// small-member anchor here.
#[test]
fn win_decides_where_unfold_exceeds_place_budget() {
    let Some(campaign) = heavy_campaign_guard("win_decides_where_unfold_exceeds_place_budget")
    else {
        return;
    };
    // (a) Soundness anchor: a small member, symbolic == explicit-BFS oracle.
    let small = build_wide_color_place(100);
    assert!(
        assert_agrees("wide_color_place(100)", &small),
        "the symbolic engine must DECIDE the small family member"
    );
    let m_small = colored_state_space_metrics(&small, None).expect("small decides");
    assert_eq!(
        m_small.state_count_u128, 1,
        "no transitions ⇒ |R| = 1 (the initial marking)"
    );

    // (b) The WIN: a member past MAX_UNFOLDED_PLACES (100_000) — `unfold_to_pt`
    // must DECLINE (over budget) ...
    let big = build_wide_color_place(120_000);
    match crate::unfold::unfold_to_pt(&big) {
        Err(crate::error::PnmlError::ColoredUnfoldUnavailable { .. }) => {}
        other => panic!(
            "expected unfold_to_pt to exceed the place budget on the 120k-color net, got {:?}",
            other
                .map(|u| (u.net.num_places(), u.net.transitions.len()))
                .err()
        ),
    }
    // ... while the SYMBOLIC engine completes and returns the exact count.
    let t0 = std::time::Instant::now();
    let m_big = metrics_on_big_stack(&big, campaign.deadline_with_cap(120))
        .expect("symbolic engine must DECIDE the 120k-color net the unfolder cannot");
    eprintln!(
        "WIN(places): wide_color_place(120_000) symbolic StateSpace |R|={} \
         max_in_place={} computed in {:?} (unfold_to_pt DECLINED over the place budget)",
        m_big.state_count_u128,
        m_big.max_token_in_place,
        t0.elapsed(),
    );
    assert_eq!(
        m_big.state_count_u128, 1,
        "120k-color net, no transitions ⇒ |R| = 1"
    );
    assert_eq!(
        m_big.max_token_in_place, 1,
        "one token of c0 ⇒ max-in-place 1"
    );
}

/// DELIVERABLE (3), HEADLINE (explicitly qualified — DELIBERATELY EXPENSIVE): a
/// colored TOKEN RING over `M` colors whose unfolded P/T form has `M` places +
/// `M` transitions, past the unfolder's caps. `unfold_to_pt` DECLINES; the
/// symbolic engine returns the exact non-trivial four metrics (`|R| = M`,
/// `edges = M`).
///
/// NOTE: at `M = 120_000` the ring has diameter ~120k AND ~120k transitions, so
/// the saturation fixpoint's per-pass relprod verification sweep is heavy — this
/// can take MINUTES (and is given a 600s engine deadline). It is the richer
/// non-trivial-metric companion to the lighter place-cap win
/// (`win_decides_where_unfold_exceeds_place_budget`), which is the primary
/// deliverable-(3) proof. Run on demand with:
/// `TY_RUN_HEAVY_SYMBOLIC_COLORED_TESTS=1 cargo test -p tla-petri
/// --features dd-backend win_headline -- --nocapture`.
#[test]
fn win_headline_token_ring_past_budget() {
    let Some(campaign) = heavy_campaign_guard("win_headline_token_ring_past_budget") else {
        return;
    };
    // Soundness anchor on a small member.
    let small = build_color_ring(50);
    assert!(assert_agrees("color_ring(50)", &small));
    assert_eq!(
        colored_state_space_metrics(&small, None)
            .unwrap()
            .state_count_u128,
        50
    );

    let big = build_color_ring(120_000);
    assert!(
        matches!(
            crate::unfold::unfold_to_pt(&big),
            Err(crate::error::PnmlError::ColoredUnfoldUnavailable { .. })
        ),
        "unfold_to_pt must exceed budget on the 120k-color ring"
    );
    let t0 = std::time::Instant::now();
    let m = metrics_on_big_stack(&big, campaign.deadline_with_cap(600))
        .expect("symbolic engine must DECIDE the 120k ring");
    eprintln!(
        "WIN(headline): color_ring(120_000) |R|={} edges={} in {:?}",
        m.state_count_u128,
        m.edge_count,
        t0.elapsed()
    );
    assert_eq!(m.state_count_u128, 120_000);
    assert_eq!(m.edge_count, 120_000);
}

// ===========================================================================
// THE BINDING-QUANTIFIED DRIVER WIN: a net whose BINDING count exceeds the
// enumerate cap (so v1 / unfold_to_pt DECLINE) but whose reachable SET is tiny.
// ===========================================================================

/// Build a colored net whose single transition has TWO binding variables over a
/// `k`-color sort — so its binding product is `k²` — guarded `x == c0` (the
/// quantified prune cuts every `x != c0` sub-tree at depth 1), input `1'x`,
/// output `1'successor(x)`. One token starts on `c0`.
///
/// - Binding product = `k²`. For `k` with `k² > MAX_BINDING_ITERATIONS` (50M),
///   v1's `enumerate_bindings` DECLINES on the PRODUCT before any guard
///   filtering — exactly the binding cap the driver defeats.
/// - The guard keeps only `x = c0` (all `k` values of the spectator `y`), and
///   the effect (move the `c0` token to `c1`) ignores `y`, so EVERY surviving
///   binding shares ONE effect. The reachable set is `{1'c0, 1'c1}` ⇒ |R| = 2,
///   independent of `k` (only `c0` may move; `c1` cannot fire the `x==c0` guard).
/// - The quantified driver branches `x` (prune cuts `x != c0` immediately) then
///   `y` under `x = c0` (≈ `k` leaves, all one shared effect) ⇒ O(k) work, no
///   binding-product blow-up.
fn build_binding_cap_net(k: usize) -> ColoredNet {
    let consts: String = (0..k)
        .map(|i| format!("<feconstant id=\"c{i}\" name=\"c{i}\"/>"))
        .collect();
    let pnml = format!(
        r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="bcap" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><useroperator declaration="c0"/></subterm></numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="move">
        <condition><structure>
          <and>
            <subterm><equality>
              <subterm><variable refvariable="x"/></subterm>
              <subterm><useroperator declaration="c0"/></subterm>
            </equality></subterm>
            <subterm><equality>
              <subterm><variable refvariable="y"/></subterm>
              <subterm><variable refvariable="y"/></subterm>
            </equality></subterm>
          </and>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="move">
        <hlinscription><structure><numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof></structure></hlinscription>
      </arc>
      <arc id="t2p" source="move" target="p">
        <hlinscription><structure><numberof><subterm><numberconstant value="1"/></subterm><subterm><successor><subterm><variable refvariable="x"/></subterm></successor></subterm></numberof></structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C"><cyclicenumeration>{consts}</cyclicenumeration></namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
      <variabledecl id="y" name="y"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#
    );
    parse_hlpnml(&pnml).expect("generated binding-cap PNML parses")
}

/// DELIVERABLE (3), FAST always-run non-vacuity + soundness anchor for the
/// BINDING-cap family. A small member (`k = 6` ⇒ binding product 36) the v1
/// enumerate path CAN decide: the binding-QUANTIFIED driver must DECIDE it (NOT
/// silently fall back), and its metrics must equal the v1 enumerate path AND the
/// explicit-BFS oracle EXACTLY. This is the non-vacuous proof that the quantified
/// recursion actually runs on this family; the heavy member that crosses the cap
/// lives in the explicitly qualified
/// `win_binding_quantified_defeats_binding_cap` below.
#[test]
fn binding_cap_family_quantified_anchor() {
    for k in [3usize, 4, 6] {
        let net = build_binding_cap_net(k);
        let m = assert_quantified_decides_and_agrees(&format!("binding_cap({k})"), &net);
        // Family invariant: only c0 moves (to c1), so |R| = 2 for every k >= 2.
        assert_eq!(m.states, 2, "binding_cap(k): only c0 moves ⇒ |R| = 2");
        // The quantified driver fires the surviving x=c0 bindings (k of them, one
        // per spectator y), each enabled at exactly the c0-marking ⇒ edges = k
        // there + 0 at the c1-marking (guard x=c0 fails) ⇒ edges = k.
        assert_eq!(
            m.edges, k as u128,
            "binding_cap(k): edges = k (x=c0 bindings)"
        );
        // v1 enumerate path also decides this small member and agrees.
        assert!(assert_agrees(&format!("binding_cap({k})"), &net));
    }
}

/// DELIVERABLE (3), the HEADLINE BINDING-cap WIN (explicitly qualified because
/// O(k)-level setup is intrinsically multi-second). A member whose binding PRODUCT exceeds
/// `MAX_BINDING_ITERATIONS` (50M) makes v1 / `unfold_to_pt` DECLINE at the
/// binding cap, while the binding-QUANTIFIED driver DECIDES it with the SAME
/// family `|R| = 2` (validated vs the small member in
/// `binding_cap_family_quantified_anchor`).
///
/// `k = 8000` ⇒ binding product `64M > 50M`, 8000 `(place,color)` levels (under
/// the 200k symbolic cap). Run on demand:
/// `TY_RUN_HEAVY_SYMBOLIC_COLORED_TESTS=1 cargo test -p tla-petri
/// --features dd-backend win_binding -- --nocapture`.
#[test]
fn win_binding_quantified_defeats_binding_cap() {
    let Some(campaign) = heavy_campaign_guard("win_binding_quantified_defeats_binding_cap") else {
        return;
    };
    // (a) NON-VACUOUS soundness anchor on a small member v1 CAN enumerate
    //     (k=3 ⇒ 9 bindings): quantified == enumerated == oracle, |R| = 2.
    let small = build_binding_cap_net(3);
    let m_small = assert_quantified_decides_and_agrees("binding_cap(3)", &small);
    assert_eq!(m_small.states, 2, "only c0 moves (c0->c1) ⇒ |R| = 2");
    // The v1 enumerate path also decides the small member and agrees (anchor).
    assert!(assert_agrees("binding_cap(3)", &small));

    // (b) THE WIN: a member whose binding PRODUCT (k² = 64M) exceeds the 50M
    //     enumerate cap. v1 / unfold_to_pt DECLINE on the product...
    let k = 8000usize;
    assert!(k * k > 50_000_000, "k² must exceed MAX_BINDING_ITERATIONS");
    let big = build_binding_cap_net(k);
    match crate::unfold::unfold_to_pt(&big) {
        Err(crate::error::PnmlError::ColoredUnfoldUnavailable { .. }) => {}
        other => panic!(
            "expected unfold_to_pt to exceed the BINDING cap on k={k} (product {}), got {:?}",
            k * k,
            other
                .map(|u| (u.net.num_places(), u.net.transitions.len()))
                .err()
        ),
    }
    // The v1 symbolic enumerate path must ALSO decline (it calls the same
    // `enumerate_bindings`).
    assert!(
        symbolic_metrics(&big)
            .expect("no hard MDD failure")
            .is_none(),
        "v1 symbolic enumerate path must DECLINE at the binding cap"
    );

    // ... while the binding-QUANTIFIED driver DECIDES it (O(k) work, the prune
    // cuts every x != c0 sub-tree), returning the SAME family |R| = 2.
    let t0 = std::time::Instant::now();
    let m_big = colored_state_space_metrics_quantified(&big, Some(campaign.deadline_with_cap(120)))
        .expect("quantified driver must DECIDE the binding-cap net v1/unfold cannot");
    eprintln!(
        "WIN(binding-cap): k={k} (binding product {} > 50M cap) binding-QUANTIFIED \
         StateSpace |R|={} edges={} max_in_place={} in {:?} \
         (v1 enumerate + unfold_to_pt DECLINED at the binding cap)",
        k * k,
        m_big.state_count_u128,
        m_big.edge_count,
        m_big.max_token_in_place,
        t0.elapsed(),
    );
    assert_eq!(
        m_big.state_count_u128, 2,
        "family |R| = 2 (only c0 moves to c1), validated vs the small member"
    );
    assert_eq!(m_big.max_token_in_place, 1, "one token ⇒ max-in-place 1");
}

#[test]
fn quantified_declines_token_producing_fail_closed() {
    // A token-PRODUCING colored transition is outside the v1 sub-class. The
    // binding-quantified driver must DECLINE (fail-closed at the FIRED leaf's
    // non-increasing check), NEVER a wrong/partial count. Source-like `gen`
    // produces a token on `p` with no input.
    const PROD_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="prodq" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><useroperator declaration="c0"/></subterm></numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="gen"/>
      <arc id="t2p" source="gen" target="p">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration><feconstant id="c0" name="c0"/><feconstant id="c1" name="c1"/></cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;
    let colored = parse_hlpnml(PROD_PNML).expect("prod net parses");
    match colored_state_space_metrics_quantified(&colored, None) {
        Err(SymbolicColoredError::OutOfSubclass(reason)) => {
            assert!(
                reason.to_lowercase().contains("producing"),
                "quantified decline reason should name token-producing: {reason}"
            );
        }
        other => panic!("token-producing net must DECLINE (quantified), got {other:?}"),
    }
}

#[test]
fn differential_committed_fixtures() {
    // The committed MCC-derived fixtures (small COL members). They must either
    // be DECIDED and agree with the oracle, or DECLINE (fail-closed) — never a
    // disagreement. At least one must decide so the test is non-vacuous.
    let philosophers =
        parse_hlpnml(include_str!("../testdata/colored/philosophers_col_5.pnml")).ok();
    let token_ring = parse_hlpnml(include_str!("../testdata/colored/token_ring_10.pnml")).ok();
    let mut decided = 0u32;
    for (label, colored) in [
        ("philosophers_col_5", philosophers),
        ("token_ring_10", token_ring),
    ] {
        let Some(colored) = colored else { continue };
        if assert_agrees(label, &colored) {
            decided += 1;
        }
    }
    assert!(
        decided >= 1,
        "at least one committed colored fixture must be decided + agree with the oracle"
    );
}

#[test]
fn declines_subtract_arc_fail_closed() {
    // A Subtract arc inscription is excluded in v1 ⇒ must DECLINE (never a wrong
    // count). `1'all - 1'(x)` broadcast-to-all-but-self.
    const SUB_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="sub" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure><all><usersort declaration="C"/></all></structure></hlinitialMarking>
      </place>
      <transition id="t"/>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <subtract>
            <subterm><all><usersort declaration="C"/></all></subterm>
            <subterm><numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof></subterm>
          </subtract>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration><feconstant id="c0" name="c0"/><feconstant id="c1" name="c1"/></cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;
    let colored = parse_hlpnml(SUB_PNML).expect("sub net parses");
    match build_colored_mdd_net(&colored) {
        Err(SymbolicColoredError::OutOfSubclass(reason)) => {
            assert!(
                reason.to_lowercase().contains("subtract"),
                "decline reason should name Subtract: {reason}"
            );
        }
        other => panic!("Subtract arc must DECLINE out-of-subclass, got {other:?}"),
    }
}

#[test]
fn declines_token_producing_net_fail_closed() {
    // A source-like colored transition that produces a token with no input ⇒
    // token-PRODUCING ⇒ outside the v1 token-non-increasing sub-class ⇒ DECLINE.
    const PROD_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="prod" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
      </place>
      <transition id="gen"/>
      <arc id="t2p" source="gen" target="p">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration><feconstant id="c0" name="c0"/><feconstant id="c1" name="c1"/></cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;
    let colored = parse_hlpnml(PROD_PNML).expect("prod net parses");
    match build_colored_mdd_net(&colored) {
        Err(SymbolicColoredError::OutOfSubclass(reason)) => {
            assert!(
                reason.to_lowercase().contains("producing"),
                "decline reason should name token-producing: {reason}"
            );
        }
        other => panic!("token-producing net must DECLINE, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// PROPTEST over randomly-generated small colored nets in the v1 sub-class.
//
// Mirrors the crosscheck_bfs.rs proptest discipline: every generated net that
// the symbolic engine DECIDES must equal the explicitly-unfolded oracle exactly.
// A decline (out-of-sub-class) is not a disagreement.
// ---------------------------------------------------------------------------

mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a random conserving colored shuttle net: `k` colors, `np` places
    /// in a line, one token per color shuttling along adjacent places via
    /// move-forward / move-back transitions (token-conserving, so the engine
    /// admits it). Optionally guarded by `x != c0`.
    fn arb_colored_net() -> impl Strategy<Value = ColoredNet> {
        (2usize..=4, 2usize..=3, any::<bool>())
            .prop_map(|(k, np, guarded)| build_shuttle_line(k, np, guarded))
    }

    /// Build a colored shuttle-line net programmatically (parsing a generated
    /// PNML string keeps the path identical to production parsing).
    fn build_shuttle_line(k: usize, np: usize, guarded: bool) -> ColoredNet {
        let consts: String = (0..k)
            .map(|i| format!("<feconstant id=\"c{i}\" name=\"c{i}\"/>"))
            .collect();
        let mut places = String::new();
        for pi in 0..np {
            let init = if pi == 0 {
                "<hlinitialMarking><structure><all><usersort declaration=\"C\"/></all></structure></hlinitialMarking>"
                    .to_string()
            } else {
                String::new()
            };
            places.push_str(&format!(
                "<place id=\"p{pi}\"><type><structure><usersort declaration=\"C\"/></structure></type>{init}</place>"
            ));
        }
        let guard = if guarded {
            "<condition><structure><inequality><subterm><variable refvariable=\"x\"/></subterm><subterm><useroperator declaration=\"c0\"/></subterm></inequality></structure></condition>"
        } else {
            ""
        };
        let mut transitions = String::new();
        let mut arcs = String::new();
        let insc = "<hlinscription><structure><numberof><subterm><numberconstant value=\"1\"/></subterm><subterm><variable refvariable=\"x\"/></subterm></numberof></structure></hlinscription>";
        for pi in 0..np.saturating_sub(1) {
            // forward p{pi} -> p{pi+1}
            transitions.push_str(&format!("<transition id=\"f{pi}\">{guard}</transition>"));
            arcs.push_str(&format!(
                "<arc id=\"af{pi}i\" source=\"p{pi}\" target=\"f{pi}\">{insc}</arc>\
                 <arc id=\"af{pi}o\" source=\"f{pi}\" target=\"p{}\">{insc}</arc>",
                pi + 1
            ));
            // backward p{pi+1} -> p{pi}
            transitions.push_str(&format!("<transition id=\"b{pi}\"/>"));
            arcs.push_str(&format!(
                "<arc id=\"ab{pi}i\" source=\"p{}\" target=\"b{pi}\">{insc}</arc>\
                 <arc id=\"ab{pi}o\" source=\"b{pi}\" target=\"p{pi}\">{insc}</arc>",
                pi + 1
            ));
        }
        let pnml = format!(
            r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="gen" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">{places}{transitions}{arcs}</page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C"><cyclicenumeration>{consts}</cyclicenumeration></namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#
        );
        parse_hlpnml(&pnml).expect("generated colored PNML must parse")
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 256, ..ProptestConfig::default() })]

        #[test]
        fn symbolic_colored_equals_unfolded_oracle(colored in arb_colored_net()) {
            // If the symbolic engine decided, it must equal the oracle exactly.
            let _ = assert_agrees("proptest", &colored);
        }
    }

    /// Non-vacuity + never-disagree sweep with statistics (proptest hides
    /// per-case outcomes): confirm a healthy fraction of generated nets are
    /// actually DECIDED (not all declined) and that every decided net agrees.
    #[test]
    fn proptest_battery_is_non_vacuous() {
        use proptest::strategy::ValueTree;
        use proptest::test_runner::{Config, TestRunner};

        let mut runner = TestRunner::new(Config {
            cases: 200,
            ..Config::default()
        });
        let mut total = 0u32;
        let mut decided = 0u32;
        let mut multi_state = 0u32;
        for _ in 0..200 {
            let tree = arb_colored_net().new_tree(&mut runner).expect("gen net");
            let colored = tree.current();
            total += 1;
            if let Some(sym) = symbolic_metrics(&colored).expect("no MDD fail-closed on small net")
            {
                // Must equal the oracle.
                let oracle = oracle_metrics(&colored).expect("oracle unfolds decided net");
                assert_eq!(sym, oracle, "symbolic != oracle on generated net");
                decided += 1;
                if sym.states > 1 {
                    multi_state += 1;
                }
            }
        }
        assert!(
            decided * 2 >= total,
            "battery near-vacuous: only {decided}/{total} nets decided"
        );
        assert!(
            multi_state * 2 >= decided,
            "battery near-vacuous: only {multi_state}/{decided} decided nets were multi-state"
        );
    }

    /// NON-VACUITY for the BINDING-QUANTIFIED driver specifically (the prompt's
    /// "the quantified path must actually run, not silently fall back"): over a
    /// random battery, count how many generated nets the QUANTIFIED driver
    /// DECIDES, assert a healthy non-trivial fraction (so the recursion really
    /// fires), and verify every quantified decision equals BOTH the v1 enumerate
    /// path AND the explicit-BFS oracle EXACTLY (quantified == enumerated ==
    /// oracle). 0 disagreements.
    #[test]
    fn quantified_battery_is_non_vacuous() {
        use proptest::strategy::ValueTree;
        use proptest::test_runner::{Config, TestRunner};

        let mut runner = TestRunner::new(Config {
            cases: 200,
            ..Config::default()
        });
        let mut total = 0u32;
        let mut q_decided = 0u32;
        let mut q_multi_state = 0u32;
        for _ in 0..200 {
            let tree = arb_colored_net().new_tree(&mut runner).expect("gen net");
            let colored = tree.current();
            total += 1;
            let quant = quantified_metrics(&colored).expect("no MDD fail-closed on small net");
            let Some(quant) = quant else { continue };
            q_decided += 1;
            // quantified == oracle.
            let oracle = oracle_metrics(&colored).expect("oracle unfolds decided net");
            assert_eq!(quant, oracle, "quantified != oracle on generated net");
            // quantified == enumerated (when v1 also decided this small net).
            if let Some(sym) = symbolic_metrics(&colored).expect("v1 no MDD fail-closed") {
                assert_eq!(quant, sym, "quantified != v1 enumerated on generated net");
            }
            if quant.states > 1 {
                q_multi_state += 1;
            }
        }
        assert!(
            q_decided * 2 >= total,
            "quantified battery near-vacuous: only {q_decided}/{total} nets decided by the \
             quantified driver"
        );
        assert!(
            q_multi_state * 2 >= q_decided,
            "quantified battery near-vacuous: only {q_multi_state}/{q_decided} decided nets \
             were multi-state"
        );
    }
}
