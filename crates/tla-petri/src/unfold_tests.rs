// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::error::PnmlError;
use crate::hlpnml::parse_hlpnml;
use crate::petri_net::PlaceIdx;

use super::*;

/// Committed MCC-derived fixtures keep these unfolding tests buildable in a clean checkout.
const PHILOSOPHERS_5_PNML: &str = include_str!("../testdata/colored/philosophers_col_5.pnml");
const TOKEN_RING_10_PNML: &str = include_str!("../testdata/colored/token_ring_10.pnml");

const FINITE_INT_CONSTANT_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="ints"/></structure></type>
        <hlinitialMarking><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><finiteintrangeconstant value="1">
              <finiteintrange start="0" end="2"/>
            </finiteintrangeconstant></subterm>
          </numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="t">
        <condition><structure>
          <equality>
            <subterm><variable refvariable="x"/></subterm>
            <subterm><intconstant value="1"/></subterm>
          </equality>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="t2p" source="t" target="p">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><successor><subterm><intconstant value="0"/></subterm></successor></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="ints" name="Int">
        <finiteintrange start="0" end="2"/>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="ints"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// GreatSPN emits arc inscriptions wrapped in `<tuple>`, even when the place is
/// scalar-sorted: `<a>` on a `finiteintrange` place is a 1-tuple
/// `<tuple><subterm><variable refvariable="a"/></subterm></tuple>`. The unfolder
/// must treat a single-component tuple over a non-product sort as the bare inner
/// term so the arc is not silently dropped (the
/// UtilityControlRoom-COL-Z4T4N08 StableMarking wrong-TRUE root cause).
///
/// Net: scalar place `p` (sort Z, range 1..=2), transition `t` with a single
/// 1-tuple input arc `<a>` and a 1-tuple output arc `<a>` back to `p`. Two
/// bindings (a=1, a=2) ⇒ two transitions, each touching exactly one `p` color
/// instance on both input and output.
const SINGLETON_TUPLE_OVER_SCALAR_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="Z"/></structure></type>
        <hlinitialMarking><structure>
          <all><usersort declaration="Z"/></all>
        </structure></hlinitialMarking>
      </place>
      <transition id="t"/>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <tuple><subterm><variable refvariable="a"/></subterm></tuple>
        </structure></hlinscription>
      </arc>
      <arc id="t2p" source="t" target="p">
        <hlinscription><structure>
          <tuple><subterm><variable refvariable="a"/></subterm></tuple>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="Z" name="Z">
        <finiteintrange start="1" end="2"/>
      </namedsort>
      <variabledecl id="a" name="a"><usersort declaration="Z"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

const UNKNOWN_GUARD_CONSTANTS_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="ints"/></structure></type>
      </place>
      <transition id="t">
        <condition><structure>
          <equality>
            <subterm><useroperator declaration="missing_a"/></subterm>
            <subterm><useroperator declaration="missing_b"/></subterm>
          </equality>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="ints" name="Int">
        <finiteintrange start="0" end="2"/>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="ints"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// Tuple-vs-tuple guard `<x, y> = <x, y>` over a product sort `Pair = C × C`.
/// Both comparison operands are tuples (no sort context from either side
/// directly) — the comparison sort is derived from the place/variable product
/// context and each tuple is resolved to a flattened product index, so the
/// reflexive equality is TRUE for every binding. Pre-fix, both tuples resolved
/// to `None` and the guard silently evaluated to `false`, dropping the whole
/// net.
const TUPLE_GUARD_RESOLVES_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="Pair"/></structure></type>
      </place>
      <transition id="t">
        <condition><structure>
          <equality>
            <subterm><tuple>
              <subterm><variable refvariable="x"/></subterm>
              <subterm><variable refvariable="y"/></subterm>
            </tuple></subterm>
            <subterm><tuple>
              <subterm><variable refvariable="x"/></subterm>
              <subterm><variable refvariable="y"/></subterm>
            </tuple></subterm>
          </equality>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><tuple>
              <subterm><variable refvariable="x"/></subterm>
              <subterm><variable refvariable="y"/></subterm>
            </tuple></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C">
        <cyclicenumeration>
          <feconstant id="c0" name="c0"/>
          <feconstant id="c1" name="c1"/>
        </cyclicenumeration>
      </namedsort>
      <namedsort id="Pair" name="Pair">
        <productsort>
          <usersort declaration="C"/>
          <usersort declaration="C"/>
        </productsort>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="C"/></variabledecl>
      <variabledecl id="y" name="y"><usersort declaration="C"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// Guard `1 < 2` — two closed integer constants with no sort context. The
/// boolean is well-defined (TRUE), so every binding of `v` survives.
const INT_CONSTANT_GUARD_TRUE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="ints"/></structure></type>
      </place>
      <transition id="t">
        <condition><structure>
          <lessthan>
            <subterm><intconstant value="1"/></subterm>
            <subterm><intconstant value="2"/></subterm>
          </lessthan>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="v"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="ints" name="Int">
        <finiteintrange start="0" end="2"/>
      </namedsort>
      <variabledecl id="v" name="v"><usersort declaration="ints"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

/// Guard `2 < 1` — the FALSE mirror of the above. The comparison resolves
/// numerically to `false`, so all bindings are SOUNDLY dropped (genuine false,
/// not an unresolved operand).
const INT_CONSTANT_GUARD_FALSE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="ints"/></structure></type>
      </place>
      <transition id="t">
        <condition><structure>
          <lessthan>
            <subterm><intconstant value="2"/></subterm>
            <subterm><intconstant value="1"/></subterm>
          </lessthan>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="v"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="ints" name="Int">
        <finiteintrange start="0" end="2"/>
      </namedsort>
      <variabledecl id="v" name="v"><usersort declaration="ints"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

const UNDECLARED_ARC_VARIABLE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="colors"/></structure></type>
      </place>
      <transition id="t"/>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="missing"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="colors" name="Colors">
        <cyclicenumeration>
          <feconstant id="c0" name="a"/>
          <feconstant id="c1" name="b"/>
        </cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

const ALL_SORT_MISMATCH_ARC_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="small"/></structure></type>
      </place>
      <transition id="t"/>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure><all><usersort declaration="large"/></all></structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="small" name="Small">
        <cyclicenumeration><feconstant id="s0" name="s0"/></cyclicenumeration>
      </namedsort>
      <namedsort id="large" name="Large">
        <cyclicenumeration>
          <feconstant id="l0" name="l0"/>
          <feconstant id="l1" name="l1"/>
        </cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

const OUT_OF_SORT_USER_CONSTANT_ARC_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="small"/></structure></type>
      </place>
      <transition id="t"/>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><useroperator declaration="l1"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="small" name="Small">
        <cyclicenumeration><feconstant id="s0" name="s0"/></cyclicenumeration>
      </namedsort>
      <namedsort id="large" name="Large">
        <cyclicenumeration>
          <feconstant id="l0" name="l0"/>
          <feconstant id="l1" name="l1"/>
        </cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

const ALL_SORT_MISMATCH_INITIAL_MARKING_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="small"/></structure></type>
        <hlinitialMarking><structure><all><usersort declaration="large"/></all></structure></hlinitialMarking>
      </place>
    </page>
    <declaration><structure><declarations>
      <namedsort id="small" name="Small">
        <cyclicenumeration><feconstant id="s0" name="s0"/></cyclicenumeration>
      </namedsort>
      <namedsort id="large" name="Large">
        <cyclicenumeration><feconstant id="l0" name="l0"/></cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn test_unfold_fails_closed_for_undeclared_arc_variable() {
    let err = expect_unfold_error(
        UNDECLARED_ARC_VARIABLE_PNML,
        "undeclared arc variable must fail closed",
    );

    assert!(
        matches!(err, PnmlError::MissingElement(ref message) if message.contains("variable 'missing' not declared")),
        "expected undeclared variable error, got: {err:?}"
    );
}

#[test]
fn test_unfold_fails_closed_for_all_sort_mismatch_on_arc() {
    let err = expect_unfold_error(
        ALL_SORT_MISMATCH_ARC_PNML,
        "mismatched all sort must fail closed",
    );

    assert!(
        matches!(err, PnmlError::InvalidMarking(ref message) if message.contains("all sort 'large'")),
        "expected all-sort mismatch error, got: {err:?}"
    );
}

#[test]
fn test_unfold_fails_closed_for_unmapped_arc_color_value() {
    let err = expect_unfold_error(
        OUT_OF_SORT_USER_CONSTANT_ARC_PNML,
        "unmapped arc color must fail closed",
    );

    assert!(
        matches!(err, PnmlError::InvalidMarking(ref message) if message.contains("outside place 'p' sort 'small'")),
        "expected unmapped arc color error, got: {err:?}"
    );
}

#[test]
fn test_unfold_fails_closed_for_all_sort_mismatch_on_initial_marking() {
    let err = expect_unfold_error(
        ALL_SORT_MISMATCH_INITIAL_MARKING_PNML,
        "mismatched all marking sort must fail closed",
    );

    assert!(
        matches!(err, PnmlError::InvalidMarking(ref message) if message.contains("all sort 'large'")),
        "expected all-sort mismatch error, got: {err:?}"
    );
}

fn expect_unfold_error(pnml: &str, message: &str) -> PnmlError {
    let colored = parse_hlpnml(pnml).expect("should parse");
    match unfold_to_pt(&colored) {
        Ok(_) => panic!("{message}"),
        Err(err) => err,
    }
}

#[test]
fn test_unfold_philosophers_5_place_count() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // 5 colored places × 5 colors = 25 P/T places.
    assert_eq!(unfolded.net.num_places(), 25, "5 places × 5 colors = 25");
}

#[test]
fn test_unfold_finite_int_range_constants_in_markings_guards_and_arcs() {
    let colored =
        parse_hlpnml(FINITE_INT_CONSTANT_PNML).expect("should parse finite int constants");
    let unfolded = unfold_to_pt(&colored).expect("should unfold finite int constants");

    assert_eq!(unfolded.net.num_places(), 3);
    assert_eq!(unfolded.net.initial_marking, vec![0, 1, 0]);
    assert_eq!(
        unfolded.net.num_transitions(),
        1,
        "guard x = 1 should retain only the matching binding"
    );

    let transition = &unfolded.net.transitions[0];
    assert_eq!(transition.inputs.len(), 1);
    assert_eq!(transition.outputs.len(), 1);
    assert_eq!(transition.inputs[0].place, PlaceIdx(1));
    assert_eq!(
        transition.outputs[0].place,
        PlaceIdx(1),
        "successor(0) in finite range 0..=2 should resolve to color value 1"
    );
}

#[test]
fn test_unfold_singleton_tuple_over_scalar_sort_keeps_arcs() {
    let colored = parse_hlpnml(SINGLETON_TUPLE_OVER_SCALAR_PNML)
        .expect("should parse singleton-tuple-over-scalar net");
    let unfolded = unfold_to_pt(&colored).expect("should unfold singleton-tuple net");

    // Scalar place Z (1..=2) ⇒ 2 unfolded places, each initially holding 1 token.
    assert_eq!(unfolded.net.num_places(), 2, "Z has cardinality 2");
    assert_eq!(unfolded.net.initial_marking, vec![1, 1]);

    // Variable `a` over Z ⇒ 2 bindings ⇒ 2 transitions.
    assert_eq!(
        unfolded.net.num_transitions(),
        2,
        "one transition per binding of `a`"
    );

    // The crux: NEITHER arc may be dropped. Before the fix, a 1-tuple `<a>` over
    // the scalar sort Z fell through to `eval_color_value`, which returns None
    // for any Tuple, producing transitions with zero arcs and leaving every Z
    // instance spuriously isolated/constant.
    for (ti, transition) in unfolded.net.transitions.iter().enumerate() {
        assert_eq!(
            transition.inputs.len(),
            1,
            "transition {ti} must keep its 1-tuple input arc `<a>`"
        );
        assert_eq!(
            transition.outputs.len(),
            1,
            "transition {ti} must keep its 1-tuple output arc `<a>`"
        );
        // Each binding a=k touches exactly the same Z instance on input/output.
        assert_eq!(
            transition.inputs[0].place, transition.outputs[0].place,
            "binding `a` resolves the input and output 1-tuple to the same Z instance"
        );
    }

    // Both Z instances are reachable as arc targets (a=1 → place 0, a=2 → place 1).
    let touched: std::collections::HashSet<_> = unfolded
        .net
        .transitions
        .iter()
        .map(|t| t.inputs[0].place)
        .collect();
    assert_eq!(
        touched.len(),
        2,
        "both Z color instances must be connected by transitions"
    );
}

#[test]
fn test_unfold_guard_equality_fails_closed_for_unresolved_terms() {
    // A guard `missing_a = missing_b` references two user constants that exist
    // in no sort. Previously this silently evaluated to `false` (None == None),
    // dropping every binding and producing a DEFINITE but corrupted empty net —
    // the exact wrong-answer class this fix targets. The sound outcome is to
    // FAIL CLOSED: propagate `ColoredUnfoldUnavailable` so the loader maps the
    // model to per-examination CANNOT_COMPUTE instead of emitting a verdict
    // from a mis-evaluated net.
    let colored =
        parse_hlpnml(UNKNOWN_GUARD_CONSTANTS_PNML).expect("should parse unknown guard constants");
    // `UnfoldedNet` is not `Debug`, so match the result rather than `expect_err`.
    match unfold_to_pt(&colored) {
        Err(PnmlError::ColoredUnfoldUnavailable { .. }) => {}
        Err(other) => panic!(
            "unresolvable guard must yield ColoredUnfoldUnavailable (→ CANNOT_COMPUTE), got {other:?}"
        ),
        Ok(unfolded) => panic!(
            "unresolvable guard operands must fail closed, but unfolding succeeded with {} transitions",
            unfolded.net.num_transitions()
        ),
    }
}

#[test]
fn test_unfold_guard_tuple_equality_resolves_and_keeps_transitions() {
    // Guard `<x, y> = <a, b>` over a product sort `Pair = C × C`. Both operands
    // are tuples; the comparison must resolve component-wise via the flattened
    // product index (NOT silently fail). The variable `p` (over Pair) carries
    // the product sort so the tuples resolve in it. The guard `<x,y> = <a,b>`
    // here compares a tuple of the SAME variables against itself, so it is
    // TRUE for every binding and NONE of the transitions may be dropped.
    let colored = parse_hlpnml(TUPLE_GUARD_RESOLVES_PNML).expect("should parse tuple-guard net");
    let unfolded = unfold_to_pt(&colored).expect("tuple-comparison guard must resolve, not fail");

    // C has cardinality 2 ⇒ two variables x,y each over C ⇒ 4 bindings. The
    // reflexive guard `<x,y> = <x,y>` holds for all 4, so 4 transitions survive.
    assert_eq!(
        unfolded.net.num_transitions(),
        4,
        "reflexive tuple equality guard is TRUE for every binding (4 = |C|²)"
    );
}

#[test]
fn test_unfold_guard_int_constant_comparison_resolves() {
    // Guard `1 < 2` is two closed integer constants with no sort context. The
    // pre-fix path routed both through color-index resolution (which returns
    // None for a bare IntegerConstant), silently dropping every binding. The
    // boolean is well-defined (TRUE), so all bindings must be kept.
    let colored =
        parse_hlpnml(INT_CONSTANT_GUARD_TRUE_PNML).expect("should parse int-constant-guard net");
    let unfolded =
        unfold_to_pt(&colored).expect("closed int-constant comparison must resolve numerically");
    // One variable `v` over a 3-valued sort ⇒ 3 bindings; `1 < 2` is TRUE for
    // all of them.
    assert_eq!(
        unfolded.net.num_transitions(),
        3,
        "`1 < 2` is true ⇒ every binding kept"
    );

    // The mirror guard `2 < 1` is FALSE ⇒ all bindings dropped (a SOUND drop,
    // because the guard genuinely evaluates to false — not an unresolved one).
    let colored_false = parse_hlpnml(INT_CONSTANT_GUARD_FALSE_PNML)
        .expect("should parse false int-constant-guard net");
    let unfolded_false =
        unfold_to_pt(&colored_false).expect("closed int-constant comparison must resolve");
    assert_eq!(
        unfolded_false.net.num_transitions(),
        0,
        "`2 < 1` is genuinely false ⇒ no binding fires (sound drop, not corruption)"
    );
}

#[test]
fn test_unfold_philosophers_5_transition_count() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // 5 transitions × 5 bindings (1 variable, 5 colors) = 25 transitions.
    assert_eq!(
        unfolded.net.num_transitions(),
        25,
        "5 transitions × 5 bindings = 25"
    );
}

#[test]
fn test_unfold_philosophers_5_initial_marking() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // Total initial tokens: Think has all 5 colors, Fork has all 5 colors.
    // Other places (Catch1, Catch2, Eat) have 0.
    let total: u64 = unfolded.net.initial_marking.iter().sum();
    assert_eq!(total, 10, "5 tokens in Think + 5 in Fork = 10");
}

#[test]
fn test_unfold_philosophers_5_place_aliases() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // "Think" should map to 5 unfolded places.
    let think_places = unfolded.aliases.resolve_places("Think");
    assert!(think_places.is_some());
    assert_eq!(
        think_places.unwrap().len(),
        5,
        "Think maps to 5 unfolded places"
    );

    // "Fork" should also map to 5 unfolded places.
    let fork_places = unfolded.aliases.resolve_places("Fork");
    assert!(fork_places.is_some());
    assert_eq!(fork_places.unwrap().len(), 5);
}

#[test]
fn test_unfold_philosophers_5_transition_aliases() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // Each colored transition should map to 5 unfolded instances.
    for name in &["FF1a", "FF1b", "FF2a", "FF2b", "End"] {
        let trans = unfolded.aliases.resolve_transitions(name);
        assert!(trans.is_some(), "transition '{name}' should have aliases");
        assert_eq!(
            trans.unwrap().len(),
            5,
            "transition '{name}' should map to 5 unfolded instances"
        );
    }
}

#[test]
fn test_unfold_philosophers_5_all_transitions_have_arcs() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // Every unfolded transition should have at least one input and one output.
    for trans in &unfolded.net.transitions {
        assert!(
            !trans.inputs.is_empty(),
            "transition '{}' should have inputs",
            trans.id
        );
        assert!(
            !trans.outputs.is_empty(),
            "transition '{}' should have outputs",
            trans.id
        );
    }
}

#[test]
fn test_unfold_philosophers_5_end_transition_arc_weights() {
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // End transition: consumes 1'(x) from Eat, produces 1'(x) to Think
    // + 1'(x) + 1'(x--1) to Fork. Total: 1 in, 3 out.
    for trans in &unfolded.net.transitions {
        if trans.id.starts_with("End_") {
            let in_w: u64 = trans.inputs.iter().map(|a| a.weight).sum();
            let out_w: u64 = trans.outputs.iter().map(|a| a.weight).sum();
            assert_eq!(in_w, 1, "{}: End consumes 1 from Eat", trans.id);
            assert_eq!(
                out_w, 3,
                "{}: End produces 1 to Think + 2 to Fork",
                trans.id
            );
        }
    }
}

#[test]
fn test_unfold_size_guard_places() {
    // Create a colored net that would exceed the place limit.
    // We can't easily do this with real models, so test the concept
    // with a direct assertion on the limit constants.
    // The caps are compile-time constants; the assertions document the
    // positivity invariant the size guard relies on.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(MAX_UNFOLDED_PLACES > 0);
        assert!(MAX_UNFOLDED_TRANSITIONS > 0);
    }
    // The memory-aware cap (v8 diagnosis 2026-07-10) is clamped to
    // [historical floor, ceiling] — no environment admits less than the old
    // fixed cap, none more than the ceiling.
    let cap = crate::unfold::max_unfolded_transitions();
    assert!(cap >= MAX_UNFOLDED_TRANSITIONS);
    assert!(cap <= crate::unfold::MAX_UNFOLDED_TRANSITIONS_CEIL);
}

/// A single colored place over a `finiteintrange` sort whose cardinality
/// (`end - start + 1`) exceeds `MAX_UNFOLDED_PLACES`, so unfolding the places
/// alone trips the size cap. Used to exercise the recoverable abort path.
const OVER_PLACE_CAP_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="overcap" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="big"/></structure></type>
      </place>
      <transition id="t"/>
      <arc id="p2t" source="p" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="t2p" source="t" target="p">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="big" name="Big">
        <finiteintrange start="0" end="200000"/>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="big"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn unfold_emits_recoverable_error_over_place_cap() {
    // The unfolded place count (200_001) exceeds MAX_UNFOLDED_PLACES, so the
    // place loop must abort with the RECOVERABLE `ColoredUnfoldUnavailable`
    // error -- NOT `UnsupportedNetType` (which means a genuinely unsupported
    // construct and collapses the whole model).
    let colored = parse_hlpnml(OVER_PLACE_CAP_PNML).expect("fixture should parse");
    let err = match unfold_to_pt(&colored) {
        Ok(_) => panic!("over-cap unfold must error, not succeed"),
        Err(err) => err,
    };
    assert!(
        matches!(err, PnmlError::ColoredUnfoldUnavailable { .. }),
        "expected recoverable ColoredUnfoldUnavailable, got: {err}"
    );
}

#[test]
fn unfold_aborts_on_past_deadline() {
    // A non-trivial colored net unfolded under a budget whose deadline is
    // already in the past must abort cleanly with `ColoredUnfoldUnavailable`.
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("fixture should parse");
    let budget = UnfoldBudget::new(Some(std::time::Instant::now()));
    let err = match unfold_to_pt_with_budget(&colored, &budget) {
        Ok(_) => panic!("past-deadline unfold must error, not succeed"),
        Err(err) => err,
    };
    assert!(
        matches!(err, PnmlError::ColoredUnfoldUnavailable { .. }),
        "expected ColoredUnfoldUnavailable on past deadline, got: {err}"
    );
}

#[test]
fn unfold_unbounded_budget_matches_plain_unfold() {
    // The plain `unfold_to_pt` and a `None`-deadline budget must agree: the
    // budget machinery cannot change success behavior for local/test runs.
    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("fixture should parse");
    let plain = unfold_to_pt(&colored).expect("should unfold");
    let budgeted = unfold_to_pt_with_budget(&colored, &UnfoldBudget::default())
        .expect("should unfold under unbounded budget");
    assert_eq!(plain.net.places.len(), budgeted.net.places.len());
    assert_eq!(plain.net.transitions.len(), budgeted.net.transitions.len());
}

#[test]
fn test_unfold_token_ring_product_place_alias_cardinality() {
    let colored = parse_hlpnml(TOKEN_RING_10_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    let state_places = unfolded
        .aliases
        .resolve_places("State")
        .expect("State aliases should exist");
    assert_eq!(
        state_places.len(),
        121,
        "11 x 11 product sort should unfold to 121 places"
    );

    let total_tokens: u64 = state_places
        .iter()
        .map(|place| unfolded.net.initial_marking[place.0 as usize])
        .sum();
    assert_eq!(
        total_tokens, 11,
        "State starts with one token per process pair (i, i)"
    );
}

#[test]
fn test_unfold_token_ring_product_transitions_keep_arcs() {
    let colored = parse_hlpnml(TOKEN_RING_10_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    let main_process = unfolded
        .aliases
        .resolve_transitions("MainProcess")
        .expect("MainProcess aliases should exist");
    assert_eq!(
        main_process.len(),
        11,
        "single variable x should leave one MainProcess binding per process"
    );

    for transition_idx in main_process {
        let transition = &unfolded.net.transitions[transition_idx.0 as usize];
        assert!(
            !transition.inputs.is_empty(),
            "{} should keep product-sort input arcs",
            transition.id
        );
        assert!(
            !transition.outputs.is_empty(),
            "{} should keep tuple/successor output arcs",
            transition.id
        );
    }
}

/// Synthetic PNML with lessthanorequal guard: t1 fires only when x <= y.
/// Sort has 3 constants {a, b, c}, so x <= y allows 6 of 9 bindings:
/// (a,a), (a,b), (a,c), (b,b), (b,c), (c,c).
const ORDERING_GUARD_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="test" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p1">
        <type><structure><usersort declaration="s1"/></structure></type>
        <hlinitialMarking><structure><all><usersort declaration="s1"/></all></structure></hlinitialMarking>
      </place>
      <place id="p2">
        <type><structure><usersort declaration="s1"/></structure></type>
      </place>
      <transition id="t1">
        <condition><structure>
          <lessthanorequal>
            <subterm><variable refvariable="x"/></subterm>
            <subterm><variable refvariable="y"/></subterm>
          </lessthanorequal>
        </structure></condition>
      </transition>
      <arc id="a1" source="p1" target="t1">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
      <arc id="a2" source="t1" target="p2">
        <hlinscription><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="y"/></subterm></numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="s1" name="S">
        <cyclicenumeration>
          <feconstant id="c1" name="a"/>
          <feconstant id="c2" name="b"/>
          <feconstant id="c3" name="c"/>
        </cyclicenumeration>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="s1"/></variabledecl>
      <variabledecl id="y" name="y"><usersort declaration="s1"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn test_unfold_ordering_guard_restricts_bindings() {
    // With `lessthanorequal(x, y)` on a 3-element sort, only 6 of 9
    // bindings survive (upper-triangular): (0,0),(0,1),(0,2),(1,1),(1,2),(2,2).
    let colored = parse_hlpnml(ORDERING_GUARD_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // Without guard: 3 × 3 = 9 bindings → 9 unfolded transitions.
    // With lessthanorequal: 6 bindings → 6 unfolded transitions.
    assert_eq!(
        unfolded.net.num_transitions(),
        6,
        "lessthanorequal guard should allow 6 of 9 bindings"
    );
}

/// Regression: PropertyAliases::colored_place_groups must NOT aggregate
/// distinct colored places by sort name. The Sudoku-COL-AN01 shape has
/// three places (Rows, Cells, Columns) all of sort N2 with N={1..1}
/// (cardinality 1 each), and one place (Board) of sort N3. Each
/// colored place becomes a SINGLE unfolded P/T place because
/// cardinality is 1, so `colored_place_groups()` must return an empty
/// vec (no multi-color groups). The earlier buggy implementation built
/// groups from `place_aliases.values().filter(len > 1)`, which picked
/// up the sort-name alias `N2 → [Rows_0, Cells_0, Columns_0]` and fed
/// it to OneSafe as if it were one colored place, summing tokens
/// across the three places (3 > 1 → wrong FALSE).
#[test]
fn test_colored_place_groups_excludes_sort_name_aggregates() {
    const SUDOKU_AN01_LIKE: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="n" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <declaration><structure><declarations>
      <namedsort id="N" name="N"><finiteintrange start="1" end="1"/></namedsort>
      <namedsort id="N2" name="N2"><productsort>
        <usersort declaration="N"/><usersort declaration="N"/>
      </productsort></namedsort>
      <variabledecl id="x" name="x"><usersort declaration="N"/></variabledecl>
      <variabledecl id="y" name="y"><usersort declaration="N"/></variabledecl>
    </declarations></structure></declaration>
    <page id="p0">
      <place id="Rows">
        <type><structure><usersort declaration="N2"/></structure></type>
        <hlinitialMarking><structure><tuple>
          <subterm><all><usersort declaration="N"/></all></subterm>
          <subterm><all><usersort declaration="N"/></all></subterm>
        </tuple></structure></hlinitialMarking>
      </place>
      <place id="Cells">
        <type><structure><usersort declaration="N2"/></structure></type>
        <hlinitialMarking><structure><tuple>
          <subterm><all><usersort declaration="N"/></all></subterm>
          <subterm><all><usersort declaration="N"/></all></subterm>
        </tuple></structure></hlinitialMarking>
      </place>
      <place id="Columns">
        <type><structure><usersort declaration="N2"/></structure></type>
        <hlinitialMarking><structure><tuple>
          <subterm><all><usersort declaration="N"/></all></subterm>
          <subterm><all><usersort declaration="N"/></all></subterm>
        </tuple></structure></hlinitialMarking>
      </place>
      <transition id="t"/>
      <arc id="a1" source="Rows" target="t">
        <hlinscription><structure><tuple>
          <subterm><variable refvariable="x"/></subterm>
          <subterm><variable refvariable="y"/></subterm>
        </tuple></structure></hlinscription>
      </arc>
    </page>
  </net>
</pnml>"#;
    let colored = parse_hlpnml(SUDOKU_AN01_LIKE).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // Each colored place has cardinality 1 (N={1..1}, N2=N×N=1), so the
    // sort-name alias `N2` aggregates three distinct P/T places.
    let n2_alias = unfolded
        .aliases
        .resolve_places("N2")
        .expect("N2 sort-name alias exists");
    assert_eq!(
        n2_alias.len(),
        3,
        "N2 sort-name alias aggregates Rows + Cells + Columns"
    );

    // But colored_place_groups must NOT treat that sort-name aggregate
    // as a color group — every colored place's group is a singleton
    // and must be filtered. Returning [Rows_0, Cells_0, Columns_0]
    // here is the exact wrong-answer path that flipped AN01 OneSafe.
    let groups = unfolded.aliases.colored_place_groups();
    assert!(
        groups.is_empty(),
        "colored_place_groups must not include sort-name aggregates; got {groups:?}"
    );

    // Initial marking must be exactly 1 token per place (the
    // `<All,All>` expansion under N={1..1} has one product element).
    assert_eq!(unfolded.net.initial_marking, vec![1, 1, 1]);
}

/// Diagnostic regression: Murphy-COL-D1N010-shaped initial marking
/// `{10'CD.all}` over a 2-element cyclic enum should expand to 10
/// tokens on EACH color of the unfolded place (20 total per place).
///
/// Before investigation: ty produced 10 states / 16 transitions where
/// the MCC consensus is 39780 / 267984. We must verify whether the
/// unfolder's initial marking is correct.
#[test]
fn test_unfold_numberof_all_over_cyclic_enum_initial_marking() {
    const MURPHY_LIKE: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="n" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <declaration><structure><declarations>
      <namedsort id="CD" name="CD"><cyclicenumeration>
        <feconstant id="c0" name="0"/>
        <feconstant id="c1" name="1"/>
      </cyclicenumeration></namedsort>
      <variabledecl id="x" name="x"><usersort declaration="CD"/></variabledecl>
    </declarations></structure></declaration>
    <page id="p0">
      <place id="p0">
        <type><structure><usersort declaration="CD"/></structure></type>
        <hlinitialMarking><structure>
          <numberof>
            <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
            <subterm><all><usersort declaration="CD"/></all></subterm>
          </numberof>
        </structure></hlinitialMarking>
      </place>
      <place id="p1">
        <type><structure><usersort declaration="CD"/></structure></type>
        <hlinitialMarking><structure>
          <numberof>
            <subterm><numberconstant value="10"><positive/></numberconstant></subterm>
            <subterm><all><usersort declaration="CD"/></all></subterm>
          </numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="t"/>
      <arc id="a" source="p0" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
  </net>
</pnml>"#;
    let colored = parse_hlpnml(MURPHY_LIKE).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");
    // 2 places × 2 colors = 4 P/T places.
    assert_eq!(unfolded.net.num_places(), 4);
    // {1'CD.all} → 1 on each of p0_0, p0_1.
    // {10'CD.all} → 10 on each of p1_0, p1_1.
    // Order: p0_0, p0_1, p1_0, p1_1.
    assert_eq!(
        unfolded.net.initial_marking,
        vec![1, 1, 10, 10],
        "{{N'CD.all}} should place N tokens on EACH color, not just one"
    );
}

/// Tier 2 #5 regression: the CLI used to fail-closed on every COL input
/// for `CTLCardinality`/`CTLFireability` (via the `should_fail_closed_ctl`
/// gate in `cli.rs`). The unfolder + downstream CTL pipeline have been
/// sound for the supported colored subset for a while; this test pins
/// the contract so the gate cannot regress.
///
/// Strategy: parse the philosophers_col_5 fixture, unfold to a P/T net,
/// build a small battery of CTL formulas that reference *colored* place
/// names (which resolve through `PropertyAliases` to the unfolded
/// indices), and verify the unified CTL pipeline produces the same
/// verdict as the full-graph oracle for each formula at the initial
/// state. Any mismatch indicates the unfolded-CTL path is not sound on
/// colored inputs and the CLI gate must stay in place.
#[test]
fn test_ctl_pipeline_on_unfolded_philosophers_5_matches_full_graph_oracle() {
    use crate::examinations::ctl::check_ctl_properties_with_aliases;
    use crate::examinations::ctl::resolve::resolve_ctl_with_aliases;
    use crate::explorer::{explore_full, ExplorationConfig};
    use crate::output::Verdict;
    use crate::property_xml::{CtlFormula, Formula, IntExpr, Property, StatePredicate};
    use crate::resolved_predicate::{eval_predicate, ResolvedPredicate};
    use tla_mc_core::{build_predecessor_adjacency, CtlAtomEvaluator, CtlEngine, IndexedCtlGraph};

    let colored = parse_hlpnml(PHILOSOPHERS_5_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");
    let net = &unfolded.net;
    let aliases = &unfolded.aliases;

    // Use colored place names (Think, Fork, Eat) so the test exercises
    // the alias resolution layer that the CLI path goes through. Each
    // such name resolves to a 5-place group in the unfolded net.
    let atom_ge = |place: &str, value: u64| -> CtlFormula {
        CtlFormula::Atom(StatePredicate::IntLe(
            IntExpr::Constant(value),
            IntExpr::TokensCount(vec![place.to_string()]),
        ))
    };
    let battery = [
        // Initial marking has Think and Fork each with 5 tokens (one per
        // philosopher), Eat with 0. Sum semantics: TokensCount over
        // multi-place aliases sums tokens across all members.
        atom_ge("Think", 1),
        atom_ge("Fork", 1),
        atom_ge("Eat", 1),
        // Reachability of an eating state.
        CtlFormula::EF(Box::new(atom_ge("Eat", 1))),
        // Inevitability (false in general — there's an all-think loop).
        CtlFormula::AG(Box::new(atom_ge("Think", 1))),
        // Global liveness sanity: from any state, Think still reachable.
        CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(atom_ge("Think", 1))))),
    ];

    // Build the full-graph oracle once.
    let config = ExplorationConfig::new(100_000);
    let full = explore_full(net, &config);
    assert!(
        full.graph.completed,
        "philosophers_col_5 unfolded state space must fit the 100k budget",
    );
    let predecessors = build_predecessor_adjacency(&full.graph.adj);

    struct PetriCtlAtomEval<'a> {
        net: &'a crate::petri_net::PetriNet,
    }
    impl<'a> CtlAtomEvaluator<Vec<u64>, ResolvedPredicate> for PetriCtlAtomEval<'a> {
        fn evaluate(&self, state: &Vec<u64>, atom: &ResolvedPredicate) -> bool {
            eval_predicate(atom, state, self.net)
        }
    }

    for (idx, ctl) in battery.iter().enumerate() {
        // Oracle verdict: full-graph CTL engine.
        let resolved = resolve_ctl_with_aliases(ctl, aliases);
        let unpacked = full.markings.unpack_all();
        let graph = IndexedCtlGraph::new(&unpacked, &full.graph.adj, &predecessors);
        let oracle_sat = CtlEngine::new(graph, PetriCtlAtomEval { net }).eval(&resolved);
        let oracle_verdict = if oracle_sat[0] {
            Verdict::True
        } else {
            Verdict::False
        };

        // Pipeline verdict: the path the CLI takes for colored inputs
        // once the gate is lifted.
        let property = Property {
            id: format!("PhilColored-CTL-{idx:02}"),
            formula: Formula::Ctl(ctl.clone()),
        };
        let results = check_ctl_properties_with_aliases(
            net,
            std::slice::from_ref(&property),
            aliases,
            &config,
        );
        assert_eq!(results.len(), 1, "single property in, single result out");
        let (id, pipeline_verdict) = &results[0];
        assert_eq!(id, &property.id);
        assert_ne!(
            *pipeline_verdict,
            Verdict::CannotCompute,
            "CTL pipeline must produce a concrete verdict on the unfolded \
             colored net (formula #{idx}: {ctl:?}); CannotCompute here \
             means the lifted CLI gate would leak indeterminate results",
        );
        assert_eq!(
            *pipeline_verdict, oracle_verdict,
            "formula #{idx} pipeline vs oracle mismatch on unfolded \
             philosophers_col_5: pipeline={pipeline_verdict:?} \
             oracle={oracle_verdict:?}; formula: {ctl:?}",
        );
    }
}

/// Constructed repro for the colored place-alias DOUBLE-COUNTING bug.
///
/// A colored place whose `id` (or `name`) equals its sort's NAME used to
/// have its unfolded-index alias bucket appended to itself by the
/// "register by sort name" aggregate in `unfold_places`. For an N-color
/// place that produced `[i0..iN-1, i0..iN-1]` — every color DUPLICATED.
///
/// That poisoned vector flowed un-deduped into:
///   1. `colored_place_groups()` → the OneSafe group listed every unfolded
///      place TWICE, so the group sum was DOUBLED → a genuinely 1-safe
///      colored place (true group sum 1) reported 2 > 1 → spurious
///      `OneSafe = FALSE`, fired even on the INITIAL marking.
///   2. `resolve_place_bound()` → the duplicate read as coefficient 2 →
///      DOUBLED `UpperBounds` for a tokens-count query naming that place.
///
/// This net is a single place `Process` typed by a 2-element sort *also*
/// named `Process`, holding exactly one token (on color `a`). A transition
/// rotates the token a→b→a, so the place is genuinely 1-safe with an exact
/// max token sum of 1. The fix must give `OneSafe = TRUE` and bound `1`.
const PLACE_NAMED_AFTER_SORT_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="n" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <declaration><structure><declarations>
      <namedsort id="Process" name="Process"><cyclicenumeration>
        <feconstant id="a" name="a"/>
        <feconstant id="b" name="b"/>
      </cyclicenumeration></namedsort>
      <variabledecl id="x" name="x"><usersort declaration="Process"/></variabledecl>
    </declarations></structure></declaration>
    <page id="p0">
      <place id="Process">
        <type><structure><usersort declaration="Process"/></structure></type>
        <hlinitialMarking><structure>
          <numberof>
            <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
            <subterm><useroperator declaration="a"/></subterm>
          </numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="t"/>
      <arc id="in" source="Process" target="t">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="out" source="t" target="Process">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
            <subterm><successor><subterm><variable refvariable="x"/></subterm></successor></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
  </net>
</pnml>"#;

/// The place `Process` typed by sort `Process` must NOT have its alias
/// bucket doubled. Pins the SOURCE fix in `unfold_places`.
#[test]
fn test_place_named_after_sort_alias_not_doubled() {
    let colored = parse_hlpnml(PLACE_NAMED_AFTER_SORT_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");

    // 1 place × 2 colors = 2 unfolded P/T places.
    assert_eq!(unfolded.net.num_places(), 2, "Process unfolds to 2 colors");

    // The alias bucket for `Process` must list each unfolded color EXACTLY
    // once. The bug produced length 4 ([i0,i1,i0,i1]).
    let alias = unfolded
        .aliases
        .resolve_places("Process")
        .expect("Process alias exists");
    assert_eq!(
        alias.len(),
        2,
        "Process alias must be the 2 colors exactly once, not doubled; got {alias:?}",
    );
    // And no duplicate indices.
    let mut sorted: Vec<_> = alias.to_vec();
    sorted.sort_unstable();
    let unique = {
        let mut s = sorted.clone();
        s.dedup();
        s
    };
    assert_eq!(sorted, unique, "Process alias must contain no duplicates");
}

/// OneSafe: the genuinely 1-safe `Process`(typed by sort `Process`) must
/// report TRUE. Under the bug, `colored_place_groups()` listed each color
/// twice → the initial-marking group sum was 2 > 1 → spurious FALSE.
///
/// Cross-checks the OneSafe verdict against the explicit group-sum truth
/// (max over ALL reachable markings of the deduplicated group sum).
#[test]
fn test_place_named_after_sort_one_safe_true() {
    use crate::examinations::one_safe::OneSafeObserver;
    use crate::explorer::{explore_full, ExplorationConfig};

    let colored = parse_hlpnml(PLACE_NAMED_AFTER_SORT_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");
    let net = &unfolded.net;

    // The colored group for OneSafe must be a single 2-element group with
    // no duplicates. The bug produced [i0,i1,i0,i1] (len 4).
    let groups = unfolded.aliases.colored_place_groups();
    assert_eq!(
        groups.len(),
        1,
        "exactly one colored group for Process; got {groups:?}"
    );
    assert_eq!(
        groups[0].len(),
        2,
        "the Process group must be its 2 colors exactly once; got {groups:?}"
    );

    // Initial-marking group sum is exactly 1 (token on color `a`). The bug
    // would have summed marking over [i0,i1,i0,i1] = 1+0+1+0 = 2 → FALSE on
    // the INITIAL marking. Assert the deduped group sums to 1 here.
    let init_sum: u64 = groups[0].iter().map(|&p| net.initial_marking[p]).sum();
    assert_eq!(
        init_sum, 1,
        "initial group sum must be 1 for 1-safe Process"
    );

    // Drive the real OneSafe observer over the full reachable state space.
    let config = ExplorationConfig::new(100_000);
    let full = explore_full(net, &config);
    assert!(full.graph.completed, "tiny net must fully explore");

    let mut observer = OneSafeObserver::new_colored(groups.clone());
    let mut max_group_sum = 0u64;
    for marking in &full.markings.unpack_all() {
        // Feed every reachable marking to the observer.
        use crate::explorer::ExplorationObserver;
        observer.on_new_state(marking);
        // Independent ground truth: deduplicated group sum.
        for group in &groups {
            let sum: u64 = group.iter().map(|&p| marking[p]).sum();
            max_group_sum = max_group_sum.max(sum);
        }
    }

    assert_eq!(
        max_group_sum, 1,
        "explicit truth: max deduplicated group sum is 1 (1-safe)"
    );
    assert!(
        observer.is_safe(),
        "OneSafe must be TRUE for the 1-safe Process place (bug gave spurious FALSE)"
    );
}

/// UpperBounds: a `tokens-count(Process)` query must bound `1`, not the
/// doubled `2` the bug produced via the duplicated alias indices.
///
/// Cross-checks the resolved place-bound multiset against the exact BFS
/// max token sum, and against the structural P-invariant set bound.
#[test]
fn test_place_named_after_sort_upper_bound_not_doubled() {
    use crate::explorer::{explore_full, ExplorationConfig};
    use crate::invariant::{compute_p_invariants, structural_set_bound};
    use crate::resolved_predicate::resolve_place_bound;

    let colored = parse_hlpnml(PLACE_NAMED_AFTER_SORT_PNML).expect("should parse");
    let unfolded = unfold_to_pt(&colored).expect("should unfold");
    let net = &unfolded.net;

    // `place-bound(Process)` resolves to the place's unfolded indices,
    // each EXACTLY once. The bug resolved to [i0,i1,i0,i1] → the structural
    // bound machinery read it as coefficient-2 → doubled bound.
    let indices = resolve_place_bound(&["Process".to_string()], &unfolded.aliases);
    assert_eq!(
        indices.len(),
        2,
        "place-bound(Process) must resolve to 2 indices once each; got {indices:?}",
    );
    let mut sorted: Vec<_> = indices.iter().map(|p| p.0).collect();
    sorted.sort_unstable();
    let mut dedup = sorted.clone();
    dedup.dedup();
    assert_eq!(
        sorted, dedup,
        "resolved place-bound must have no duplicates"
    );

    // Exact max token sum over Process via full BFS.
    let config = ExplorationConfig::new(100_000);
    let full = explore_full(net, &config);
    assert!(full.graph.completed, "tiny net must fully explore");
    let exact_max: u64 = full
        .markings
        .unpack_all()
        .iter()
        .map(|m| indices.iter().map(|p| m[p.0 as usize]).sum::<u64>())
        .max()
        .expect("at least the initial state");
    assert_eq!(exact_max, 1, "exact UpperBounds for Process is 1");

    // Structural P-invariant set bound over the deduplicated set must also
    // be 1 (the token-rotating transition conserves the single token).
    let invariants = compute_p_invariants(net);
    let places: Vec<usize> = indices.iter().map(|p| p.0 as usize).collect();
    let structural = structural_set_bound(&invariants, &places);
    assert_eq!(
        structural,
        Some(1),
        "structural set bound for Process must be 1, not the doubled 2",
    );
}

// ---------------------------------------------------------------------------
// `<subtract>` multiset-difference inscriptions (DatabaseWithMutex-COL,
// PhilosophersDyn-COL, PolyORBLF-COL). The "broadcast to all-but-self"
// pattern `1'all - 1'(x)` and its n-ary `1'all - 1'(x) - 1'(y)` form.
// ---------------------------------------------------------------------------

/// Sort `S = {a, b}`; transition `t(x:S)` with output arc `S->t` carrying
/// `1'all - 1'(x)` (all colors of S except the bound `x`). Mirrors the
/// DatabaseWithMutex `1'[(site.all),(f)] - 1'[(s),(f)]` broadcast-minus-self
/// pattern reduced to a scalar sort.
const SUBTRACT_BROADCAST_MINUS_SELF_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="sub" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="p0">
      <place id="P"><name><text>P</text></name>
        <type><structure><usersort declaration="S"/></structure></type>
      </place>
      <transition id="t"><name><text>t</text></name></transition>
      <arc id="a1" source="t" target="P">
        <hlinscription><structure>
          <subtract>
            <subterm><all><usersort declaration="S"/></all></subterm>
            <subterm>
              <numberof>
                <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
                <subterm><variable refvariable="vx"/></subterm>
              </numberof>
            </subterm>
          </subtract>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="S" name="S">
        <cyclicenumeration>
          <feconstant id="a" name="a"/>
          <feconstant id="b" name="b"/>
        </cyclicenumeration>
      </namedsort>
      <variabledecl id="vx" name="x"><usersort declaration="S"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn test_unfold_subtract_broadcast_minus_self() {
    let colored =
        parse_hlpnml(SUBTRACT_BROADCAST_MINUS_SELF_PNML).expect("subtract net must parse");
    let unfolded = unfold_to_pt(&colored).expect("subtract net must unfold");

    // S has 2 colors ⇒ place P unfolds to 2 P/T places (P_a=0, P_b=1).
    assert_eq!(unfolded.net.num_places(), 2, "S has cardinality 2");
    // 1 variable over a 2-color sort ⇒ 2 bindings ⇒ 2 transitions.
    assert_eq!(unfolded.net.num_transitions(), 2, "two bindings of x");

    // For binding x=a: `1'all - 1'(a) = 1'(b)` ⇒ exactly one output token on
    // P_b, none on P_a (the subtracted color cancels). For x=b: `1'(a)`.
    // Collect the multiset of (output place, weight) sets across both bindings.
    let mut output_sets: Vec<Vec<(u32, u64)>> = unfolded
        .net
        .transitions
        .iter()
        .map(|t| {
            let mut v: Vec<(u32, u64)> = t.outputs.iter().map(|a| (a.place.0, a.weight)).collect();
            v.sort();
            v
        })
        .collect();
    output_sets.sort();

    // Each binding leaves exactly the single *other* color with weight 1; the
    // self color is fully cancelled (truncated subtraction, no zero-weight arc).
    assert_eq!(
        output_sets,
        vec![vec![(0u32, 1u64)], vec![(1u32, 1u64)]],
        "all-but-self must yield exactly one unit token on the complementary color"
    );
}

/// n-ary subtract `1'all - 1'(x) - 1'(succ(x))` over `S = {a, b, c}`: removes
/// both `x` and its cyclic successor, leaving the single remaining color.
/// Exercises the left-associative folding of multiple subtrahends
/// (PhilosophersDyn-COL `1'all - 1'(p) - 1'(q)` shape).
const SUBTRACT_NARY_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="subn" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="p0">
      <place id="P"><name><text>P</text></name>
        <type><structure><usersort declaration="S"/></structure></type>
      </place>
      <transition id="t"><name><text>t</text></name></transition>
      <arc id="a1" source="t" target="P">
        <hlinscription><structure>
          <subtract>
            <subterm><all><usersort declaration="S"/></all></subterm>
            <subterm>
              <numberof>
                <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
                <subterm><variable refvariable="vx"/></subterm>
              </numberof>
            </subterm>
            <subterm>
              <numberof>
                <subterm><numberconstant value="1"><positive/></numberconstant></subterm>
                <subterm><successor><subterm><variable refvariable="vx"/></subterm></successor></subterm>
              </numberof>
            </subterm>
          </subtract>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="S" name="S">
        <cyclicenumeration>
          <feconstant id="a" name="a"/>
          <feconstant id="b" name="b"/>
          <feconstant id="c" name="c"/>
        </cyclicenumeration>
      </namedsort>
      <variabledecl id="vx" name="x"><usersort declaration="S"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn test_unfold_subtract_nary_removes_two_colors() {
    let colored = parse_hlpnml(SUBTRACT_NARY_PNML).expect("n-ary subtract net must parse");
    let unfolded = unfold_to_pt(&colored).expect("n-ary subtract net must unfold");

    assert_eq!(unfolded.net.num_places(), 3, "S has cardinality 3");
    assert_eq!(unfolded.net.num_transitions(), 3, "three bindings of x");

    // For each binding x=k: `1'all - 1'(k) - 1'(succ(k))` leaves exactly the
    // one remaining color (the predecessor of k, equivalently succ(succ(k))),
    // each with weight 1. x=a(0): removes a,b ⇒ leaves c(2). x=b(1): removes
    // b,c ⇒ leaves a(0). x=c(2): removes c,a ⇒ leaves b(1).
    let mut output_sets: Vec<Vec<(u32, u64)>> = unfolded
        .net
        .transitions
        .iter()
        .map(|t| {
            let mut v: Vec<(u32, u64)> = t.outputs.iter().map(|a| (a.place.0, a.weight)).collect();
            v.sort();
            v
        })
        .collect();
    output_sets.sort();

    assert_eq!(
        output_sets,
        vec![vec![(0u32, 1u64)], vec![(1u32, 1u64)], vec![(2u32, 1u64)]],
        "each binding leaves exactly the single non-subtracted color"
    );
}

#[test]
fn test_parse_subtract_fails_closed_for_single_operand() {
    // A `<subtract>` with only one subterm has no well-defined difference;
    // it must fail-closed (→ whole-model CANNOT_COMPUTE) rather than silently
    // dropping the missing subtrahend.
    let pnml = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="bad" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="p0">
      <place id="P"><name><text>P</text></name>
        <type><structure><usersort declaration="S"/></structure></type>
      </place>
      <transition id="t"><name><text>t</text></name></transition>
      <arc id="a1" source="t" target="P">
        <hlinscription><structure>
          <subtract>
            <subterm><all><usersort declaration="S"/></all></subterm>
          </subtract>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="S" name="S">
        <cyclicenumeration><feconstant id="a" name="a"/></cyclicenumeration>
      </namedsort>
    </declarations></structure></declaration>
  </net>
</pnml>"#;
    match parse_hlpnml(pnml) {
        Err(PnmlError::MissingElement(msg)) => {
            assert!(
                msg.contains("subtract requires at least 2 subterms"),
                "unexpected error message: {msg}"
            );
        }
        other => panic!("expected MissingElement for 1-operand subtract, got {other:?}"),
    }
}
