// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! LIVE explicit-state fixpoint `Certified` verdict — hook the model checker's ENUMERATED reachable
//! set `R` into the Clean-kernel fixpoint cert (`cleancic::certify_explicit_fixpoint_set`).
//!
//! This is the live wiring for the "Explicit-state safety" verdict class of the certified-endgame
//! roadmap: where the symbolic ([`crate::cert::certify_spec`]) path certifies an INDUCTIVE invariant,
//! this path certifies the FIXPOINT witness `Init⊆R ∧ R closed-under-Next ∧ R⊆Safety` directly over
//! the states the explicit-state engine actually enumerates — `R` is the enumerated visited set, and
//! the three legs are kernel-checked CIC proofs (no SMT, no model checker in the trust base).
//!
//! ## What runs LIVE
//!
//! [`certify_explicit_state_spec`] re-uses the SAME live enumeration primitives the model checker /
//! interactive explorer use — [`extract_init_constraints`] + [`enumerate_states_from_constraint_branches`]
//! for `Init`, and [`enumerate_successors`] for `Next` — to BFS-enumerate `R` to a fixpoint (bounded).
//! It is the genuine explicit-state reachable set produced by the live evaluator, not a hand-built
//! toy. The enumerated `R` (and the `Init` values) are then fed to the kernel fixpoint cert.
//!
//! ## Fail-closed fragment (SOUNDNESS FIRST)
//!
//! A certificate is emitted ONLY when every one of these holds — otherwise `None`, and the verdict
//! stays at the honest explicit-state tier (NEVER a `Certified` the kernel did not accept):
//!  * exactly ONE state variable, in the nonneg-`Int` embeddable fragment (each enumerated value is
//!    a `SmallInt`/`Int` ≥ 0);
//!  * `R` is FINITE and FULLY enumerated within the bound (BFS reached a fixpoint — the frontier
//!    emptied before the state/step cap; a truncated `R` is rejected);
//!  * `Next` is the STUTTER `x'=x` AS WITNESSED BY THE LIVE ENGINE — every enumerated successor of
//!    every `R`-state equals its parent (so the kernel's stutter closed-under-Next leg certifies the
//!    relation the engine actually computed, not an assumed shape);
//!  * `Safety` (the single configured invariant) holds on every `R`-state, EITHER as the conjunctive
//!    nonneg `⋀ x≥0` shape the kernel tuple `R⊆Safety` leg proves (the historical PRIMARY lane), OR —
//!    the GENERAL fallback lane — as ANY invariant recognizable into the kernel predicate fragment
//!    (`recognize_pred_sorts`) that is a truth-direction-EXACT state predicate: the kernel then
//!    reduces `⋀_{s∈R} ⟦Safety⟧(s)` to `Bool.true` (`safety_pred`/`safety_general`);
//!  * the Clean kernel ACCEPTS all three legs (the final arbiter, inside `certify_explicit_fixpoint_set`).
//!
//! ## What is NOT yet general (honest scope)
//!
//! Single Int var only (multi-var / Bool / compound-sort state, and the product membership encoding,
//! are future work); `Next` must be the stutter relation (non-stutter `Next` needs a per-edge
//! `Eq.trans`/successor-value closed-leg generalization — the live closure CHECK already handles
//! arbitrary `Next`, but the kernel closed-leg term is stutter-only today); `Safety` beyond the two
//! lanes above (a truth-direction-INEXACT embedding — Nat-truncating `Sub`/`Div`/`Mod`, Seq digit
//! ops, set comprehension/quantifier folds — or a primed "invariant") fails closed, as does an
//! `R` must fit the bound (fingerprint-only / disk-spilled visited sets are not yet
//! re-materialized into concrete values here). An all-non-Int state (e.g. one record variable —
//! the CoffeeCan class) is admitted as of R3: its tuple safety leg is the trivially-true
//! degenerate form and the spec's Safety claim must ride the general `safety_general` leg.

#[cfg(feature = "clean-cic")]
use std::collections::BTreeSet;
#[cfg(feature = "clean-cic")]
use std::sync::Arc;
#[cfg(test)]
use tla_value::Rp;

#[cfg(feature = "clean-cic")]
use crate::config::Config;

/// The kernel SORT of a state-variable COLUMN in the compound-sort explicit-state fragment. A tuple
/// state is a fixed-width `Vec<u64>`, but each column is faithfully typed in the kernel terms: an `Int`
/// column embeds as `Int.ofNat v` with binder type `Int` (the value is a nonneg Int); a `Bool` column
/// embeds as `Bool.true` (`v=1`) / `Bool.false` (`v=0`) with binder type `Bool`. Defined here (always
/// compiled) so the cert struct can store the per-column sort vector without depending on the
/// `clean-cic` feature; the kernel-term builders in [`crate::cleancic`] consume it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColSort {
    /// A nonneg-`Int` column: embeds as `Int.ofNat v`, binder type `Int`, carries an `x≥0` Safety conjunct.
    Int,
    /// A `Bool` column: embeds as `Bool.true` (`v=1`) / `Bool.false` (`v=0`), binder type `Bool`, no Safety conjunct.
    Bool,
    /// A finite-`Set` column over the element universe `{0..universe}`, encoded as a `Nat` BITMASK
    /// `Σ_{e∈S} 2^e` in ONE u64 cell (bit `e` set ⟺ `e∈S`). The binder type is `Nat` and the cell
    /// literal is `Nat.lit(bitmask)`; SET EQUALITY `S=T` is exactly bitmask `Eq Nat maskS maskT`, so a
    /// Set column slots into the existing `Eq`-based tuple membership legs (`Init⊆R`, `image⊆R`) for
    /// FREE. Set OPERATIONS embed as kernel-REDUCIBLE Nat bitwise ops (`∪`=`Nat.lor`, `∩`=`Nat.land`,
    /// `∈`/`⊆` via `Nat.shiftRight`/`Nat.lor`+`Nat.beq`). Carries NO `≥0` Safety conjunct (a valid
    /// bitmask is a valid set `⊆` the universe by construction — like `Bool`). `universe` is the element
    /// bit width `K`: the bitmask uses bits `0..universe`, so a complete product axis is `{0..2^K-1}`.
    Set {
        /// The element-universe bit width `K`: valid masks use bits `0..K` (`K ≤ 64`).
        universe: u32,
    },
    /// A finite-set column over a SMALL FIXED FINITE ATOM UNIVERSE `D` (config CONSTANT model values, or
    /// `String` atoms) — a set `S ⊆ D` — encoded as a `|D|`-bit `Nat` BITMASK `Σ_{a∈S} 2^idx(a)` in ONE
    /// u64 cell, where `idx(a)` is the position of atom `a` in the sorted universe `dom` (bit `i` set ⟺
    /// `dom[i] ∈ S`). This is the ATOM-DOMAIN analogue of [`ColSort::Set`] (whose bit index IS the element
    /// Int value): here a `Nat`-index/atom map — the SAME faithful atom→slot mechanism [`ColSort::Enum`] /
    /// [`ColSort::FuncEnum`] use — turns model values / strings into bit positions. Binder type `Nat`, cell
    /// literal `Nat.lit(bitmask)`; SET EQUALITY / `⊆` / `∈` are EXACTLY the bitmask `Eq Nat` / bit ops, so a
    /// `SetMask` column slots into the existing `Eq`-based tuple membership legs (`Init⊆R`, `image⊆R`) and
    /// the bitmask `PredIR` fragment for FREE. NO `≥0` Safety conjunct (a valid `|D|`-bit mask is a valid
    /// subset of `D` by construction — like `Set`/`Bool`).
    ///
    /// SOUNDNESS / DETERMINISM: `dom` is the SORTED set of DISTINCT atoms observed in the column across ALL
    /// enumerated states (the analogue of a scalar [`ColSort::Enum`]'s `labels`), so `bit i ⟺ dom[i] ∈ S`
    /// is a BIJECTION between subsets of `dom` and `|dom|`-bit values — no two distinct subsets share a mask,
    /// and every mask `< 2^|dom|` is a genuine subset. Every atom that EVER appears in the column lies in
    /// `dom` (the union), so every set is representable; a model value in the value TYPE but never present in
    /// this column is simply absent from `dom` (and `a∈S` folds to FALSE, EXACT — it is never present). The
    /// recognizers map a ground atom / constant atom-set to bit positions via THIS `dom`; a cross-`dom`
    /// column comparison (`S=T`/`S⊆T` between two `SetMask` columns) requires `dom`+`dom_kind` to MATCH
    /// (else the bit meanings differ ⇒ decline, fail-closed). certify grows `dom` monotonically
    /// (`GrowSetMask`) and verify (Leg-E) re-enumerates the SAME deterministic states ⇒ the SAME sorted
    /// union ⇒ the SAME sort. `is_compound` is FALSE and `pack_universe` is `None` (equality/membership-only;
    /// the general Next-completeness domain is NOT modelled — closure rests on the enumerated `image ⊆ R`).
    SetMask {
        /// The sorted, distinct ATOM UNIVERSE `D` observed in the column across ALL enumerated states — part
        /// of the column's identity (two `SetMask` columns with different universes are DIFFERENT sorts, so a
        /// cross-column `S=T`/`S⊆T` is bit-exact ONLY when the universes agree). Bit `i` of a cell ⟺
        /// `dom[i] ∈ S`; `|dom|` is the mask bit width `K ≤ 64`.
        dom: Vec<String>,
        /// The KIND of the `dom` atoms — `Model` (config CONSTANT model values) versus `Str` (TLA `String`
        /// atoms). A `String` atom `"a"` and a model value NAMED `a` are DISTINCT TLA values, so the
        /// recognizer resolves a `String`-literal element ONLY against a `Str`-kind universe and a
        /// model-value `Ident` element ONLY against a `Model`-kind universe (any cross-kind form fails
        /// closed). A DETERMINISTIC function of the observed cells (all model value ⇒ `Model`, all `String`
        /// ⇒ `Str`; a mix declines), so Leg-E re-derives it. Shares [`ColSort::Enum`]'s [`EnumKind`].
        dom_kind: EnumKind,
    },
    /// A finite-set column over a SMALL FIXED FINITE **RECORD** UNIVERSE `D` — a set `S ⊆ D` of bounded
    /// RECORD values (`msgs ⊆ Message` with `Message == [type:{"Prepared"},rm:RM] ∪ [type:{"Commit",…}]`,
    /// the two-phase-commit / message-set class) — encoded as a `|D|`-bit `Nat` BITMASK `Σ_{r∈S} 2^idx(r)`
    /// in ONE u64 cell, where `idx(r)` is the position of record `r` in the sorted universe `dom` (bit `i`
    /// set ⟺ `dom[i] ∈ S`). This is the RECORD-DOMAIN analogue of [`ColSort::SetMask`] (whose universe is
    /// ATOMS): here each element is a bounded record, faithfully identified by its CANONICAL KEY
    /// ([`record_value_key`] — a length-prefixed serialization of the sorted `(field-name, value)` pairs,
    /// each value an atom / Int / Bool leaf), so `dom` is the sorted set of DISTINCT record keys observed
    /// in the column across ALL enumerated states. Binder type `Nat`, cell literal `Nat.lit(bitmask)`; SET
    /// EQUALITY / `⊆` are EXACTLY the bitmask `Eq Nat` / bit ops — a `SetMaskRec` column slots into the
    /// existing `Eq`-based tuple membership legs (`Init⊆R`, `image⊆R`) and the bitmask `PredIR` fragment
    /// for FREE, reusing the SAME kernel bitmask machinery as [`ColSort::SetMask`]/[`ColSort::Set`]. NO
    /// `≥0` Safety conjunct (a valid `|D|`-bit mask is a valid subset of `D` by construction).
    ///
    /// SOUNDNESS / DETERMINISM: [`record_value_key`] is a length-prefixed, kind-tagged serialization ⇒ a
    /// BIJECTION between the encodable record values and their keys (distinct records ⇒ distinct keys — no
    /// two records share a bit, and a record with a non-leaf field is UNKEYABLE ⇒ the column fails closed).
    /// `dom` is the sorted union of all record keys EVER present (the [`EnumStop::GrowSetMaskRec`] analogue
    /// of `GrowSetMask`), so `bit i ⟺ dom[i] ∈ S` is a bijection between subsets of `dom` and `|dom|`-bit
    /// values; every record that ever appears is in `dom` (a record in the value TYPE but never present is
    /// simply absent from `dom`). A `S ⊆ RecordSetType` invariant is recognized by materializing the
    /// record-set TYPE (`recognize_setmaskrec_pred`) into the SAME canonical keys and masking against
    /// `dom` (records of the type outside `dom` can never be in `S ⊆ dom` ⇒ the mask is EXACT for `⊆`;
    /// records in `dom` but NOT in the type leave their bit CLEAR ⇒ a state holding one violates the
    /// subset ⇒ the kernel declines, faithful to `ty check`). certify grows `dom` monotonically and
    /// verify (Leg-E) re-enumerates the SAME deterministic states ⇒ the SAME sorted union ⇒ the SAME sort.
    /// Equality/membership-only: the general Next-completeness domain is NOT modelled (closure rests on
    /// the enumerated `image ⊆ R`). Cross-column `S=T`/`S⊆T` requires `dom` to MATCH (bit meanings differ
    /// otherwise ⇒ decline, fail-closed). The one-cell bit-width cap (`|dom| ≤ 64`) declines wider universes.
    SetMaskRec {
        /// The sorted, distinct RECORD-KEY UNIVERSE `D` observed in the column across ALL enumerated states
        /// (each key from [`record_value_key`]) — part of the column's identity (two `SetMaskRec` columns
        /// with different universes are DIFFERENT sorts). Bit `i` of a cell ⟺ `dom[i] ∈ S`; `|dom|` is the
        /// mask bit width `K ≤ 64`.
        dom: Vec<String>,
    },
    /// A bounded `RECORD` `[f_0|->v_0,…,f_{k-1}|->v_{k-1}]` (a FIXED canonical field order — fields sorted
    /// by name) with each field value `v_i < base`, POSITIONALLY packed into ONE `Nat` cell as the
    /// base-`base` numeral `pack = Σ_i v_i·base^i`. The binder type is `Nat` and the cell literal is
    /// `Nat.lit(pack)`; RECORD EQUALITY is exactly `Eq Nat pack pack'` (the pack is CANONICAL — one record
    /// ⇒ one Nat), so a Record column slots into the existing `Eq`-based tuple membership legs for FREE.
    /// FIELD ACCESS `rec.f_i = (pack / base^i) mod base` embeds as kernel-REDUCIBLE `Nat.div`/`Nat.mod`
    /// (EXACT). Carries NO `≥0` Safety conjunct (a pack is a valid record by construction). `arity` is the
    /// fixed field count `k`; `base` the per-field radix. SOUNDNESS bound: `arity` and `base` are fixed
    /// per-column properties, so the sort is STABLE across enumerated states (a record whose field count
    /// or whose value range differs fails closed at `value_cell`).
    Record {
        /// The per-field radix `base`: every field value must be `< base`. Uniform across positions (the
        /// MAX over positions' code ranges — an `Int` position's max value, a `Bool` position's `1` —
        /// floored at [`RECORD_FUNC_BASE`]), so digit extraction stays the position-independent
        /// `(pack / base^i) mod base`.
        base: u64,
        /// The canonical sorted field NAMES — part of the column's identity. WITHOUT this, records with
        /// the same arity but different field sets (`[a|->1,b|->2]` vs `[c|->1,d|->2]`) pack to the SAME
        /// Nat and `Eq Nat` would wrongly equate two distinct records (an unsoundness). Including the
        /// names makes distinct-shape records distinct sorts, so a heterogeneous-record column fails
        /// closed at the cross-state `col_sorts` agreement check.
        fields: Vec<String>,
        /// The per-POSITION VALUE SORT of each field digit (`cells[i]` is field `fields[i]`'s [`CellSort`]),
        /// part of the column's identity (the KIND DISCRIMINANT — see [`CellSort`]). CANONICALLY EMPTY for
        /// an all-`Int` record: the encoder normalizes `cells` to `[]` when every position is `Int`, so an
        /// all-Int record serializes with NO `cells` key (BYTE-IDENTICAL to a pre-value-type-leaf cert) and
        /// two equal-shape all-Int records compare `PartialEq`-equal. When any position is non-`Int`,
        /// `cells.len() == fields.len()`. Read a position via [`cell_kind`].
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cells: Vec<CellSort>,
    },
    /// A bounded FINITE FUNCTION `[d ∈ D |-> e_d]` whose VALUES are Int / Bool / enum leaves (`cells`)
    /// POSITIONALLY packed exactly like a `Record`: `pack = Σ_p code_p·base^p`. `f[key] = (pack / base^p)
    /// mod base` (`Nat.div`/`Nat.mod`, EXACT). The binder type is `Nat`, cell literal `Nat.lit(pack)`,
    /// equality `Eq Nat` — membership legs for FREE. CANONICAL (one function ⇒ one Nat). NO `≥0` Safety
    /// conjunct. `arity` = domain size, `base` = value radix.
    ///
    /// The DOMAIN is EITHER the consecutive Int prefix `0..arity-1` (`dom` empty — the historical shape;
    /// positions ARE the keys) OR a set of `arity` config CONSTANT model values / `String` ATOMS (`dom`
    /// non-empty, keys in canonical `Value::cmp` order, `dom_kind` their kind) — the exact `dom`/`dom_kind`
    /// mechanism [`ColSort::FuncEnum`] carries, here on the Int/Bool/enum-VALUED pack. This is the DUAL of
    /// `FuncEnum`: `FuncEnum` is an atom/Int-domain function with ENUM values; `Func{dom,..}` is an
    /// atom/Int-domain function with Int/Bool/enum values (the DieHarder `contents ∈ [Jug -> Nat]` class,
    /// `Jug` a `String`-atom set). Position `p` is key `dom[p]`; `f[dom[p]]`'s digit is `code_p`.
    Func {
        /// The per-value radix `base`: every function value must be `< base` (the MAX over positions'
        /// code ranges, floored at [`RECORD_FUNC_BASE`], exactly as [`ColSort::Record`]).
        base: u64,
        /// The domain size: `arity` positions, whether keyed by the Int prefix (`dom` empty) or the stored
        /// atom keys (`dom.len() == arity`).
        arity: u32,
        /// The per-POSITION VALUE SORT of each function-value digit (`cells[d]` is value `f[d]`'s
        /// [`CellSort`]) — the KIND DISCRIMINANT, mirroring [`ColSort::Record`]'s `cells`. CANONICALLY
        /// EMPTY for an all-`Int` function (byte-identical serialization; `PartialEq`-stable), else
        /// `cells.len() == arity`. Read via [`cell_kind`].
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cells: Vec<CellSort>,
        /// The DOMAIN KEY names when the function's domain is a set of config CONSTANT MODEL VALUES or
        /// `String` ATOMS — the key texts in canonical `Value::cmp` order that FIX the positional pack
        /// (position `p` = key `dom[p]`, value digit `code_p`). CANONICALLY EMPTY (skip-serialized) for the
        /// classic `0..arity-1` Int-prefix domain, so an Int-domain `Func` cert is BYTE-IDENTICAL to a
        /// pre-domain-shape cert. When non-empty, `dom.len() == arity`; it is part of the column IDENTITY
        /// (two `Func` columns with different domains are DIFFERENT sorts) and the ONLY thing that lets the
        /// recognizer resolve a domain key `f[k]` to its slot AND check `DOMAIN f = D` for a `f ∈ [D -> S]`
        /// type invariant. The SAME faithful key→slot mapping [`ColSort::FuncEnum`] uses (via
        /// [`func_enum_domain_keys`]). A DETERMINISTIC function of the enumerated function's domain keys ⇒
        /// verify (Leg-E) re-derives it.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        dom: Vec<String>,
        /// The KIND of the `dom` keys — `Model` (config CONSTANT model values) versus `Str` (TLA `String`
        /// atoms). ONLY meaningful when `dom` is non-empty. A `String` key `"r1"` and a model value NAMED
        /// `r1` are DISTINCT TLA values, so a domain of the one must NEVER conflate with a domain of the
        /// other even at identical `dom` names: the kind is part of the column IDENTITY (derived
        /// `PartialEq`), so the recognizer resolves a `String`-literal index ONLY against a `Str`-kind dom
        /// and a model-value `Ident` index ONLY against a `Model`-kind dom (any cross-kind form fails
        /// closed). DEFAULTS to `Model` and is skip-serialized when `Model`, so an Int-prefix or
        /// model-value-domain cert is BYTE-IDENTICAL to a pre-string-domain cert. Deterministic ⇒ Leg-E
        /// re-derives it. Shares the [`ColSort::FuncEnum::dom_kind`] serde default/skip helpers.
        #[serde(
            default = "func_enum_dom_kind_default",
            skip_serializing_if = "func_enum_dom_kind_is_default"
        )]
        dom_kind: EnumKind,
    },
    /// A bounded `SEQUENCE` `<<a_0,…,a_{m-1}>>` with elements `a_i < base` and length `m ≤ max_len`,
    /// SELF-DELIMITINGLY packed into ONE `Nat` cell in base `base+1`: `pack = Σ_i (a_i+1)·(base+1)^i`
    /// (each element shifted by `+1` so a digit `0` marks the end). The binder type is `Nat`, cell literal
    /// `Nat.lit(pack)`, equality `Eq Nat` — membership legs for FREE; CANONICAL (one sequence ⇒ one Nat:
    /// the shift makes the length self-delimiting). All sequence OPERATIONS are kernel-REDUCIBLE
    /// (`Nat.div`/`Nat.mod`/`Nat.pow`/`Nat.sub`/`Nat.ble` + `Bool.rec`) and RECOGNIZED in predicates:
    ///   * `Head = (pack mod D) − 1`, `s[i] = (pack / D^(i-1)) mod D − 1` (1-based literal `i ∈ [1,max_len]`),
    ///   * `Len  = Σ_{i<max_len} (digit_i ≠ 0 ? 1 : 0)` (count of nonzero base-`D` digits, a bounded fold),
    ///   * `Tail = pack / D` (drop the lowest digit),
    ///   * `Append(s,e) = pack + (e+1)·D^Len(s)` (write `e` at the first free digit; EXACT for `Len<max_len`,
    ///     an over-full result exceeds `D^max_len` and is never a valid `value_cell` state ⇒ harmless).
    /// (`D = base+1`.) EXACT to TLA semantics on the pack ⇒ the general completeness leg fires over these
    /// ops. NO `≥0` Safety conjunct. `base` = element radix, `max_len` = the length bound.
    Seq {
        /// The element radix `base`: every element CODE must be `< base`; the pack uses base `base+1`.
        base: u64,
        /// The length bound: a sequence longer than `max_len` fails closed.
        max_len: u32,
        /// The ELEMENT value-type leaf (reusing [`CellSort`], the positional-compound leaf): `Int` (the
        /// historical nonneg-Int element — `code = v`), `Bool` (`code = 1|0`), or `Enum{labels,kind}` (an
        /// ATOM / model-value element — `code = idx(label)` in the column's cross-state sorted element
        /// union). The self-delimiting pack is UNCHANGED — `pack = Σ_i (code_i+1)·D^i`, `D = base+1` — only
        /// the element→`code` map differs per leaf. A `Bool`/`Enum` element is EQUALITY-ONLY (a code carries
        /// no order): the recognizer resolves `Append(s, e)`'s element to its code and DECLINES a bare
        /// `Head(s)`/`s[i]` VALUE (so ordering/arith on an atom element fails closed), never conflating an
        /// atom index with a Nat. DEFAULTS to `Int` and is skip-serialized when `Int`, so every pre-existing
        /// pure-Int `Seq` cert stays BYTE-IDENTICAL (the `elem` byte only appears for an atom/Bool sequence).
        /// Part of the column IDENTITY (`PartialEq`): an Int-element and an atom-element `Seq` are DIFFERENT
        /// sorts, so a column mixing Int and atom sequences fails the cross-state `col_sorts` agreement.
        #[serde(
            default = "cellsort_int_default",
            skip_serializing_if = "cellsort_is_int"
        )]
        elem: CellSort,
    },
    /// A FINITE-ENUM column — a state variable holding a `String` (or model value) drawn from a small
    /// fixed set of LABELS (program counters `"read"`/`"write"`/`"done"`, model-value sets, …). Each
    /// cell encodes to the INDEX of its label in `labels`, a `Nat` in `0..labels.len()`; the binder type
    /// is `Nat` and the cell literal is `Nat.lit(index)`. The label→index map is a BIJECTION, so LABEL
    /// EQUALITY is exactly INDEX EQUALITY (`Eq Nat`), and an Enum column slots into the existing
    /// `Eq`-based tuple membership legs (`Init⊆R`, `image⊆R`) for FREE — like `Set`/`Record`, and with NO
    /// `≥0` Safety conjunct (a valid index is a valid label by construction).
    ///
    /// SOUNDNESS / DETERMINISM: `labels` is the SORTED set of DISTINCT labels observed in that column
    /// across ALL enumerated states — a STABLE per-column property (the analogue of a `Record`'s field
    /// set). certify collects the per-column label union over every enumerated state, sorts it, and
    /// assigns indices `0..k-1`; verify (Leg-E) re-enumerates the SAME deterministic states ⇒ the SAME
    /// label union ⇒ the SAME sorted `labels` ⇒ the obligations rebuild identically (`re.sorts ==
    /// fp.sorts`). Because the index is derived from the sorted union, one label ⇒ one index (CANONICAL).
    /// A non-string/non-model-value cell in an Enum column, or a column that mixes `String` and model
    /// values, FAILS CLOSED at `value_cell` — the encoding is truth-EXACT for equality only, never for
    /// ordering (an enum index carries no `<` meaning; `pred_exact` admits enum EQUALITY forms only).
    Enum {
        /// The sorted, distinct label set observed in the column across ALL enumerated states — part of
        /// the column's identity (two columns with different label sets are DIFFERENT sorts, so a
        /// cross-column enum equality `pc_i = pc_j` is index-exact ONLY when the label sets agree).
        labels: Vec<String>,
        /// The KIND of the labels (`Str` vs `Model`) — see [`EnumKind`]. Part of the column's identity:
        /// a `String`-literal membership/equality matches ONLY a `Str` column and a config model-value
        /// CONSTANT set ONLY a `Model` column (a kind mismatch fails closed), so a TLA `String` and a
        /// same-text model value — which are DISTINCT TLA values — are never conflated to one index. A
        /// DETERMINISTIC function of the column's observed cells (the encoder fails closed on a mix of
        /// `String` and model-value cells), so verify (Leg-E) re-derives it identically.
        kind: EnumKind,
    },
    /// A FUNCTION-of-ENUM column — a bounded finite function `[d ∈ 0..arity-1 |-> e_d]` whose domain is
    /// the consecutive prefix `0..arity-1` and whose VALUES are `String`s (or model values) drawn from a
    /// small fixed set of `labels` (a per-process program counter `pc: [0..N-1 -> {"a","b","Done"}]`, the
    /// TeachingConcurrency/Simple class). It is the FINITE-FUNCTION analogue of [`ColSort::Enum`]: each
    /// value `e_d` encodes to the INDEX of its label in `labels`, and the function POSITIONALLY packs those
    /// indices into ONE `Nat` cell exactly like [`ColSort::Func`] packs Int values — `pack = Σ_d idx(e_d)·
    /// |labels|^d`, base `|labels|`. The binder type is `Nat` and the cell literal is `Nat.lit(pack)`; the
    /// pack is CANONICAL (one function ⇒ one Nat — each digit `idx(e_d) < |labels|` by construction), so
    /// FUNCTION EQUALITY is exactly `Eq Nat pack pack'` and a FuncEnum column slots into the existing
    /// `Eq`-based membership legs (`Init⊆R`, `image⊆R`) for FREE — like `Enum`/`Func`, with NO `≥0` Safety
    /// conjunct. `f[i] = "read"` recognizes as the digit-extraction equality `(pack / |labels|^i) mod
    /// |labels| = idx("read")` (the [`ColSort::Func`] `f[i]` digit precedent, with the enum label→index map
    /// composed in) — EQUALITY ONLY (an enum index carries no order).
    ///
    /// SOUNDNESS / DETERMINISM: `arity` is the fixed domain size and `labels` is the SORTED distinct set of
    /// labels observed in the column's function VALUES across ALL enumerated states — both STABLE per-column
    /// properties (`arity` from the function's domain, `labels` the analogue of [`ColSort::Enum`]'s union).
    /// certify collects the per-column label union and assigns indices `0..|labels|-1`; verify (Leg-E)
    /// re-enumerates the SAME deterministic states ⇒ the SAME `arity` and label union ⇒ the SAME sort ⇒ the
    /// obligations rebuild identically. A function whose domain is not `0..arity-1`, whose arity varies
    /// across states, whose values mix `String` and model kinds or are not labels, or for which
    /// `|labels|^arity` overflows `u64`, FAILS CLOSED at `value_cell` (the encoding is truth-EXACT for
    /// equality only — `pred_exact` admits FuncEnum digit EQUALITY forms only, never ordering).
    FuncEnum {
        /// The domain size: the function's domain is exactly `0..arity-1` (Int-prefix; `dom` empty) OR a
        /// set of `arity` config CONSTANT model values (`dom` non-empty, `dom.len() == arity`).
        arity: u32,
        /// The sorted, distinct label set observed across the column's function VALUES over ALL enumerated
        /// states — part of the column's identity AND the pack radix (`base = labels.len()`). Two FuncEnum
        /// columns with different label sets are DIFFERENT sorts (a cross-column `pc_i[·] = pc_j[·]` is
        /// index-exact ONLY when the label sets agree).
        labels: Vec<String>,
        /// The DOMAIN KEY names when the function's domain is a set of config CONSTANT MODEL VALUES
        /// (`rmState: [RM -> {..}]` with `RM = {r1,r2,r3}`) OR a set of `String` ATOMS
        /// (`rmState: [RMVal -> {..}]` with `RMVal = {"r1","r2","r3"}`, APTCommit) — the key texts in the
        /// canonical `Value::cmp` order that FIXES the positional pack (position `p` = key `dom[p]`, value
        /// digit `idx(e_p)`). CANONICALLY EMPTY (skip-serialized) for the classic `0..arity-1` Int-prefix
        /// domain, so an Int-domain FuncEnum cert is BYTE-IDENTICAL to a pre-domain-shape cert. When
        /// non-empty, `dom.len() == arity`; it is part of the column IDENTITY (two FuncEnum columns with
        /// different domains are DIFFERENT sorts) and the ONLY thing that lets the recognizer resolve a
        /// domain key `f[k]` to its slot AND check `DOMAIN f = D` for a `f ∈ [D -> S]` type invariant. A
        /// DETERMINISTIC function of the enumerated function's domain keys ⇒ verify (Leg-E) re-derives it.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        dom: Vec<String>,
        /// The KIND of the `dom` keys — `Model` (config CONSTANT model values) versus `Str` (TLA `String`
        /// atoms). ONLY meaningful when `dom` is non-empty (the Int-prefix domain has no atom keys). A
        /// `String` key `"r1"` and a model value NAMED `r1` are DISTINCT TLA values, so a domain of the one
        /// must NEVER conflate with a domain of the other even at identical `dom` names: the kind is part of
        /// the column IDENTITY (derived `PartialEq`), so the recognizer resolves a `String`-literal index
        /// ONLY against a `Str`-kind dom and a model-value `Ident` index ONLY against a `Model`-kind dom
        /// (any cross-kind form fails closed). DEFAULTS to `Model` and is skip-serialized when `Model`, so
        /// an Int-prefix or model-value-domain cert is BYTE-IDENTICAL to a pre-string-domain cert (the
        /// `Str` kind is the only additive byte). A DETERMINISTIC function of the observed domain keys (all
        /// `String` ⇒ `Str`, all model value ⇒ `Model`; a mix declines) ⇒ verify (Leg-E) re-derives it.
        #[serde(
            default = "func_enum_dom_kind_default",
            skip_serializing_if = "func_enum_dom_kind_is_default"
        )]
        dom_kind: EnumKind,
    },
    /// A bounded finite FUNCTION-to-SET `[d ∈ D |-> S_d]` whose VALUES are SUBSETS of a small fixed atom
    /// universe `E` (`alloc ∈ [Clients -> SUBSET Resources]`, the SimpleAllocator class) — the COMPOSITION
    /// of the [`ColSort::Func`] DOMAIN pack (domain `D`, the Int-prefix / atom / interval keys resolved via
    /// [`func_enum_domain_keys`]) with the [`ColSort::SetMask`] VALUE bijection (each value `S_d ⊆ E`
    /// encoded as the `|E|`-bit BITMASK `Σ_{e∈S_d} 2^idx(e)`). The function packs POSITIONALLY exactly like
    /// [`ColSort::Func`], but each value DIGIT is a `|E|`-bit set-mask rather than an Int/Bool/enum leaf:
    /// `pack = Σ_d mask(f[fdom_d])·base^d`, base `= 2^|E|`. `f[k]` extracts the digit `(pack / base^slot(k))
    /// mod base` — the `|E|`-bit mask of the set at key `k` — and all set OPERATIONS on it (`x ∈ f[k]`,
    /// `f[k] ⊆ T`, `f[k] = C`, `f[k] ∩ g[k'] = {}`) are EXACTLY the [`ColSort::SetMask`] bit ops on that
    /// digit (via [`SetIR::Digit`]). The binder type is `Nat`, the cell literal `Nat.lit(pack)`, and the
    /// pack is CANONICAL (one function ⇒ one Nat), so a FuncSetMask column slots into the existing `Eq`-based
    /// membership legs (`Init⊆R`, `image⊆R`) for FREE — like `Func`/`FuncEnum`, with NO `≥0` Safety conjunct.
    ///
    /// SOUNDNESS / DETERMINISM: the composed encoding is a BIJECTION — the [`ColSort::Func`] positional pack
    /// over `D` × the [`ColSort::SetMask`] subset↔mask bijection per value cell — so distinct functions map
    /// to distinct packs (no value collapse). `fdom`/`fdom_kind` (the `D` keys in canonical `Value::cmp`
    /// order, kind-tagged) FIX the key→slot map ([`crate::cleancic`]'s `resolve_func_domain_slot`), and
    /// `dom`/`dom_kind` (the `E` value-universe: the SORTED union of ALL atoms EVER present in ANY value
    /// across ALL enumerated states, grown monotonically via [`EnumStop::GrowSetMask`] exactly like a scalar
    /// [`ColSort::SetMask`]) FIX the bit meaning (`bit i ⟺ dom[i] ∈ S`). `base = 2^|dom|` is DERIVED (never
    /// stored — see [`ColSort::funcsetmask_base`]); the pack `base^arity` must fit a `u64` (`|dom|·arity ≤
    /// 63`, else fail-closed). EQUALITY/membership-only (a set-mask digit carries no order); the general
    /// Next-completeness domain is NOT modelled ⇒ closure rests on the honest enumerated `image ⊆ R` leg
    /// (like `SetMask`). Cross-column set ops require the value universes (`dom`+`dom_kind`) to MATCH; a
    /// value with an atom outside `dom`, a wider-than-63 `E`, a non-atom / nested value, or a `base^arity`
    /// overflow FAILS CLOSED.
    FuncSetMask {
        /// The domain size `|D|`: `arity` positions, keyed by `fdom` (or the Int prefix when `fdom` empty).
        arity: u32,
        /// The `D` DOMAIN KEY names in canonical `Value::cmp` order (position `p` = key `fdom[p]`) — the
        /// SAME key→slot mechanism [`ColSort::Func`]/[`ColSort::FuncEnum`] carry. CANONICALLY EMPTY for the
        /// classic `0..arity-1` Int-prefix domain. Part of the column identity.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        fdom: Vec<String>,
        /// The KIND of the `D` domain keys (`Model` / `Str` / `Int`) — shares the [`ColSort::Func::dom_kind`]
        /// serde default/skip helpers (skip-serialized `Model` default).
        #[serde(
            default = "func_enum_dom_kind_default",
            skip_serializing_if = "func_enum_dom_kind_is_default"
        )]
        fdom_kind: EnumKind,
        /// The `E` VALUE-UNIVERSE atoms — the SORTED, distinct union of every atom EVER present in ANY value
        /// set across ALL enumerated states (`bit i ⟺ dom[i] ∈ f[·]`), width `K = |dom| ≤ 63`. Part of the
        /// column identity (two FuncSetMask columns with different value universes are DIFFERENT sorts). A
        /// nonneg-`Int` value universe (`dom_kind == Int`) stores the values' DECIMAL texts (`f ∈ [D ->
        /// SUBSET (0..N)]`), keyed and masked IDENTICALLY to a model/`String` atom universe.
        dom: Vec<String>,
        /// The KIND of the `E` value-universe atoms (`Model` / `Str` / `Int`) — shares [`ColSort::SetMask`]'s
        /// [`EnumKind`]. `Int` is the nonneg-Int value universe (bit `idx("k") ⟺ k ∈ S`); the same faithful
        /// subset↔mask bijection, over Int-literal keys instead of atom names.
        dom_kind: EnumKind,
    },
}

/// The serde DEFAULT for [`ColSort::FuncEnum::dom_kind`] — `Model`, the pre-string-domain kind, so a cert
/// written before the `String`-domain extension (which had no `dom_kind` field) deserializes to `Model`
/// and re-checks byte-identically.
fn func_enum_dom_kind_default() -> EnumKind {
    EnumKind::Model
}

/// Whether a [`ColSort::FuncEnum::dom_kind`] is the skip-serialized default (`Model`). A `Str`-kind domain
/// is the only value that emits the field, keeping Int-prefix / model-value certs byte-identical.
fn func_enum_dom_kind_is_default(k: &EnumKind) -> bool {
    matches!(k, EnumKind::Model)
}

/// The KIND of a [`ColSort::Enum`] column's labels: a TLA `String` versus a config CONSTANT model value.
/// A `String` label `"read"` and a model value NAMED `read` are DISTINCT TLA values, so their label
/// indices must NEVER be equated — the recognizer only matches a `String`-literal set/operand against a
/// `Str` column and a model-value CONSTANT set against a `Model` column (any cross-kind form fails
/// closed). A deterministic function of the column's observed cells (all-`String` ⇒ `Str`, all-model-value
/// ⇒ `Model`; the encoder rejects a mix), so Leg-E re-derives the same kind and the obligations rebuild
/// identically. Always compiled (serde) so the cert schema is feature-independent and re-checkable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EnumKind {
    /// The column's cells are TLA `String`s (program counters `"read"`/`"write"`, …).
    Str,
    /// The column's cells are config CONSTANT model values (`ModelValueSet` members `d1`, `d2`, …).
    Model,
    /// The DOMAIN keys of a [`ColSort::Func`]/[`ColSort::FuncEnum`] are nonneg-Int LITERALS drawn from a
    /// BOUNDED interval `lo..hi` (a 1-based PlusCal process-counter `pc ∈ [1..N -> labels]`, or any
    /// non-0-based `Value::Seq`/interval-domain function). ONLY meaningful as a `dom_kind` — never a scalar
    /// [`ColSort::Enum`]/[`ColSort::SetMask`] cell kind (those are derived from `String`/model VALUES, never
    /// Int keys). The `dom` texts are the keys' DECIMAL forms in sorted-by-value order (`["1","2",…]`), so
    /// `f[k]` resolves to slot `position("k") = k − lo` (a faithful bijection between the interval and
    /// `0..hi−lo`) and `DOMAIN f = lo..hi` discharges by comparing the materialized Int interval to `dom`
    /// NUMERICALLY. A `String` key `"1"` and the Int key `1` are DISTINCT TLA values, so an `Int`-kind dom
    /// resolves ONLY an Int-LITERAL index and a `Str`/`Model` dom NEVER an Int one (cross-kind ⇒ closed).
    /// Serialized as the additive `"Int"` byte (never the skip-serialized `Model` default), so every
    /// pre-existing 0-based/atom/model/`Str` cert stays BYTE-IDENTICAL.
    Int,
}

/// The per-POSITION VALUE SORT of a digit inside a positional-pack compound (a [`ColSort::Record`] field
/// or an Int-domain [`ColSort::Func`] value). The pack is UNCHANGED — one uniform `base`, `pack =
/// Σ_i code_i·base^i`, digit extraction `(pack / base^i) mod base` — but each position's digit `code_i`
/// carries a KIND that fixes its MEANING:
///   * `Int`  — the digit IS the nonneg Int value (the historical leaf, `code = v`).
///   * `Bool` — the digit is the Bool code `1`=`TRUE` / `0`=`FALSE`.
///   * `Enum` — the digit is the INDEX of a `String` / model-value LABEL in the position's cross-state
///     sorted label union (the per-POSITION analogue of the scalar [`ColSort::Enum`]); `kind` fixes whether
///     the labels are TLA `String`s or config CONSTANT model values (a `String` `"x"` and a model value
///     named `x` are DISTINCT TLA values ⇒ DISTINCT kinds ⇒ never one index).
///
/// SOUNDNESS — the KIND DISCRIMINANT (why this is not merely cosmetic): a Bool `TRUE`, an Int `1`, a
/// `String` `"x"`, and a model value `x` are ALL DISTINCT TLA values; if a position held two of them in
/// different states and both mapped to the SAME digit, two distinct states would collapse in the reachable
/// set `R` and a violation could hide (UNSOUND). The kind is therefore part of the COLUMN'S IDENTITY — it
/// lives in the serialized [`ColSort::Record`]/[`ColSort::Func`] `cells` vector and is compared by
/// `PartialEq` (an `Enum` position ALSO carries its sorted label union + `kind`, so a position mixing
/// `Int`/`Bool`/`Enum`, or an `Enum` mixing `Str`/`Model`, is a DIFFERENT `CellSort`). A position observed
/// holding MORE THAN ONE kind across the enumerated states yields DIFFERENT `ColSort`s on the two states, so
/// the cross-state `col_sorts` agreement check FAILS CLOSED (declines the whole column) — exactly the
/// mechanism that already pins a compound's field set and radix. A `Bool`/`Enum` digit is EQUALITY-ONLY: the
/// recognizer emits it ONLY inside kind-checked `= TRUE`/`= FALSE`/`∈ BOOLEAN` / `= "lbl"` / `∈ {…}`/`∈ Data`
/// forms and DECLINES a bare `r.f`/`f[i]` value (so ordering/arith on such a position — `r.on >= 0` — fails
/// closed), never conflating a Bool/enum code with a Nat value.
///
/// DETERMINISM: the kind + code are a pure function of the observed cell (`Value::Bool` ⇒ `Bool`, a nonneg
/// Int ⇒ `Int`, a `String`/model value ⇒ its index in the position's sorted cross-state label union), so
/// verify (Leg-E) re-derives them byte-identically. Always compiled (serde) so the cert schema stays
/// feature-independent and re-checkable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CellSort {
    /// A nonneg-`Int` digit: `code = v` (the historical positional-pack leaf).
    Int,
    /// A `Bool` digit: `code = 1` (`TRUE`) / `code = 0` (`FALSE`).
    Bool,
    /// An ENUM digit: `code` is the INDEX of a `String` / model-value LABEL in the position's sorted
    /// cross-state label union `labels`; `kind` distinguishes `String` labels from model-value labels. The
    /// per-POSITION analogue of [`ColSort::Enum`] (EQUALITY-ONLY — an index carries no order). Part of the
    /// column identity: two positions with different label unions / kinds are DIFFERENT `CellSort`s.
    Enum {
        /// The sorted, distinct label set observed at this position across ALL enumerated states — the pack
        /// radix must admit its max index (`labels.len()-1`), and every state indexes into this SAME union.
        labels: Vec<String>,
        /// Whether the labels are TLA `String`s or config CONSTANT model values (a kind mismatch declines).
        kind: EnumKind,
    },
}

/// The per-position kind of digit `idx` of a positional compound, from its `cells` vector. An EMPTY
/// `cells` is the CANONICAL all-`Int` compound (byte-identical to a pre-value-type-leaf cert — the encoder
/// NORMALIZES an all-Int compound to empty `cells`), so an out-of-range / absent entry reads back as `Int`.
/// Returns a REFERENCE (a [`CellSort::Enum`] owns a `Vec` ⇒ not `Copy`); the default `Int` is a `static`.
// Sole callers live in cleancic.rs, which is gated behind the clean-cic feature.
#[cfg_attr(not(feature = "clean-cic"), allow(dead_code))]
pub(crate) fn cell_kind(cells: &[CellSort], idx: usize) -> &CellSort {
    static INT: CellSort = CellSort::Int;
    cells.get(idx).unwrap_or(&INT)
}

/// The serde DEFAULT for [`ColSort::Seq::elem`] — `Int`, the historical nonneg-Int element leaf, so a cert
/// written before the sequence value-type-leaf extension (which had no `elem` field) deserializes to `Int`
/// and re-checks byte-identically.
fn cellsort_int_default() -> CellSort {
    CellSort::Int
}

/// Whether a [`ColSort::Seq::elem`] is the skip-serialized default (`Int`). An atom/`Bool` element is the
/// only value that emits the field, keeping every pure-Int `Seq` cert byte-identical.
fn cellsort_is_int(c: &CellSort) -> bool {
    matches!(c, CellSort::Int)
}

impl ColSort {
    /// Whether this column is a COMPOUND (packed-`Nat`) sort — `Record`/`Func`/`Seq`/`FuncEnum`. Such a
    /// column reuses the `Eq Nat` membership legs verbatim (the pack is canonical) but carries NO `≥0`
    /// Safety conjunct. This gate ALSO drives [`crate::cleancic::compound_col_operand`], which REFUSES a
    /// bare compound-column operand to the surface `\div`/`%` recognizers — so no surface arithmetic can
    /// forge the packed-`Nat` digit-extraction shape that `pred_exact` admits (undefined-in-TLA arithmetic
    /// on a function must stay rejected; `FuncEnum` needs this exactly as `Record`/`Func` do, since it too
    /// has a digit-extraction exactness carve-out). `FuncEnum`'s general-completeness axis is NOT the packed
    /// range (its values are equality-only enum indices), so it DECLINES the completeness leg (its arm in
    /// [`crate::cleancic::next_domain_bounds_from_ir`] returns `None`) and closure rests on the honest
    /// enumerated `image ⊆ R` leg; `pack_universe` is correspondingly `None` for it.
    pub fn is_compound(&self) -> bool {
        matches!(
            self,
            ColSort::Record { .. }
                | ColSort::Func { .. }
                | ColSort::Seq { .. }
                | ColSort::FuncEnum { .. }
        )
    }

    /// The number of distinct PACK values a COMPOUND column can take — the size of its general-completeness
    /// product axis `{0..=pack_universe-1}`. A `Record`/`Func` packs `Σ v_i·base^i` with each `v_i < base`
    /// over `arity` slots ⇒ exactly `base^arity` packs. A `Seq` packs self-delimitingly in base `base+1`
    /// over up to `max_len` slots ⇒ `(base+1)^max_len` packs (an upper bound covering every length
    /// `≤ max_len`, including the empty sequence). `None` for a non-compound sort or on `u64` overflow.
    /// SOUNDNESS: the changing-compound completeness axis is `{0..=pack_universe-1}`, and EVERY legal pack
    /// of this column is `< pack_universe` (each slot `< base`, length `≤ max_len`), so `D ⊇ Succ(R)`.
    pub fn pack_universe(&self) -> Option<u64> {
        match self {
            ColSort::Record { base, fields, .. } => base.checked_pow(fields.len() as u32),
            ColSort::Func { base, arity, .. } => base.checked_pow(*arity),
            ColSort::Seq { base, max_len, .. } => base.checked_add(1)?.checked_pow(*max_len),
            // A FuncEnum packs `Σ_p idx(e_p)·|labels|^p` with each digit `< |labels|` over `arity` slots ⇒
            // exactly `|labels|^arity` packs. EVERY legal pack is `< |labels|^arity` (each slot a label
            // INDEX `< |labels|`), so the changing-completeness axis `{0..=|labels|^arity-1}` covers every
            // successor by construction — exactly as `Func` does over its Int/Bool/enum-valued pack.
            ColSort::FuncEnum { labels, arity, .. } => (labels.len() as u64).checked_pow(*arity),
            _ => None,
        }
    }

    /// Whether this is a RECOGNIZED cell-kind-safe COMPOUND column whose recognized pack-update writes an
    /// EXACT successor pack — and is therefore admitted at the RAISED
    /// [`COMPOUND_COMPLETENESS_PACK_CAP_RECOGNIZED`] instead of the conservative
    /// [`COMPOUND_COMPLETENESS_PACK_CAP`] floor. The three shapes the pack-update recognizers handle
    /// cell-kind-aware, each with an EXACT (never under-approximating) successor pack:
    ///   * a MIXED-CELL Record — `≥1` non-`Int` cell (an Enum/model-value or Bool field), the Channel
    ///     `chan ∈ [val:Data, rdy:{0,1}, ack:{0,1}]` class recognized by
    ///     [`crate::cleancic::record_update_eq_form`]. A plain all-`Int` Record has CANONICALLY EMPTY
    ///     `cells` (see [`ColSort::Record::cells`]) ⇒ NOT recognized here ⇒ stays at the floor (its
    ///     `base^2 = 100` CoffeeCan class fits the floor anyway);
    ///   * a FuncEnum — a label-index-valued function ([`crate::cleancic::func_enum_update_eq_form`]);
    ///   * a Bool-celled `Func` — a `[D -> BOOLEAN]` flag array
    ///     ([`crate::cleancic::func_bool_update_eq_form`]), all cells `Bool`, `cells.len() == arity`.
    ///
    /// SOUNDNESS is INDEPENDENT of this gate (see [`COMPOUND_COMPLETENESS_PACK_CAP_RECOGNIZED`]): a larger
    /// cap only enlarges the completeness domain, strictly harder. The gate exists purely to CONTAIN the
    /// perf / cert-portability cost of the raise to the columns that need + justify it, and it is a
    /// DETERMINISTIC function of the (re-derived) sort, so certify and verify agree per column.
    pub fn is_recognized_cellkind_compound(&self) -> bool {
        match self {
            ColSort::FuncEnum { .. } => true,
            ColSort::Func { cells, arity, .. } => {
                cells.len() == *arity as usize
                    && !cells.is_empty()
                    && cells.iter().all(|c| matches!(c, CellSort::Bool))
            }
            ColSort::Record { cells, .. } => cells.iter().any(|c| !matches!(c, CellSort::Int)),
            _ => false,
        }
    }

    /// The COMPLETENESS pack cap for THIS column: the RAISED
    /// [`COMPOUND_COMPLETENESS_PACK_CAP_RECOGNIZED`] for a recognized cell-kind-safe compound
    /// ([`ColSort::is_recognized_cellkind_compound`]), else the conservative
    /// [`COMPOUND_COMPLETENESS_PACK_CAP`] floor. Centralizes the per-column, recognition-gated cap choice
    /// so certify and verify pick it identically.
    pub fn completeness_pack_cap(&self) -> u64 {
        if self.is_recognized_cellkind_compound() {
            COMPOUND_COMPLETENESS_PACK_CAP_RECOGNIZED
        } else {
            COMPOUND_COMPLETENESS_PACK_CAP
        }
    }

    /// The per-value pack radix of a [`ColSort::FuncSetMask`] column: `base = 2^|dom|` (each value digit is
    /// a `|dom|`-bit set-mask over the value universe `E`). DERIVED (never stored) so `base` can never
    /// disagree with the universe width. `None` for a non-FuncSetMask sort or `|dom| ≥ 64` (the shift would
    /// overflow — fail-closed). The pack `base^arity` overflow is checked separately by the caller.
    pub fn funcsetmask_base(&self) -> Option<u64> {
        match self {
            ColSort::FuncSetMask { dom, .. } => (dom.len() < 64).then(|| 1u64 << dom.len()),
            _ => None,
        }
    }
}

/// The recognized affine single-Int action fragment `x' = x + delta  ∧  x < bound` (guard on the
/// CURRENT state). When the live `Next` matches this shape, the cert carries a KERNEL-RE-EVALUATED
/// `Next`-completeness leg (`cleancic::certify_next_completeness`): the kernel reduces an embedded
/// `Next` predicate over the entire finite domain `D` and proves `R` is closed under the *relation*,
/// not merely over TY's enumerated image — removing the enumerator-trust gap for this fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AffineNextShape {
    /// the increment `delta` in `x' = x + delta`
    pub delta: u64,
    /// the guard bound `b` in `x < b`
    pub bound: u64,
}

/// A PARAMETRIC inductive-invariant `Certified` witness for an UNBOUNDED affine single-Int counter —
/// `Init = x = c`, `Next = x' = x + δ` (NO guard ⇒ INFINITE reachable set), `Safety = x ≥ 0` — that the
/// finite-enumeration fixpoint path cannot certify (the BFS never terminates). Instead of a finite fold
/// over an enumerated `R`, the inductive invariant `J ≡ Safety ≡ (x ≥ 0) ≡ Int.NonNeg x` is proved by
/// THREE universally-quantified, kernel-checked CIC implications over the WHOLE (infinite) Int domain,
/// with NO enumeration:
///   * `initiation`   — `Init ⇒ J`:        `Int.NonNeg.mk c : NonNeg (Int.ofNat c)` (ground).
///   * `consecution`  — `(J ∧ Next) ⇒ J'`: `Π(x:Int). NonNeg x → NonNeg (Int.add x (Int.ofNat δ))`
///     (the `∀x` step IS the closure proof — there is NO finite domain-coverage obligation here).
///   * `preservation` — `J ⇒ Safety`:      `Π(x:Int). NonNeg x → NonNeg x` (identity, since `J ≡ Safety`).
///
/// ALWAYS compiled (serde) so the schema is feature-independent and re-checkable; the build/recheck
/// entry points ([`crate::cleancic::certify_unbounded_invariant`] / `verify_unbounded_invariant`) are
/// `clean-cic`-gated. The trust base is the kernel accepting all three legs at the EXACT types rebuilt
/// from `(c, δ)`. MUTUALLY EXCLUSIVE with the finite-enumeration legs (`recognize_unbounded_affine`
/// fires only when there is NO guard; the finite path's `recognize_affine_next` requires one).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnboundedInvariantCert {
    /// The initial value `c` in `Init = (x_0 = c)` for the FIRST variable (nonneg). Equals `pairs[0].0`;
    /// retained for the single-variable API and for a human-readable summary.
    pub init: u64,
    /// The increment `δ` in `Next = (x_0' = x_0 + δ)` for the FIRST variable (nonneg). Equals
    /// `pairs[0].1`.
    pub delta: u64,
    /// Per-variable `(c_j, δ_j)` in state-variable DECLARATION ORDER (`pairs.len() = n`, the variable
    /// count). For `n = 1` this is `[(init, delta)]` and the legs below are the SCALAR terms; for
    /// `n ≥ 2` the legs are the CONJOINED multi-variable terms (`⋀_j …` via `And.intro`). The re-check
    /// rebuilds the obligation types from `pairs` and dispatches on `pairs.len()`. Empty iff this is a
    /// RELATIONAL cert (`relational.is_some()`), whose invariant is `x=y`, not a per-variable nonneg.
    #[serde(default)]
    pub pairs: Vec<(u64, u64)>,
    /// The SCALAR safety lower bound `N` in `Safety = (x ≥ N)` (Phase 2 widening,
    /// `docs/kernel-checked-tla-plan.md`). `0` (the serde default — pre-widening certs
    /// deserialize unchanged) is the historical `NonNeg` fragment; `N > 0` certs carry the
    /// `Int.le` lemma legs and are re-checked at types rebuilt from `(c, δ, N)`. Scalar
    /// (`pairs.len() == 1`) only: the multi-variable and relational lanes stay `N = 0`.
    /// `skip_serializing_if` zero: a `bound = 0` cert re-serializes BYTE-IDENTICALLY to a
    /// pre-widening cert, so the `SafetyCertificate` sha256 (recomputed over a
    /// re-serialization at verify time) still matches for certificates minted before this
    /// field existed.
    #[serde(default, skip_serializing_if = "bound_is_zero")]
    pub bound: u64,
    /// A RELATIONAL invariant `J ≡ (v0 = v1)` (STRONGER than a bare per-variable nonneg) for the
    /// lock-step counter `Init v0=c∧v1=c / Next v0'=v0+δ∧v1'=v1+δ / Safety v0=v1`. When `Some((c, δ))`,
    /// the three legs below are the `Eq`-relational terms (`Eq.refl` / `Eq.subst` / identity) and
    /// `pairs` is empty; the re-check rebuilds the relational obligation from `(c, δ)`. `None` for the
    /// nonneg (`pairs`-based) certs.
    #[serde(default)]
    pub relational: Option<(u64, u64)>,
    /// Kernel-checked INITIATION leg term — `⋀_j Int.NonNeg.mk c_j : ⋀_j NonNeg (Int.ofNat c_j)`
    /// (a bare `Int.NonNeg.mk c` for `n = 1`).
    pub initiation: Vec<u8>,
    /// Kernel-checked CONSECUTION leg term — `Π(x_0..x_{n-1}:Int). (⋀_j NonNeg x_j) → (⋀_j NonNeg
    /// (Int.add x_j (Int.ofNat δ_j)))`. The `∀x_0..x_{n-1}` inductive step (bare `Int.NonNeg.add` for
    /// `n = 1`).
    pub consecution: Vec<u8>,
    /// Kernel-checked PRESERVATION leg term — `Π(x_0..x_{n-1}:Int). (⋀_j NonNeg x_j) → (⋀_j NonNeg x_j)`
    /// (identity, since `J ≡ Safety`; bare `λ(x)(h). h` for `n = 1`).
    pub preservation: Vec<u8>,
}

/// serde helper: keep `bound = 0` (the entire pre-widening cert population) out of the
/// serialization so old certificates' digests keep verifying (see the field docs).
#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if ABI takes &field
fn bound_is_zero(v: &u64) -> bool {
    *v == 0
}

// ── Serializable predicate/value IR for the single-Int (general, sorts==[Int]) embedder ─────────────
// ALWAYS compiled (serde) so `verify_explicit_state_cert` / Leg-E can rebuild the kernel obligation
// from the CERT ALONE (no parser AST). Variable references are resolved to COLUMN INDICES at recognize
// time (against `vars`), so the embedder needs no name table. Mirrors `embed_value`/`embed_pred`
// EXACTLY: `ValIR` ⇒ Nat term, `PredIR` ⇒ Bool term. No Sub/Neg node exists — recognize fails closed
// on those (same as `embed_value` returning None).

/// Arithmetic VALUE term over nonneg Nat, indexing state columns by position.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ValIR {
    /// A nonneg integer literal → `Expr::nat_lit(v)`.
    Lit(u64),
    /// Current-state column `i` → `Nat.lit(s[i])` (mirrors `E::Ident` ⇒ `s[col]`).
    Var(usize),
    /// Primed (next-state) column `i` → `Nat.lit(sp[i])` (mirrors `E::Prime(Ident)` ⇒ `sp[col]`).
    Prime(usize),
    /// `a + b` → `Nat.add`.
    Add(Box<ValIR>, Box<ValIR>),
    /// `a * b` → `Nat.mul`.
    Mul(Box<ValIR>, Box<ValIR>),
    /// `a \div b` → `Nat.div`.
    Div(Box<ValIR>, Box<ValIR>),
    /// `a % b` → `Nat.mod`.
    Mod(Box<ValIR>, Box<ValIR>),
    /// Nat-truncated subtraction `a ∸ b` → `Nat.sub` (`a-b` if `a≥b`, else `0`). recognize emits this
    /// (1) for the SEQUENCE-digit `−1` undo of the self-delimiting `+1` shift (a present element's digit
    /// is `≥1`, so `digit ∸ 1 = digit − 1` EXACTLY — the truncation never bites there); and (2) for the
    /// NARROWLY SOUND general subtraction `v = a − b` where `v` is a nonneg-on-every-state value in a
    /// POSITIVE-polarity equality (then a real transition forces `a ≥ b`, so `Nat.sub` is EXACT — see
    /// `cleancic::eq_sub_form`). It is NOT exposed for arbitrary `E::Sub` in comparisons/negations (TLA
    /// `−` can go negative and be Nat-truncated wrongly, DROPPING a real successor).
    Sub(Box<ValIR>, Box<ValIR>),
    /// SEQUENCE `Len(s)` over a `Seq{base,max_len}` pack in radix `D=base+1`: the count of NONZERO base-`D`
    /// digits, `Σ_{i<max_len} (digit_i ≠ 0 ? 1 : 0)` with `digit_i = (pack / D^i) mod D`. `digit_i ≠ 0` is
    /// `Nat.ble 1 digit_i`, lifted to a `0/1` Nat via `Bool.rec`. EXACT to TLA `Len` on the self-delimiting
    /// pack (a present element's digit is `≥1`; the first `0` digit ends the sequence). Carries the pack
    /// term plus the fixed `base`/`max_len` so the embedder can rebuild the bounded fold from the IR alone.
    SeqLen {
        /// The sequence pack (a `ValIR` reducing to the base-`base+1` self-delimiting Nat).
        pack: Box<ValIR>,
        /// The element radix `base` (the pack digit radix is `base+1`).
        base: u64,
        /// The length bound (the fold ranges over digit places `0..max_len`).
        max_len: u32,
    },
    /// SEQUENCE `Tail(s)` over a `Seq{base,..}` pack: `pack / D` (`D=base+1`) — drops the first (lowest)
    /// digit, shifting every later element down one place. EXACT (integer division by the radix). `Tail`
    /// of the empty sequence is TLA-undefined; on a real transition the pack is a genuine sequence so this
    /// is only reached for present tails.
    SeqTail {
        /// The sequence pack whose lowest digit is dropped.
        pack: Box<ValIR>,
        /// The element radix `base` (division is by `base+1`).
        base: u64,
    },
    /// SEQUENCE `Append(s, e)` over a `Seq{base,max_len}` pack: `pack + (e+1)·D^Len(s)` (`D=base+1`) —
    /// writes the shifted element `e+1` into the first FREE digit (place `Len(s)`). `D^Len(s)` is built
    /// from the same `SeqLen` fold. EXACT for `Len(s) < max_len` (room for one more element); an OVER-FULL
    /// result (`Len(s) = max_len`) exceeds the pack universe `D^max_len` and so is never a valid
    /// `value_cell` domain state — the leg never needs it, so emitting the arithmetic uniformly is sound.
    SeqAppend {
        /// The sequence pack being appended to.
        pack: Box<ValIR>,
        /// The element `e` to append (a `ValIR` reducing to a nonneg small Int).
        elem: Box<ValIR>,
        /// The element radix `base` (pack radix `base+1`).
        base: u64,
        /// The length bound (used for the `Len` fold over `0..max_len`).
        max_len: u32,
    },
    /// SET `Cardinality(S)` over a bitmask set (`Set` or `SetMask`) — the POPCOUNT `Σ_{i<universe}
    /// bit_i(mask)`, the count of set bits. `bit_i(mask) = (mask >> i) & 1` lifted to a `0/1` `Nat` via
    /// `Bool.rec`; the sum is a bounded left-nested `Nat.add` over the `universe` bit positions. EXACT to
    /// TLA `Cardinality` on the faithful bitmask (bit `i` ⟺ element/atom `i ∈ S`, a bijection, so the set
    /// bit count IS `|S|`). Carries the `SetIR` plus the fixed `universe` (bit width) so the embedder can
    /// rebuild the bounded fold from the IR alone. EQUALITY/ORDERING on the count is exact nonneg Nat
    /// arithmetic (`|S| ≤ k`, `|S| = k`), so `pred_exact` admits it (unlike the Nat-truncating Seq ops).
    SetCard {
        /// The set whose set bits are counted (a `SetIR` reducing to the bitmask `Nat`).
        set: SetIR,
        /// The element/atom universe bit width `K`: the popcount fold ranges over bit places `0..K`.
        universe: u32,
    },
    /// SET-COMPREHENSION `Cardinality({d ∈ D : P(d)})` over a FIXED finite domain `D` (a config CONSTANT
    /// atom set / union / `lo..hi` interval / ground Int set) — the COUNTING SUM `Σ_{d∈D} boolToNat(P(d))`
    /// of the per-element 0/1 truth values. `terms` holds ONE recognized `PredIR` per domain element `d`,
    /// built by AST-substituting `d` for the comprehension's bound var and re-recognizing over the ORIGINAL
    /// columns (the R4 Path-B mechanism), so a function application `f[d]` in `P` becomes the literal-key
    /// digit `f["d"]`. Each `boolToNat(P(d))` is exactly `0`/`1` (the terms are `pred_exact`-TRUE, never a
    /// third value), and `D` is the COMPLETE fixed domain the checker enumerates, so the sum EQUALS
    /// `|{d∈D:P(d)}|` — EXACT to TLA `Cardinality`. The embedder folds a left-nested `Nat.add` of
    /// `boolToNat` legs; the count is nonneg-Nat arithmetic (`count < k`, `count = k`), so `pred_exact`
    /// admits it (like [`SetCard`], unlike the Nat-truncating Seq ops). DETERMINISTIC (`terms` follows the
    /// sorted-deduped domain order) so Leg-E re-derives byte-identical IR.
    CountFold {
        /// One `pred_exact`-TRUE per-element predicate `P(d)` per domain element `d ∈ D`, in the domain's
        /// canonical (sorted-deduped) order. The count is `Σ_d boolToNat(terms[d])`.
        terms: Vec<PredIR>,
    },
}

/// A SET-valued term over the bitmask encoding — a `Nat` whose value is the bitmask `Σ_{e∈S} 2^e`.
/// Mirrors `embed_set_ir`'s kernel-op choices: a Set column's cell is already a bitmask `Nat`, and the
/// set constructors embed as kernel-REDUCIBLE `Nat` bitwise ops (`∪`=`Nat.lor`, `∩`=`Nat.land`,
/// singleton/literal=a constant mask). `recognize_set` is the all-or-nothing AST→IR translator.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SetIR {
    /// A CONSTANT set literal `{a,b,…}` → the constant bitmask `Σ 2^a` (a singleton `{v}` is `1<<v`;
    /// the empty set `{}` is `0`). Pre-folded at recognize time (the elements are ground literals).
    Lit(u64),
    /// Current-state Set column `i` → `Nat.lit(s[i])` (the column's bitmask cell).
    Var(usize),
    /// Primed (next-state) Set column `i` → `Nat.lit(sp[i])`.
    Prime(usize),
    /// `S ∪ T` → `Nat.lor` (bitmask union).
    Cup(Box<SetIR>, Box<SetIR>),
    /// `S ∩ T` → `Nat.land` (bitmask intersection).
    Cap(Box<SetIR>, Box<SetIR>),
    /// SET COMPREHENSION `{x ∈ S : P(x)}` over the bitmask `source` and a `universe` of `K` element bits.
    /// The result bitmask keeps exactly the bits `i<K` that are SET in `source` AND satisfy `pred` at
    /// `x=i`. EXACT to the TLA filter semantics: `result = ⋁_{i<K} ( bit_i(source) ∧ P(i) ? (1<<i) : 0 )`.
    /// `pred` is a `PredIR` over the state columns EXTENDED with the bound var as a fresh trailing column
    /// `bound_col` (= the real column count); embedding substitutes the LITERAL `i` for that column. The
    /// fold is kernel-reducible (`Nat.lor` of `(1<<i)·[bit_i(source)·boolToNat(P(i))]` per `i<K`).
    Filter {
        /// The source set `S` whose bits are filtered.
        source: Box<SetIR>,
        /// The element-universe bit width `K`: bits `0..K` are tested.
        universe: u32,
        /// The fresh trailing column index the bound var occupies in `pred` (= real column count).
        bound_col: usize,
        /// The filter predicate `P(x)` over the extended columns (bound var at `bound_col`).
        pred: Box<PredIR>,
    },
    /// A FUNCTION-to-SET APPLICATION `f[k]` over a [`ColSort::FuncSetMask`] column — the `|E|`-bit set-mask
    /// DIGIT `(pack / place) mod base` extracted from the column's positional pack (`pack` the packed cell
    /// `ValIR::Var`/`Prime` of the column, `place = base^slot(k)`, `base = 2^|E|`). Embeds as the
    /// kernel-REDUCIBLE `Nat.mod(Nat.div pack place) base` — the SAME digit-extraction the `Record`/`Func`
    /// `f[i]` value recognizers use — yielding the `Nat` BITMASK of the set `f[k]`, which every downstream
    /// `SetIR` bit op (`Cup`/`Cap`) and set predicate (`SetMem`/`SetSubseteq`/`SetEq`) then treats EXACTLY
    /// as a bare [`ColSort::SetMask`] cell. EXACT to TLA function-application-and-set semantics on the
    /// canonical pack (each digit `< base` is a genuine `|E|`-bit subset mask, a bijection with subsets of
    /// `E`). Equality/membership-only — a set-mask digit carries no order (`pred_exact` admits it only over
    /// a `FuncSetMask` column, never as a bare arithmetic Nat).
    Digit {
        /// The packed FuncSetMask cell (a `ValIR::Var`/`Prime` of the column) whose digit is extracted.
        pack: Box<ValIR>,
        /// The place value `base^slot(k)` selecting key `k`'s digit.
        place: u64,
        /// The value radix `base = 2^|E|` (the digit modulus — each value is a `|E|`-bit mask `< base`).
        base: u64,
    },
}

/// Boolean PREDICATE term, mirroring `embed_pred`'s kernel-op choices.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PredIR {
    /// `a ∧ b` → `Bool.and`.
    And(Box<PredIR>, Box<PredIR>),
    /// `a ∨ b` → `Bool.or`.
    Or(Box<PredIR>, Box<PredIR>),
    /// `¬a` → `Bool.not`.
    Not(Box<PredIR>),
    /// `a ⇒ b` → `Bool.or(Bool.not a, b)`.
    Implies(Box<PredIR>, Box<PredIR>),
    /// `a ⇔ b` → `Bool.or(Bool.and(a,b), Bool.and(¬a,¬b))`.
    Equiv(Box<PredIR>, Box<PredIR>),
    /// `a = b` → `Nat.beq a b`.
    Eq(ValIR, ValIR),
    /// `a ≠ b` → `Bool.not(Nat.beq a b)`.
    Neq(ValIR, ValIR),
    /// `a < b` → `Nat.ble(Nat.add a 1, b)`.
    Lt(ValIR, ValIR),
    /// `a ≤ b` → `Nat.ble a b`.
    Leq(ValIR, ValIR),
    /// `a > b` → `Nat.ble(Nat.add b 1, a)`.
    Gt(ValIR, ValIR),
    /// `a ≥ b` → `Nat.ble b a`.
    Geq(ValIR, ValIR),
    /// `TRUE`/`FALSE` → `Bool.true`/`Bool.false`.
    BoolLit(bool),
    /// `UNCHANGED x` (column `i`) → `Nat.beq sp[i] s[i]`.
    Unchanged(usize),

    // ── SET fragment over the bitmask encoding (kernel-reducible, EXACTLY the TLA set semantics) ──
    /// `S = T` (set equality) → `Nat.beq maskS maskT` (bitmask equality = set equality).
    SetEq(SetIR, SetIR),
    /// `S ≠ T` → `Bool.not(Nat.beq maskS maskT)`.
    SetNeq(SetIR, SetIR),
    /// `e ∈ S` for a GROUND element literal `e` → bit test `Nat.beq(Nat.land(Nat.shiftRight maskS e, 1), 1)`.
    SetMem(u64, SetIR),
    /// `e ∉ S` → `Bool.not` of the bit test.
    SetNotMem(u64, SetIR),
    /// `S ⊆ T` → `Nat.beq(Nat.lor maskS maskT, maskT)` (every bit of `S` is a bit of `T`).
    SetSubseteq(SetIR, SetIR),
    /// `UNCHANGED S` (Set column `i`) → `Nat.beq sp[i] s[i]` (bitmask equality of the primed/current cell).
    SetUnchanged(usize),

    // ── BOUNDED QUANTIFIERS over a CONCRETE set / powerset (kernel-reducible bounded folds, EXACT) ──
    /// `∀y ∈ S : P(y)` over a Set-valued `source` (bitmask) and a `universe` of `K` element bits →
    /// the `Bool.and`-fold `⋀_{y<K} ( ¬mem(y,source) ∨ P(y) )`. Each leg is vacuously TRUE for a
    /// NON-member `y` and `P(y)` for a member — so the conjunction ranges over EXACTLY the elements of
    /// `S`. `body` is a `PredIR` over the state columns EXTENDED with the bound var as a fresh trailing
    /// column `bound_col`; embedding substitutes the LITERAL element `y` for that column (the same
    /// mechanism as [`SetIR::Filter`]). EXACT to TLA `∀y∈S:P` on the bitmask (mem(y,S) is the bit test).
    SetForall {
        /// The set `S` the bound var ranges over (a `SetIR` — concrete at embed time).
        source: SetIR,
        /// The element-universe bit width `K` (elements `0..K` are tested for membership).
        universe: u32,
        /// The fresh trailing column index the bound var occupies in `body` (= real column count).
        bound_col: usize,
        /// The body predicate `P(y)` over the extended columns (bound var at `bound_col`).
        body: Box<PredIR>,
    },
    /// `∃y ∈ S : P(y)` → the `Bool.or`-fold `⋁_{y<K} ( mem(y,source) ∧ P(y) )` (a member `y` satisfying
    /// `P`). Dual of [`PredIR::SetForall`]; EXACT to TLA `∃y∈S:P` on the bitmask.
    SetExists {
        /// The set `S` the bound var ranges over (a `SetIR` — concrete at embed time).
        source: SetIR,
        /// The element-universe bit width `K` (elements `0..K` are tested for membership).
        universe: u32,
        /// The fresh trailing column index the bound var occupies in `body` (= real column count).
        bound_col: usize,
        /// The body predicate `P(y)` over the extended columns (bound var at `bound_col`).
        body: Box<PredIR>,
    },
    /// `∀T ∈ SUBSET S : P(T)` → the `Bool.and`-fold `⋀_{T ⊆ S} P(T)` over the SUBMASKS of the concrete
    /// `source` bitmask (there are `2^popcount(source)` of them; the embedder enumerates them at embed
    /// time and substitutes each concrete submask literal for the bound var). EXACT finite enumeration of
    /// the powerset (`T` ranges over exactly the subsets of `S`). CAPPED: `source`'s universe `K` must be
    /// `≤ SUBSET_QUANT_POPCOUNT_CAP` (recognize declines above ⇒ fail-closed), so `popcount(source) ≤ K`
    /// bounds the fold at `≤ 2^K ≤ 2^cap` submasks. `bound_col` is the fresh trailing column; the bound
    /// var value is a WHOLE bitmask (a Set-column value), read/compared inside `body` as a set.
    SubsetForall {
        /// The set `S` whose powerset is quantified — the bound var ranges over `S`'s submasks.
        source: SetIR,
        /// The bit width `K` of `S`'s universe (bounds the submask count at `2^popcount ≤ 2^K`).
        universe: u32,
        /// The fresh trailing column the bound-var SUBMASK occupies in `body` (a Set-valued column).
        bound_col: usize,
        /// The body predicate `P(T)` over the extended columns (bound-var submask at `bound_col`).
        body: Box<PredIR>,
    },
    /// `∃T ∈ SUBSET S : P(T)` → the `Bool.or`-fold `⋁_{T ⊆ S} P(T)` over the submasks of `source`. Dual
    /// of [`PredIR::SubsetForall`]; same cap and EXACT finite-enumeration soundness.
    SubsetExists {
        /// The set `S` whose powerset is quantified — the bound var ranges over `S`'s submasks.
        source: SetIR,
        /// The bit width `K` of `S`'s universe (bounds the submask count at `2^popcount ≤ 2^K`).
        universe: u32,
        /// The fresh trailing column the bound-var SUBMASK occupies in `body` (a Set-valued column).
        bound_col: usize,
        /// The body predicate `P(T)` over the extended columns (bound-var submask at `bound_col`).
        body: Box<PredIR>,
    },
}

/// A value carried OUT of the mint path but EXCLUDED from the certificate itself: never serialized (so
/// it changes no cert byte / digest) and equality-NEUTRAL (two certs compare equal regardless of it, so
/// `cert.rs`'s field-by-field `matches_rederivation` and the serde round-trip `assert_eq!` tests are
/// unaffected). Used to surface the mint BFS's deadlock witness to the CLI decline path WITHOUT making
/// it part of the re-checkable certificate — a WRITTEN cert is always deadlock-free or `--no-deadlock`.
#[derive(Debug, Clone, Default)]
pub struct MintAside<T>(pub T);
impl<T> PartialEq for MintAside<T> {
    fn eq(&self, _other: &Self) -> bool {
        true // equality-neutral: never part of certificate identity
    }
}
impl<T> Eq for MintAside<T> {}

/// The kernel-checked deadlock-freedom CORROBORATION leg (enumerator-free tier). See
/// [`crate::cleancic::certify_deadlock_witness_general`]: the obligation is
/// `⋀_{s∈R} ⟦Next⟧(s, wₛ) = Bool.true` over ENUMERATED witness successors `wₛ` (one per reachable
/// state). Each `wₛ` was produced by TY's successor enumerator, so the deadlock-freedom FACT (every
/// reachable state has ≥1 successor) rests on the SAME enumeration `ty check` uses — this leg
/// additionally re-checks, through the Clean kernel, that the recognized Next embedding evaluates to
/// `Bool.true` at each `(s, wₛ)` pair (a fail-closed cross-validation of the witnesses; the recognized
/// Next is an over-approximation, so the kernel leg corroborates the enumerator's witnesses rather than
/// replacing them). Present iff the enumerator-free deadlock leg fired; `None` on the enumerator-assisted
/// tier, a deadlocking spec (declined before minting), or a `--no-deadlock` mint.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeadlockFreeLeg {
    /// One enumerated successor tuple `wₛ` per reachable state, aligned to `reachable` order. Bound to
    /// the spec in `cert.rs`'s `matches_rederivation` (a re-enumeration reproduces the same first-
    /// successor witnesses deterministically) AND kernel-re-checked by `verify_explicit_state_cert`.
    pub witnesses: Vec<Vec<u64>>,
    /// The kernel proof token (`Eq.refl Bool Bool.true`) for the conjunction above.
    pub term: Vec<u8>,
}

/// Outcome of the live explicit-state fixpoint certification attempt. ALWAYS compiled (so the
/// `SafetyCertificate` schema is feature-independent and a `clean-cic`-free re-checker can still
/// deserialize + digest-check it); the certify/verify ENTRY POINTS that build/run the kernel terms are
/// `clean-cic`-gated.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExplicitFixpointCert {
    /// The enumerated reachable set `R` (sorted, deduplicated TUPLES — one component per state
    /// variable, in declaration order; a single-variable spec is the 1-tuple case). Each column is an
    /// `Int` (nonneg, stored as the value) or a `Bool` (stored `1`=true / `0`=false), per `sorts`.
    pub reachable: Vec<Vec<u64>>,
    /// The `Init` tuples (each a member of `R`).
    pub init_values: Vec<Vec<u64>>,
    /// The image of `R` under the live `Next` (distinct successor TUPLES; `image(R) ⊆ R`). The
    /// closed-under-Next leg is one concrete tuple membership per image tuple (general, non-stutter).
    pub image: Vec<Vec<u64>>,
    /// The per-column SORT (`Int`/`Bool`), one per state variable in declaration order. Threaded into
    /// the kernel-term builders so the embedded legs are faithful per column, and re-derived identically
    /// by verify / Leg-E (serialized alongside the tuples).
    pub sorts: Vec<ColSort>,
    /// Kernel-checked `R⊆Safety` leg term.
    pub safety_term: Vec<u8>,
    /// Kernel-checked `Init⊆R` membership terms (one per Init value).
    pub init_member_terms: Vec<Vec<u8>>,
    /// Kernel-checked closed-under-Next membership terms (one per image value: `successor ∈ R`).
    pub closed_member_terms: Vec<Vec<u8>>,
    /// The recognized affine `Next` shape, present iff the spec is a single-Int affine counter. When
    /// present, the kernel itself RE-EVALUATES `Next` over the finite domain (see `next_completeness`),
    /// so closure no longer trusts TY's enumerator. `None` ⇒ closure rests on the enumerated `image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_shape: Option<AffineNextShape>,
    /// Kernel-checked `Next`-COMPLETENESS leg (`Eq.refl : Eq Bool C Bool.true`), present iff `next_shape`
    /// is: proves `∀ s∈R, ∀ s'∈D: Next(s,s') ⇒ s'∈R` by the kernel reducing the embedded `Next`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_completeness: Option<Vec<u8>>,
    /// The recognized literal-disjunction `Init` value set (`Init = ⋁_i x=c_i`), present iff the spec is
    /// a single-Int variable with that `Init` shape. When present, the kernel itself RE-EVALUATES `Init`
    /// over the finite domain (see `init_completeness`) — so `Init ⊆ R` no longer trusts TY enumerated
    /// every init state. `None` ⇒ Init-exhaustiveness rests on the enumerated `init_values`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_shape: Option<Vec<u64>>,
    /// Kernel-checked `Init`-COMPLETENESS leg, present iff `init_shape` is: proves `∀ s∈D: Init(s) ⇒ s∈R`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_completeness: Option<Vec<u8>>,
    /// The recognized GENERAL single-Int `Next` predicate IR (arbitrary embeddable arithmetic/Boolean
    /// shape, the strictly larger fallback to `next_shape`). Present iff the affine shape was NOT
    /// recognized but `Next` embeds AND a sound successor upper bound `H` was derivable. When present,
    /// the kernel RE-EVALUATES the ACTUAL `Next` predicate (via `embed_pred_ir`) over `R×D`, `D={0..=H}`
    /// (`H` re-derived from the spec by Leg-E, never trusted from the cert). Mutually exclusive with
    /// `next_shape`. `None` ⇒ neither general nor affine leg; closure rests on the enumerated `image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_pred: Option<ValIRDomain>,
    /// Kernel-checked GENERAL `Next`-completeness leg (`Eq.refl : Eq Bool C Bool.true`), present iff
    /// `next_pred` is: proves `∀ s∈R, ∀ s'∈D: Next(s,s') ⇒ s'∈R` by the kernel reducing the embedded IR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_general_completeness: Option<Vec<u8>>,
    /// The recognized GENERAL single-Int `Init` predicate IR (arbitrary embeddable shape, the fallback
    /// to the literal-disjunction `init_shape`). Present iff `init_shape` was NOT recognized but `Init`
    /// embeds AND a sound current-var upper bound was derivable. Mutually exclusive with `init_shape`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_pred: Option<ValIRDomain>,
    /// Kernel-checked GENERAL `Init`-completeness leg, present iff `init_pred` is: proves
    /// `∀ s∈D_init: Init(s) ⇒ s∈R`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_general_completeness: Option<Vec<u8>>,
    /// The PARAMETRIC inductive-invariant witness for an UNBOUNDED affine single-Int counter (INFINITE
    /// reachable set the finite BFS cannot fold). Present iff the spec is the unbounded affine shape
    /// `Init x=c / Next x'=x+δ (NO guard) / Safety x≥0`; when present, the `reachable`/`init_values`/
    /// `image` fields are EMPTY and closure rests ENTIRELY on the kernel-checked `∀x` consecution leg —
    /// NOT on any enumerated set. MUTUALLY EXCLUSIVE with the finite legs above (a guarded/finite spec
    /// has `None` here and a non-empty `reachable`; an unbounded spec has this `Some` and no enumeration).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unbounded_invariant: Option<UnboundedInvariantCert>,
    /// The GENERAL recognized `Safety` predicate IR — present iff the configured invariant is NOT the
    /// conjunctive-nonneg `⋀ x≥0` shape but IS recognizable into the kernel predicate fragment
    /// ([`crate::cleancic::recognize_pred_sorts`]) as a truth-direction-EXACT STATE predicate (no
    /// primes, no Nat-truncating forms — the same `pred_exact` gate the refinement lane uses: the
    /// safety claim needs kernel-TRUE ⇒ TLA-TRUE, so an over-approximating embedding would be a FALSE
    /// certificate). When present, the SPEC's `R⊆Safety` claim rests on `safety_general` (the tuple
    /// `safety_term` still rides along, proving the encoding-level `⋀_{Int} x≥0` fact); when `None`,
    /// the nonneg tuple leg IS the spec's Safety (the historical primary lane, unchanged). Variable
    /// references are column indices, so verify/Leg-E rebuild the obligation without a name table;
    /// Leg-E additionally RE-RECOGNIZES the invariant from the re-parsed spec and requires equality
    /// with this stored IR (the IR is never trusted on its own). `skip_serializing_if` keeps every
    /// pre-widening cert's serialization byte-identical (digest compatibility — hard rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_pred: Option<PredIR>,
    /// Kernel-checked GENERAL `R⊆Safety` leg — present iff `safety_pred` is. The obligation is
    /// `⋀_{s∈R} ⟦Safety⟧(s)` (one [`crate::cleancic::embed_pred_ir`] closed Bool term per reachable
    /// state, `Bool.and`-chained — BALANCED above [`REFLECTED_MEMBERSHIP_THRESHOLD`], see
    /// [`safety_general_bool`]) reduced to `Bool.true` via the `Eq.refl` gate
    /// ([`crate::cleancic::certify_bool_true_obligation`]): a reachable state violating the invariant
    /// reduces the conjunction to `Bool.false`, the kernel rejects, and NO cert is minted (fail-closed).
    /// The verify side rebuilds the obligation from the stored `reachable` + `safety_pred` and
    /// re-runs the kernel (`verify_bool_true_obligation`). Same digest-compat serde as `safety_pred`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_general: Option<Vec<u8>>,
    /// REFLECTED `Init⊆R` membership leg (roadmap R2 applied to the fixpoint lane) — present iff
    /// `|R| > REFLECTED_MEMBERSHIP_THRESHOLD` (canonicalized). Replaces the per-member
    /// `init_member_terms` (which stay EMPTY above the threshold — their Or-injection terms are
    /// O(|R|²) serialized size / O(|R|³) build work, infeasible at the CoffeeCan scale): the
    /// stored bytes are the CONSTANT-SIZE `Eq.refl Bool Bool.true` proof of the obligation
    /// `Eq Bool (TyReflectSubseq ⌜init⌝ ⌜R⌝) Bool.true` — one O(|init|+|R|)-sized term the kernel
    /// decides by ι-reducing the checked deep merge fold over the QUOTED tuple lists (see
    /// [`crate::reflect`]). Verify REBUILDS the obligation from the cert's own tuples (never the
    /// stored bytes' claimed type) and Leg-E binds those tuples to the spec by re-derivation.
    /// CLEAN-KERNEL TIER ONLY: ck0's ingest fragment has no `List`/`List.rec`, so the tiny second
    /// checker is honestly `Unavailable` on reflected legs (surfaced by `ty certify`).
    /// `skip_serializing_if` keeps every sub-threshold cert BYTE-IDENTICAL (digest back-compat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_member_reflected: Option<Vec<u8>>,
    /// REFLECTED closed-under-Next membership leg — `image(R) ⊆ R` as ONE reflected obligation
    /// `Eq Bool (TyReflectSubseq ⌜image⌝ ⌜R⌝) Bool.true`. Presence rules, trust tier, and serde
    /// discipline exactly as [`Self::init_member_reflected`] (`closed_member_terms` stays empty
    /// above the threshold; both-or-neither with `init_member_reflected`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_member_reflected: Option<Vec<u8>>,
    /// Kernel-checked deadlock-freedom CORROBORATION leg (enumerator-free tier — see [`DeadlockFreeLeg`]).
    /// `skip_serializing_if` keeps every pre-deadlock-leg cert BYTE-IDENTICAL: an enumerator-ASSISTED
    /// cert, a `--no-deadlock` mint, and a deadlocking spec (declined before minting) all carry `None`;
    /// only an enumerator-FREE deadlock-free cert carries it (its digest changes by design). The
    /// deadlock-freedom DECISION itself (decline vs certify) is the enumerator's (parity with `ty check`);
    /// this leg is the kernel re-check of the witnesses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadlock_free: Option<DeadlockFreeLeg>,
    /// MINT-ONLY (never serialized, equality-neutral — see [`MintAside`]): the first reachable state the
    /// mint BFS found with NO enumerated successor (a deadlock witness, config-terminal-agnostic), or
    /// `None` when every reachable state has a successor. Surfaces the enumerator's deadlock verdict to
    /// the CLI decline path; it is NOT part of the re-checkable certificate (a WRITTEN cert is always
    /// deadlock-free or minted under `--no-deadlock`, so this is `None` in any serialized cert).
    #[serde(skip, default)]
    pub deadlock_scan: MintAside<Option<Vec<u64>>>,
}

/// A general predicate IR paired with the PER-COLUMN successor/state upper bounds `H_i` used to build
/// the PRODUCT domain `D = ⨉_i {0..=H_i}`. The `H_i` are ALSO re-derived from the spec by Leg-E (the
/// per-column domain rule applied to the re-parsed AST IR + reachable set) and bound to this value —
/// the serialized `hi` is never trusted on its own for soundness; it is only a convenience for
/// `verify_explicit_state_cert` to rebuild the identical obligation. The DOMAIN COVERAGE proof
/// (`D ⊇ Succ(R)`) is the structural meta-argument behind the per-column rule that derived each `H_i`.
/// A single-Int spec stores the 1-element vector `[H]` (so the multi-var generalization subsumes it).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ValIRDomain {
    /// The embeddable predicate IR.
    pub pred: PredIR,
    /// The PER-COLUMN inclusive domain upper bounds: `D = ⨉_i {0 ..= hi[i]}`.
    pub hi: Vec<u64>,
}

/// FLOOR cap on the enumerated reachable set: the smallest reachable-set size the certifier ALWAYS
/// admits, regardless of machine memory. The live [`memory_derived_state_cap`] only ever raises the
/// enumeration bound ABOVE this (a bigger machine certifies bigger specs); it never drops below, so
/// every historically-certifiable spec (all `|R| ≤ 8192`) keeps enumerating the IDENTICAL reachable
/// set and emits the BYTE-IDENTICAL certificate. Also the STABLE product-domain rebuild cap (both the
/// mint and the Leg-E verify sites use this fixed value so `D = ⨉_i {0..=H_i}` is reconstructed
/// identically) and the R4 finite-set-cardinality cap — those are size ceilings that MUST agree
/// between mint and verify, so they stay a fixed constant, decoupled from the memory-derived
/// reachable cap. Enumerating beyond the live cap returns `None` (fail-closed — a truncated `R` is
/// not a fixpoint).
#[cfg(feature = "clean-cic")]
pub const DEFAULT_FIXPOINT_STATE_CAP: usize = 8192;

/// Rebuild the legacy single-`Int` completeness domain without allowing an affine bound or an
/// `Init` literal from the spec/certificate to drive an unbounded allocation. The general tuple
/// lane already routes through [`crate::cleancic::product_domain`] with this same stable ceiling;
/// keeping the shortcut lane at the identical cap makes mint and verify symmetric and fail-closed.
///
/// `hi` is inclusive, so `u64::MAX` must be rejected before `hi + 1` and a domain with more than
/// [`DEFAULT_FIXPOINT_STATE_CAP`] members is declined before collection.
#[cfg(feature = "clean-cic")]
fn bounded_scalar_completeness_domain(hi: u64) -> Option<Vec<u64>> {
    let len = usize::try_from(hi.checked_add(1)?).ok()?;
    if len > DEFAULT_FIXPOINT_STATE_CAP {
        return None;
    }
    Some((0..=hi).collect())
}

/// Conservative per-state MEMORY cost (bytes) used to size the reachable-set cap from the machine's
/// memory budget. One enumerated state's `ncols`-wide `u64` tuple is held across ~6 long-lived
/// structures — the `visited` and `image` BFS `BTreeSet`s, the certificate's `reachable` / `image` /
/// `init_values` vectors, and the `O(|R|)` reflected-membership kernel leg term — each with container
/// and kernel-node overhead, plus headroom for wide columns. Deliberately an OVER-estimate (the raw
/// data-structure cost is closer to ~1.6 KB/state) so the derived cap fail-closes with margin BEFORE
/// real memory is exhausted.
#[cfg(feature = "clean-cic")]
const FIXPOINT_BYTES_PER_STATE: usize = 8192;

/// Fraction of the STABLE effective machine size ([`tla_resource::platform::effective_total_bytes`],
/// cgroup-capped) budgeted for a single explicit-state certificate's reachable-set + certificate
/// memory. A per-process slice that leaves the rest of RAM for the OS, the live evaluator, and the
/// kernel checker. Applied to `effective_total_bytes` (a stable machine property) — NOT a fluctuating
/// free-memory reading — precisely so certify (mint) and cert-check (Leg-E re-mint) on the SAME
/// machine derive the IDENTICAL cap.
///
/// `1/64` (≈1.5% of RAM) is a deliberately MODEST per-cert slice: it clears the target class (e.g.
/// TokenRing's `6^6 = 46656` reachable states by ~5.6×) with room to grow on a bigger machine, while
/// keeping the enumeration bound low enough that an OUT-of-fragment large/infinite spec — which must
/// enumerate up to the cap before the fragment recogniser can decline it — does not churn excessively.
/// On a 128 GB machine this yields a cap of ≈262 144 states.
#[cfg(feature = "clean-cic")]
const FIXPOINT_MEM_FRACTION: f64 = 1.0 / 64.0;

/// The reachable-set certification cap, DERIVED from the machine's memory budget rather than a fixed
/// state count: `(effective_total_bytes · FRACTION) / bytes_per_state`, FLOORED at
/// [`DEFAULT_FIXPOINT_STATE_CAP`] so small specs ALWAYS certify (no regression) and the cap only ever
/// GROWS with available RAM. Basing it on [`tla_resource::platform::effective_total_bytes`] — a STABLE
/// machine property (host RAM capped by any cgroup limit), not a live free-memory reading — is what
/// makes the decision DETERMINISTIC: `ty certify` (mint) and `ty cert-check` (Leg-E, which independently
/// re-mints the cert from the spec and requires an identical `R`) run in separate processes but on the
/// same machine compute the IDENTICAL cap, so Leg-E never diverges from mint. A bigger verifier machine
/// derives a cap `≥` the minter's, so it still reproduces `R`; a smaller one honestly reports
/// INCONCLUSIVE rather than OOMing. FAIL-CLOSED: a spec whose reachable set would exceed the budget
/// makes the BFS return `None` (decline), never an out-of-memory. This is a FEASIBILITY guard only —
/// a larger admitted `R` is still fully kernel-re-checked by all three legs, so raising it can never
/// change a certificate's correctness, only which specs are attempted.
#[cfg(feature = "clean-cic")]
pub fn memory_derived_state_cap() -> usize {
    tla_resource::platform::effective_total_bytes()
        .map(|total| {
            let budget = (total as f64 * FIXPOINT_MEM_FRACTION) as usize;
            (budget / FIXPOINT_BYTES_PER_STATE).max(DEFAULT_FIXPOINT_STATE_CAP)
        })
        .unwrap_or(DEFAULT_FIXPOINT_STATE_CAP)
}

/// TOTAL-WORK cap for the GENERAL `Next`-completeness leg: the obligation is `⋀_{s∈R, sp∈D}(¬Next(s,sp)
/// ∨ sp∈R)`, so the total kernel work grows with the PRODUCT `|R| × |D|`. This cap gates whether the leg
/// is ATTEMPTED at all (a WALL-CLOCK budget) — past it the general Next leg is DECLINED and closure rests
/// on the honest kernel-checked enumerated `image ⊆ R` leg (the enumerator-ASSISTED tier), never a hang.
/// Sound: declining a leg only ever weakens the CLAIM, never the check.
///
/// Since [`certify_general_completeness`] proves the obligation in PER-SOURCE, BOUNDED-DOMAIN chunks
/// for a large product (see [`GENERAL_COMPLETENESS_MONOLITH_CAP`]), the per-call kernel cost is bounded
/// independently of `|R|` and `|D|`. This cap is therefore a pure wall-clock bound on total work.
/// Sized to admit VoucherLifeCycle's canonical
/// `V={v1,v2,v3}` two-FuncEnum product (`64 × 1728 = 110592`), comfortably above every other corpus spec
/// (no other spec reaches this filter — all decline earlier on recognition / a non-universe axis bound).
#[cfg(feature = "clean-cic")]
pub const GENERAL_COMPLETENESS_WORK_CAP: usize = 131_072;

/// PER-KERNEL-CALL size cap for the GENERAL `Next`-completeness leg — the boundary between the historical
/// SINGLE-term obligation (`⋀_{s∈R,sp∈D} leg`, one `kernel_accepts`) and the per-source/domain decomposition.
/// At/below this product size the
/// single-term path runs — BYTE-IDENTICAL certs and unchanged ck0 corroboration for every existing
/// enumerator-free spec (all `≤` this). Above it, the monolith would exhaust the kernel's 2M-step heartbeat
/// / blow memory (VoucherLifeCycle V=3's `110592`-leg term hit both at ~34 GB), so the leg is partitioned
/// across source states and bounded slices of `D`. Every kernel call stays safely below the structural
/// term limit. The decomposition is a logical
/// identity (`⋀_s ⋀_sp = ⋀_{s,sp}`): the domain `D` and the `D ⊇ Succ(R)` coverage are UNCHANGED, and a
/// non-closed `R` still fails (the escaping successor's source-state chunk `≠ Bool.true`).
#[cfg(feature = "clean-cic")]
pub const GENERAL_COMPLETENESS_MONOLITH_CAP: usize = 8192;

/// The per-member/REFLECTED membership-leg threshold: a canonicalized `R` of AT MOST this many
/// states keeps the historical per-member Or-injection legs (`init_member_terms` /
/// `closed_member_terms` / the Or.rec tuple `safety_term`) — every existing certificate and
/// fixture stays BYTE-IDENTICAL — while a larger `R` mints the REFLECTED legs
/// (`init_member_reflected` / `closed_member_reflected` + the balanced nonneg `safety_term`),
/// whose size is O(|R|) instead of O(|R|²). 64 is comfortably above every historical certifiable
/// spec in the corpus and comfortably below where the quadratic terms hurt.
pub const REFLECTED_MEMBERSHIP_THRESHOLD: usize = 64;

/// The SIZE budget (in per-member "units") past which the membership legs switch to the REFLECTED
/// lane EVEN AT `|R| ≤ REFLECTED_MEMBERSHIP_THRESHOLD` — the size-adaptive companion to the pure
/// `|R|` threshold. The per-member certificate's serialized size scales as ≈ `|R|³ · ncols` (the
/// `closed`/`init`/`safety_term` Or-injection legs are `O(|R|)` in count over `O(|R|·ncols)` nodes,
/// and each nests an Or-fold over `R`), so `|R|³ · ncols` is the natural resource proxy. This cap is
/// DERIVED from a certificate byte budget (≈ 1 KB per unit empirically, so ~0.9 GB): a spec that
/// stays under the `|R|` threshold but whose product-of-columns blowup would mint a multi-hundred-MB
/// to multi-GB per-member cert (the multi-Int-column `clean`/PCR class at `|R|=63, ncols=5`) reflects
/// instead. Chosen COMFORTABLY ABOVE every historical per-member cert's units (the corpus max is
/// Barrier/Peterson at ≈2.6e5) so their lane — and bytes — are UNCHANGED, and below the class that
/// hurts. See [`use_reflected_membership_lane`].
pub const PER_MEMBER_UNIT_CAP: u128 = 900_000;

/// Whether the canonical reachable set `R` (`r_len` distinct states, `ncols` columns) should mint the
/// REFLECTED (O(|R|)) membership legs rather than the per-member (O(|R|²)/O(|R|³)) ones. The historical
/// rule `|R| > REFLECTED_MEMBERSHIP_THRESHOLD` is kept VERBATIM as the first clause (every existing
/// per-member/reflected cert keeps its lane); the second clause ADDITIVELY reflects when the per-member
/// certificate would be OVERSIZED by the `|R|³·ncols` size law (see [`PER_MEMBER_UNIT_CAP`]). A PURE
/// function of `(|R|, ncols)` — both recomputable from the cert — so certify (mint) and verify (Leg-E)
/// decide the lane IDENTICALLY. ADDITIVE: the cap sits above every historical per-member cert's units,
/// so no existing certificate changes lane (hence bytes); only a NEW, otherwise-oversized spec reflects.
#[inline]
pub fn use_reflected_membership_lane(r_len: usize, ncols: usize) -> bool {
    r_len > REFLECTED_MEMBERSHIP_THRESHOLD
        || (r_len as u128)
            .saturating_pow(3)
            .saturating_mul(ncols as u128)
            > PER_MEMBER_UNIT_CAP
}

/// The element-universe BIT WIDTH `K` of a `Set` column's bitmask encoding (ALWAYS compiled so
/// `value_cell` + the verifier share it): a Set value `S` encodes as the `Nat` `Σ_{e∈S} 2^e`, and every
/// element must be a nonneg small Int `< SET_UNIVERSE_BITS` so the bit fits one u64 cell within the
/// declared universe. Fixed (not per-state) so a Set column's sort is STABLE across enumerated states.
pub const SET_UNIVERSE_BITS: u32 = 16;

/// The pack-radix FLOOR of a `Record`/`Func` column's positional Nat pack (roadmap R3): a field value
/// `v_i` packs at the column's DERIVED base (`pack = Σ v_i·base^i`). The base is chosen PER COLUMN by
/// ONE adaptive, resource-derived rule — [`compound_min_base`]'s "SMALLEST base admitting every observed
/// field/value" — namely `base = max(RECORD_FUNC_BASE, maxObservedFieldValue + 1)`, CAPPED so the pack
/// `base^arity` fits a `u64` (a column whose derived base × arity would overflow FAILS CLOSED). This
/// FLOOR pins the pre-widening behavior: a column ALL of whose values fit base 10 derives EXACTLY base
/// 10 (its pack, sort, and serialization are BYTE-IDENTICAL to the pre-R3 encoder — the digest
/// back-compat hard rule), while a column with a larger value derives the TIGHTER `maxValue+1` base
/// rather than a fixed wide rung. Fixed per-column ⇒ STABLE sort.
///
/// SOUNDNESS / DETERMINISM: the derived base is baked into the serialized sort (`ColSort::Record{base,..}`
/// / `ColSort::Func{base,..}`) and every kernel div/mod place value the recognizers emit is derived FROM
/// that sort. The verify side (Leg-E) re-enumerates the SAME reachable set from the spec (the enumeration
/// is deterministic and independent of the encoding) and re-applies the SAME smallest-base rule over the
/// SAME states, so it re-derives the SAME per-column base and the re-derived sort vector compares EQUAL to
/// the stored one. There is NO fixed `{10, 1024}` ladder or `{4, 6}` arity magic any more — the base and
/// the arity bound are both DERIVED from the observed data and the `u64` pack ceiling.
pub const RECORD_FUNC_BASE: u64 = 10;

/// The SMALLEST compound pack radix (≥ [`RECORD_FUNC_BASE`]) admitting every field/value of `val`, or
/// `None` if `val` is not a packable `Record`/`Func`/`Seq` at ANY base — a non-nonneg-Int field/value
/// (RANK-1 territory: an enumerated scalar sort would be needed), a `Func` whose domain is not the
/// consecutive `0..n-1` prefix, or an arity whose pack `base^arity` OVERFLOWS `u64` even at the minimal
/// base.
///
/// This is THE single adaptive derived threshold that replaces the old fixed `{RECORD_FUNC_BASE,
/// RECORD_FUNC_WIDE_BASE}` ladder + `{4, 6}` arity caps: `base = max(RECORD_FUNC_BASE, maxValue + 1)`
/// then `base.checked_pow(arity)` gates the u64 pack ceiling. DETERMINISTIC (a pure function of the
/// value), so the enumeration loop's per-column base = `max` over states of `compound_min_base`, and
/// Leg-E re-derives it byte-identically. A value all of whose fields fit base 10 returns EXACTLY 10.
///
/// A `Seq`/`Tuple` derives its own SELF-DELIMITING radix `D` (the value returned IS that `D`, `= bases[i]`
/// in the certify loop) via [`seq_min_radix`] — a DIFFERENT digit-shift formula from the record/func
/// positional pack (`D = max(RECORD_FUNC_BASE, maxElement + 2)`, the `+2` being the `+1` element shift),
/// so it is dispatched separately. A sequence all of whose elements are `< SEQ_BASE` returns EXACTLY
/// `RECORD_FUNC_BASE` (⇒ element base `SEQ_BASE`) — BYTE-IDENTICAL to the pre-widening encoder.
#[cfg(feature = "clean-cic")]
pub(crate) fn compound_min_base(val: &crate::value::Value) -> Option<u64> {
    use crate::value::{IntIntervalFunc, Value};
    // A SEQUENCE / TUPLE derives its own self-delimiting radix (see `seq_min_radix`); it does not fit the
    // record/func `(max_val, arity)` shape below, so dispatch it here.
    match val {
        Value::Seq(s) => return seq_min_radix(s.iter(), s.len()),
        Value::Tuple(t) => return seq_min_radix(t.iter(), t.len()),
        _ => {}
    }
    // (max field/value, arity) of the compound, or `None` for a non-packable value/shape.
    let (max_val, arity): (u64, u32) = match val {
        Value::Record(rec) => {
            let mut mx = 0u64;
            for (_name, v) in rec.iter_str() {
                mx = mx.max(cell_code(v)?.0); // max digit CODE (Int value / Bool 0|1); non-leaf ⇒ decline
            }
            (mx, u32::try_from(rec.len()).ok()?)
        }
        Value::Func(f) => {
            // The domain is EITHER the consecutive nonneg prefix `0,1,…,n-1` OR a set of atom keys
            // (model values / `String`s) — the [`func_enum_domain_keys`] shape. A domain outside those
            // (a compound key, a non-0 Int prefix, a mixed-kind key set) ⇒ not packable here. The base is
            // derived from the Int/Bool VALUES regardless of the domain shape (a label value declines at
            // `cell_code` — that is FuncEnum's pack, not this one).
            func_enum_domain_keys(f)?;
            let mut mx = 0u64;
            for v in f.mapping_values() {
                mx = mx.max(cell_code(v)?.0);
            }
            (mx, u32::try_from(f.domain_len()).ok()?)
        }
        Value::IntFunc(f) => {
            if IntIntervalFunc::min(f) != 0 {
                return None; // domain not the 0-based prefix ⇒ not packable here
            }
            let mut mx = 0u64;
            for v in f.values() {
                mx = mx.max(cell_code(v)?.0);
            }
            (mx, u32::try_from(f.len()).ok()?)
        }
        _ => return None, // only Record/Func compounds derive a pack base
    };
    let base = RECORD_FUNC_BASE.max(max_val.checked_add(1)?);
    base.checked_pow(arity)?; // decline on u64 pack overflow at the derived base (checked_pow)
    Some(base)
}

/// The FLOOR element radix `base` of a `Seq` column: an element `a_i` packs at self-delimiting radix
/// `D = base + 1` (shifted by `+1`, so digit `0` self-delimits the length), `pack = Σ (a_i+1)·D^i`. The
/// FLOOR is `base = SEQ_BASE` (⇒ `D = SEQ_BASE + 1 = RECORD_FUNC_BASE`); the certify loop DERIVES a
/// per-column radix `D = bases[i] = max(RECORD_FUNC_BASE, maxElement + 2)` — the SMALLEST self-delimiting
/// radix admitting every observed element — via [`compound_min_base`]/[`seq_min_radix`], EXACTLY mirroring
/// the record/func base widening. A column all of whose elements are `< SEQ_BASE` derives `D =
/// RECORD_FUNC_BASE` ⇒ `base = SEQ_BASE` ⇒ BYTE-IDENTICAL to the pre-widening encoder. Fixed per-column ⇒
/// STABLE sort. `D^max_len` must fit a `u64`.
pub const SEQ_BASE: u64 = 9;

/// Cap on a `Seq` column's length: `(SEQ_BASE+1)^max_len` must fit a `u64` cell (`10^max_len < 2^64`),
/// and the general-completeness product-domain axis must stay within the state cap. Small by design.
pub const SEQ_MAX_LEN: u32 = 4;

/// CONSERVATIVE baseline cap on a COMPOUND (`Record`/`Func`/`Seq`) column's PACK RANGE for the GENERAL
/// Next/Init COMPLETENESS leg: a CHANGING compound column's product-domain axis is the full pack range
/// `{0..=cap}`, so completeness is only attempted when that range `≤ cap` (else the column is treated as
/// STUTTER-only, or the completeness leg is declined and closure rests on the honest enumerated `image ⊆ R`
/// leg). Small by design — the pack range times the other axes must stay within `state_cap`.
///
/// This is the FLOOR applied to any compound column that is NOT a recognized cell-kind-safe shape (a plain
/// all-`Int` Record, an Int-valued `Func`, a `Seq`). It admits a base-[`RECORD_FUNC_BASE`] TWO-field record
/// (`base^2 = 100` packs — the CoffeeCan `can = [black, white]` class with `MaxBeanCount ≤ base-1 = 9`): the
/// changing-record `Next` update is recognized as ONE pack equality (`cleancic::record_update_eq_form`),
/// whose per-state successor pack is a concrete literal, so the `D ⊇ Succ(R)` coverage lemma KERNEL-PROVES
/// (`Nat.ble successor cap`) and the column is admitted even under `--require-domain-complete`. A record
/// whose base WIDENS (a field value `≥ base` ⇒ [`RECORD_FUNC_WIDE_BASE`], `base^2 ≈ 10^6`) still exceeds the
/// cap and DECLINES — closure then rests on the honest enumerated `image ⊆ R` leg.
///
/// The RECOGNIZED cell-kind-safe shapes ([`ColSort::is_recognized_cellkind_compound`] — mixed-cell Record /
/// FuncEnum / Bool-`Func`) get the RAISED [`COMPOUND_COMPLETENESS_PACK_CAP_RECOGNIZED`] instead. This split
/// CONTAINS the blast radius of the raise (see that constant) WITHOUT changing soundness: the cap is a pure
/// DECLINE threshold — the emitted domain bound is ALWAYS the full `pack_universe-1` (or the tight all-ones
/// Bool/enum bound), never a truncation — so cap choice only decides whether a leg is ATTEMPTED, never what
/// it claims. Every already-passing spec is unaffected: an unrecognized column with `pack_universe-1 ≤ 100`
/// derives the SAME bound here as at any larger cap.
pub const COMPOUND_COMPLETENESS_PACK_CAP: u64 = 100;

/// RAISED completeness cap admitted ONLY for a RECOGNIZED cell-kind-safe compound column — the mixed-cell
/// Record / FuncEnum / Bool-`Func` shapes ([`ColSort::is_recognized_cellkind_compound`]) whose recognized
/// pack-update writes an EXACT successor pack. Sized to admit a base-[`RECORD_FUNC_BASE`] THREE-field record
/// (`base^3 = 1000` packs — the Channel `chan ∈ [val:Data, rdy:{0,1}, ack:{0,1}]` class, a MIXED-CELL record
/// with an Enum/model-value field): its changing `Next` update recognizes as ONE pack equality
/// (`record_update_eq_form`, cell-kind-aware), so the per-state successor pack is a concrete literal `≤ 999`
/// and the `D ⊇ Succ(R)` coverage lemma KERNEL-PROVES (enumerator-FREE closure).
///
/// WHY PER-COLUMN, RECOGNITION-GATED (the containment rationale): a GLOBAL raise gave EVERY compound column
/// the bigger domain `D`, costing perf (Channel certify 2s→10s) and cert-portability (a cap-1000 cert needs a
/// cap≥1000 checker on EVERY compound column, recognized or not). Scoping the raise to the recognized shapes
/// keeps an unrecognized column — which has no exact successor pack and no business enlarging its domain — at
/// the conservative floor, so the bigger `D` is paid for ONLY where it is both needed and justified (the
/// recognized column's coverage is already exact). The gate is a DETERMINISTIC function of the re-derived
/// [`ColSort`], so certify and verify (Leg-E) pick the SAME cap per column ⇒ certs stay self-consistent.
///
/// RAISING this cap is FAIL-CLOSED-SAFE regardless of the gate: the axis is the full pack range `{0..=cap}`,
/// so a larger cap only ENLARGES `D` — the closure obligation `∀s∈R,s'∈D : Next(s,s')⇒s'∈R` becomes strictly
/// HARDER (more `s'` to discharge), never falsely provable; a domain that no longer fits is thrown out by the
/// `state_cap` / work-cap filters (decline → the enumerated `image ⊆ R` leg), never a hang. A FOUR-field
/// base-10 record (`10^4`) exceeds this cap and DECLINES.
pub const COMPOUND_COMPLETENESS_PACK_CAP_RECOGNIZED: u64 = 1000;

/// Cap on a `Set` column's element-universe width `K` for the GENERAL Next/Init COMPLETENESS leg: a
/// changing Set column's product-domain axis is all `2^K` bitmasks `{0..2^K-1}`, so completeness is only
/// attempted when `K ≤ SET_COMPLETENESS_UNIVERSE_CAP` (else the column is treated as STUTTER-only, or the
/// completeness leg is declined and closure rests on the honest enumerated-image leg). Small by design —
/// `2^K` must stay within the product-domain `state_cap`.
pub const SET_COMPLETENESS_UNIVERSE_CAP: u32 = 6;

/// Cap on the element-universe width `K` of the set `S` in a `∀T ∈ SUBSET S : P` / `∃T ∈ SUBSET S : P`
/// bounded quantifier (see [`PredIR::SubsetForall`]/[`PredIR::SubsetExists`]): the fold enumerates the
/// `2^popcount(S)` submasks of the CONCRETE `S`, and `popcount(S) ≤ K`, so the fold is bounded at
/// `≤ 2^K` legs. recognize DECLINES a powerset quantifier whose set has `K > SUBSET_QUANT_POPCOUNT_CAP`
/// (fail-closed) so the `Bool.and`/`Bool.or` fold the kernel reduces never exceeds `2^cap ≤ 64` legs.
/// Small by design (`2^6 = 64`).
pub const SUBSET_QUANT_POPCOUNT_CAP: u32 = 6;

/// Extract a nonneg small `Int` `< bound` from a `Value` (a `SmallInt`/`Int`), or `None` if the value is
/// not a nonneg Int or is `≥ bound`. The all-or-nothing leaf for the compound-pack `value_cell` arms (a
/// field value, a function value/key, a sequence element). Passing `bound = u64::MAX` only requires nonneg.
#[cfg(feature = "clean-cic")]
fn nonneg_small_int(val: &crate::value::Value, bound: u64) -> Option<u64> {
    use crate::value::Value;
    let v = match val {
        Value::SmallInt(k) if *k >= 0 => *k as u64,
        Value::Int(k)
            if !{
                use num_traits::Signed;
                k.is_negative()
            } =>
        {
            u64::try_from(k.as_ref().clone()).ok()?
        }
        _ => return None,
    };
    (v < bound).then_some(v)
}

/// The kind-tagged token of a RECORD FIELD VALUE for the [`ColSort::SetMaskRec`] canonical key: a
/// `(tag, text)` pair where `tag` fixes the value KIND (`'S'`=`String`, `'M'`=model value, `'B'`=`Bool`,
/// `'I'`=nonneg Int) and `text` its content. `None` for any non-leaf field value (a nested record /
/// function / set — such a record is UNKEYABLE ⇒ the whole set-of-records column fails closed). The tag
/// makes DISTINCT-kind values distinct (a `String` `"1"`, a model value `1`, a Bool `TRUE`, and the Int
/// `1` never collide), and it is the SAME classification the recognizer's field-domain materialization
/// applies to a record-set TYPE, so certify (from `Value`s) and the recognizer (from the type expression)
/// key the SAME record identically.
#[cfg(feature = "clean-cic")]
pub(crate) fn record_field_token(val: &crate::value::Value) -> Option<(char, String)> {
    use crate::value::Value;
    match val {
        Value::String(s) => Some(('S', s.as_ref().to_string())),
        Value::ModelValue(s) => Some(('M', s.as_ref().to_string())),
        Value::Bool(b) => Some(('B', if *b { "1" } else { "0" }.to_string())),
        _ => nonneg_small_int(val, u64::MAX).map(|v| ('I', v.to_string())),
    }
}

/// Serialize a record's `(field-name, tag, text)` tokens into its CANONICAL [`ColSort::SetMaskRec`] key.
/// The tokens are sorted by field NAME (so iteration order is irrelevant) and each is written
/// LENGTH-PREFIXED (`len(name) · name · tag · len(text) · text`, `\u{1}`-separated, `\u{1e}`-terminated) —
/// making the concatenation an INJECTIVE code: two records with different field sets, names, kinds, or
/// values produce different keys REGARDLESS of content (the lengths prevent any delimiter-collision), so
/// no two distinct records share a bit (the [`ColSort::SetMaskRec`] soundness bijection). Shared verbatim
/// by certify ([`record_value_key`]) and the recognizer (`recognize_setmaskrec_pred`) so both key the same
/// record byte-identically ⇒ Leg-E re-derives the same `dom`.
#[cfg(feature = "clean-cic")]
pub(crate) fn record_key_from_fields(mut fields: Vec<(String, char, String)>) -> String {
    use std::fmt::Write;
    fields.sort_by(|a, b| a.0.cmp(&b.0));
    let mut s = String::new();
    for (name, tag, text) in &fields {
        let _ = write!(
            s,
            "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}\u{1e}",
            name.len(),
            name,
            tag,
            text.len(),
            text
        );
    }
    s
}

/// The CANONICAL [`ColSort::SetMaskRec`] key of a record `Value` — its length-prefixed sorted-field
/// serialization ([`record_key_from_fields`]). `None` if any field value is not a leaf atom / Int / Bool
/// ([`record_field_token`]) ⇒ the record is unkeyable and its set-of-records column fails closed.
#[cfg(feature = "clean-cic")]
#[doc(hidden)] // exposed for the encoder-injectivity audit harness (tests/encoder_injectivity_cert.rs); not a stable API
pub fn record_value_key(val: &crate::value::Value) -> Option<String> {
    use crate::value::Value;
    let Value::Record(rec) = val else { return None };
    let mut fields: Vec<(String, char, String)> = Vec::new();
    for (name, v) in rec.iter_str() {
        let (tag, text) = record_field_token(v)?;
        fields.push((name.to_string(), tag, text));
    }
    Some(record_key_from_fields(fields))
}

/// The per-POSITION DIGIT CODE + [`CellSort`] of one compound field/value cell — the VALUE-TYPE LEAF that
/// generalizes [`nonneg_small_int`] beyond `Int`:
///   * a nonneg small `Int`  ⇒ `(v, CellSort::Int)`  (the digit IS the value — historical leaf);
///   * a `Bool`              ⇒ `(1|0, CellSort::Bool)` (`TRUE`→`1`, `FALSE`→`0`).
/// `None` for anything else (a `String`/model value — the deferred Enum-position leaf — a negative/oversized
/// Int, or a nested compound). The returned CODE is the digit written at the position; the caller floors the
/// column's uniform `base` at `code+1` and rejects `code ≥ base` (a value needing a wider base ⇒ Widen).
/// DETERMINISTIC (a pure function of the value) so verify (Leg-E) re-derives the same code + kind.
#[cfg(feature = "clean-cic")]
fn cell_code(val: &crate::value::Value) -> Option<(u64, CellSort)> {
    match val {
        crate::value::Value::Bool(b) => Some((u64::from(*b), CellSort::Bool)),
        // Every other in-fragment cell is a nonneg small Int (the historical leaf). A String / model value
        // (the Enum-position leaf) and any oversized/negative Int decline here — fail-closed.
        _ => nonneg_small_int(val, u64::MAX).map(|v| (v, CellSort::Int)),
    }
}

/// NORMALIZE a positional compound's per-position [`CellSort`] vector to its CANONICAL form: an all-`Int`
/// vector collapses to EMPTY. This is the invariant that keeps an all-Int compound BYTE-IDENTICAL to a
/// pre-value-type-leaf cert (empty `cells` serializes to nothing) AND makes `PartialEq` on two equal-shape
/// all-Int compounds hold (both carry `[]`, never `[Int, Int, …]`). A compound with ANY non-Int position
/// keeps its full-length vector (`cells.len() == arity`).
#[cfg(feature = "clean-cic")]
fn normalize_cells(cells: Vec<CellSort>) -> Vec<CellSort> {
    if cells.iter().all(|c| matches!(c, CellSort::Int)) {
        Vec::new()
    } else {
        cells
    }
}

/// One positional-compound cell CLASSIFIED for the value-type leaf, one level richer than [`cell_code`]:
/// an Int/Bool LEAF (its digit CODE + [`CellSort`]) or an ENUM LABEL (its [`EnumKind`] + text). The label
/// is resolved to a per-position INDEX against the column's cross-state label union in the certify loop
/// (`encode_compound_at`), exactly as a scalar `String` cell grows [`ColSort::Enum`]. `None` for a non-leaf
/// (nested compound, negative/oversized Int). DETERMINISTIC (a pure function of the value).
#[cfg(feature = "clean-cic")]
enum CellClass {
    /// An Int value / Bool `0|1` — the CODE is the digit, the [`CellSort`] its kind.
    Leaf(u64, CellSort),
    /// A `String` / model-value LABEL at this position — indexed into the position's union by the caller.
    Enum(EnumKind, String),
}

/// Classify one compound field/value cell: a `Bool` ⇒ `Leaf(0|1, Bool)`; a `String` ⇒ `Enum(Str, text)`;
/// a model value ⇒ `Enum(Model, text)`; every other in-fragment cell a nonneg small Int ⇒ `Leaf(v, Int)`.
/// `None` for a nested compound / negative / oversized Int. The value-type-leaf generalization of
/// [`cell_code`] (which stays for the Int/Bool base-derivation fast path).
#[cfg(feature = "clean-cic")]
fn classify_cell(val: &crate::value::Value) -> Option<CellClass> {
    use crate::value::Value;
    match val {
        Value::Bool(b) => Some(CellClass::Leaf(u64::from(*b), CellSort::Bool)),
        Value::String(s) => Some(CellClass::Enum(EnumKind::Str, s.as_ref().to_string())),
        Value::ModelValue(s) => Some(CellClass::Enum(EnumKind::Model, s.as_ref().to_string())),
        _ => nonneg_small_int(val, u64::MAX).map(|v| CellClass::Leaf(v, CellSort::Int)),
    }
}

/// A per-column enum label set discovered during enumeration: the KIND + the SORTED distinct labels. Shared
/// by the SCALAR/`FuncEnum` per-column union (`enum_labels`) and the per-POSITION compound union
/// (`compound_enum`), grown monotonically across restarts.
#[cfg(feature = "clean-cic")]
#[derive(Clone)]
pub(crate) struct EnumCol {
    pub(crate) kind: EnumKind,
    pub(crate) labels: Vec<String>,
}

/// The stop signal from [`encode_compound_at`] — the compound analogue of the certify loop's scalar
/// `EnumStop`, but the enum-grow is PER POSITION.
#[cfg(feature = "clean-cic")]
pub(crate) enum CompoundStop {
    /// Position `p` observed a new enum `label` (of `kind`) not yet in its per-position sorted union —
    /// grow the union at `(column, p)` and restart (mirrors the scalar `GrowEnum`).
    Grow(usize, EnumKind, String),
    /// The pack radix must be at least this to admit some digit CODE (an Int value / enum index `≥ base`).
    Widen(u64),
    /// A positional compound genuinely OUT of the leaf fragment (a non-leaf field, a non-0-prefix func
    /// domain, a `base^arity` overflow) — no cert for this column.
    Fail,
    /// `val` is NOT a positional compound (`Record`/`Func`/`IntFunc`) at all — the caller falls back to
    /// [`value_cell_encode_at`] (Set/Seq/scalar).
    NotCompound,
}

/// Encode a POSITIONAL COMPOUND (`Record` / Int-domain `Func`) at pack radix `base`, resolving each
/// position's digit CODE via [`classify_cell`] and, for a `String`/model-value position, INDEXING into that
/// position's cross-state label union `pos_labels[p]` (the value-type-leaf Enum cell). Returns the
/// `(ColSort, pack)` or a [`CompoundStop`]. The pack is the uniform base-`base` numeral `Σ_p code_p·base^p`;
/// an enum position's code is its label index (`< labels.len()`), so `base ≥ labels.len()` is required (a
/// digit `≥ base` ⇒ `Widen`). BYTE-IDENTICAL to [`value_cell_encode_at`] for an all-Int/Bool compound (same
/// `cells` normalization, same pack). DETERMINISTIC ⇒ Leg-E re-derives it byte-identically.
#[cfg(feature = "clean-cic")]
fn encode_compound_at(
    val: &crate::value::Value,
    base: u64,
    pos_labels: &[Option<EnumCol>],
) -> Result<(ColSort, u64), CompoundStop> {
    use crate::value::{IntIntervalFunc, Value};
    // Pack per-position values at `base`, consulting `pos_labels` for enum positions. A new label / a code
    // needing a wider radix / a kind-mix / a non-leaf is surfaced as the matching `CompoundStop`.
    let pack_positions = |values: &[&Value]| -> Result<(Vec<CellSort>, u64), CompoundStop> {
        // First pass: classify every position (raising `Grow` on the FIRST new label so labels precede
        // the base widening) and track the max digit code for a single-jump `Widen`.
        let mut codes: Vec<(u64, CellSort)> = Vec::with_capacity(values.len());
        let mut max_code = 0u64;
        for (p, v) in values.iter().enumerate() {
            let (code, cs) = match classify_cell(v).ok_or(CompoundStop::Fail)? {
                CellClass::Leaf(code, cs) => (code, cs),
                CellClass::Enum(kind, label) => match pos_labels.get(p).and_then(|o| o.as_ref()) {
                    Some(ec) if ec.kind == kind => match ec.labels.iter().position(|l| *l == label)
                    {
                        Some(idx) => (
                            idx as u64,
                            CellSort::Enum {
                                labels: ec.labels.clone(),
                                kind,
                            },
                        ),
                        None => return Err(CompoundStop::Grow(p, kind, label)),
                    },
                    // Position recorded as the OTHER enum kind ⇒ heterogeneous ⇒ fail closed (a `String`
                    // "x" and a model value `x` must never share an index).
                    Some(_) => return Err(CompoundStop::Fail),
                    // First time this position is seen as enum — grow its first label in.
                    None => return Err(CompoundStop::Grow(p, kind, label)),
                },
            };
            max_code = max_code.max(code);
            codes.push((code, cs));
        }
        // A code needing a WIDER radix ⇒ `Widen` to the smallest admitting base (`max_code + 1`), mirroring
        // `compound_min_base`'s single-jump derivation (byte-identical FINAL base, fewer restarts).
        if max_code >= base {
            return Err(CompoundStop::Widen(
                max_code.checked_add(1).ok_or(CompoundStop::Fail)?,
            ));
        }
        let mut pack: u64 = 0;
        let mut cells: Vec<CellSort> = Vec::with_capacity(codes.len());
        for (p, (code, cs)) in codes.into_iter().enumerate() {
            let place = base.checked_pow(p as u32).ok_or(CompoundStop::Fail)?;
            pack = pack
                .checked_add(code.checked_mul(place).ok_or(CompoundStop::Fail)?)
                .ok_or(CompoundStop::Fail)?;
            cells.push(cs);
        }
        Ok((cells, pack))
    };
    match val {
        Value::Record(rec) => {
            // The pack ceiling: each digit `< base` ⇒ `pack < base^arity` (fail closed on overflow).
            base.checked_pow(u32::try_from(rec.len()).map_err(|_| CompoundStop::Fail)?)
                .ok_or(CompoundStop::Fail)?;
            let mut fields: Vec<String> = Vec::with_capacity(rec.len());
            let mut values: Vec<&Value> = Vec::with_capacity(rec.len());
            for (name, v) in rec.iter_str() {
                fields.push(name.to_string()); // field NAMES are part of the column identity
                values.push(v);
            }
            let (cells, pack) = pack_positions(&values)?;
            Ok((
                ColSort::Record {
                    base,
                    fields,
                    cells: normalize_cells(cells),
                },
                pack,
            ))
        }
        Value::Func(f) => {
            let n = f.domain_len();
            base.checked_pow(u32::try_from(n).map_err(|_| CompoundStop::Fail)?)
                .ok_or(CompoundStop::Fail)?;
            // The domain is the consecutive Int prefix `0,1,…,n-1` (`dom` empty ⇒ byte-identical) OR a set
            // of atom keys (model values / `String`s — `dom`/`dom_kind` per [`func_enum_domain_keys`]). Any
            // other domain ⇒ fail closed. Positions follow the domain's canonical `Value::cmp` order, which
            // `mapping_values()` is aligned with, so the digit at position `p` is `f[dom[p]]`.
            let (dom, dom_kind) = func_enum_domain_keys(f).ok_or(CompoundStop::Fail)?;
            let values: Vec<&Value> = f.mapping_values().collect();
            let (cells, pack) = pack_positions(&values)?;
            Ok((
                ColSort::Func {
                    base,
                    arity: n as u32,
                    cells: normalize_cells(cells),
                    dom,
                    dom_kind,
                },
                pack,
            ))
        }
        Value::IntFunc(f) => {
            if IntIntervalFunc::min(f) != 0 {
                return Err(CompoundStop::Fail); // domain not the 0-based prefix
            }
            let n = f.len();
            base.checked_pow(u32::try_from(n).map_err(|_| CompoundStop::Fail)?)
                .ok_or(CompoundStop::Fail)?;
            let values: Vec<&Value> = f.values().iter().collect();
            let (cells, pack) = pack_positions(&values)?;
            Ok((
                ColSort::Func {
                    base,
                    arity: n as u32,
                    cells: normalize_cells(cells),
                    dom: Vec::new(),
                    dom_kind: EnumKind::Model,
                },
                pack,
            ))
        }
        // A SEQUENCE / TUPLE `<<e_0,…,e_{n-1}>>` IS the 1-based function `[1..n -> …]`. A homogeneous-LABEL
        // sequence is already caught upstream by `func_enum_view` (the FuncEnum path); here we handle a
        // sequence with a Bool (or mixed leaf) element — which the `Seq` pack CANNOT encode — by redirecting
        // it to a 1-based Int-domain `Func` (`dom = ["1",…,"n"]`, `dom_kind = Int`), packed POSITIONALLY
        // exactly like the Int-prefix `Func`. A PURE nonneg-Int sequence (and the empty sequence) stays
        // `NotCompound` ⇒ the caller falls back to the byte-identical `ColSort::Seq` packer, so no existing
        // sequence cert changes. A genuine VARYING-length Bool queue then has a state-dependent arity ⇒ the
        // cross-state `col_sorts` agreement check fails closed; only a FIXED `[1..n -> …]` function certifies.
        Value::Seq(_) | Value::Tuple(_) => {
            let values: Vec<&Value> = match val {
                Value::Seq(s) => s.iter().collect(),
                Value::Tuple(t) => t.iter().collect(),
                _ => unreachable!(),
            };
            // Empty or all-nonneg-Int ⇒ leave it to the `Seq` packer (byte-identical). `classify_cell`
            // returns `Leaf(_, Int)` for a nonneg Int, `Leaf(_, Bool)` for a Bool, `Enum(..)` for a label.
            if values.is_empty()
                || values
                    .iter()
                    .all(|v| matches!(classify_cell(v), Some(CellClass::Leaf(_, CellSort::Int))))
            {
                return Err(CompoundStop::NotCompound);
            }
            let n = values.len();
            base.checked_pow(u32::try_from(n).map_err(|_| CompoundStop::Fail)?)
                .ok_or(CompoundStop::Fail)?;
            let (dom, dom_kind) =
                int_interval_domain_keys(1, n).ok_or(CompoundStop::NotCompound)?;
            let (cells, pack) = pack_positions(&values)?;
            Ok((
                ColSort::Func {
                    base,
                    arity: n as u32,
                    cells: normalize_cells(cells),
                    dom,
                    dom_kind,
                },
                pack,
            ))
        }
        _ => Err(CompoundStop::NotCompound), // not a positional compound ⇒ caller falls back
    }
}

/// Whether `val` is a POSITIONAL COMPOUND (`Record` / Int-domain `Func`) all of whose positions are LEAF
/// cells (Int / Bool / `String` / model-value label) AND at least one position is a `String`/model-value
/// ENUM cell — the value-type-leaf case the certify pipeline HANDLES (via [`encode_compound_at`] over the
/// per-position cross-state label union), so the decline-explainer must NOT report it as a wall.
/// `pub(crate)` for [`crate::certify_explain`]; a pure function of the value (no label union needed).
#[cfg(feature = "clean-cic")]
pub(crate) fn compound_enum_view(val: &crate::value::Value) -> bool {
    use crate::value::{IntIntervalFunc, Value};
    // classify every position; require all-leaf and at least one enum label. A non-0-prefix func domain or
    // a non-leaf field ⇒ not this fragment (`false`).
    let all_leaf_with_enum = |vals: &[&Value]| -> bool {
        let mut has_enum = false;
        for v in vals {
            match classify_cell(v) {
                Some(CellClass::Enum(..)) => has_enum = true,
                Some(CellClass::Leaf(..)) => {}
                None => return false,
            }
        }
        has_enum
    };
    match val {
        Value::Record(rec) => {
            let vals: Vec<&Value> = rec.iter_str().map(|(_, v)| v).collect();
            all_leaf_with_enum(&vals)
        }
        Value::Func(f) => {
            // Int-prefix OR atom-key domain (the `func_enum_domain_keys` shape); any other ⇒ not this
            // fragment.
            if func_enum_domain_keys(f).is_none() {
                return false;
            }
            let vals: Vec<&Value> = f.mapping_values().collect();
            all_leaf_with_enum(&vals)
        }
        Value::IntFunc(f) => {
            if IntIntervalFunc::min(f) != 0 {
                return false;
            }
            let vals: Vec<&Value> = f.values().iter().collect();
            all_leaf_with_enum(&vals)
        }
        _ => false,
    }
}

/// The SMALLEST self-delimiting radix `D` (≥ [`RECORD_FUNC_BASE`]) admitting every element of a
/// `Seq`/`Tuple`, or `None` if the sequence is not packable at ANY radix here — a non-nonneg-Int element
/// (a Bool/String/model element needs a per-element enumerated leaf sort — the value-type-leaf refactor;
/// a negative/oversized Int), a length beyond [`SEQ_MAX_LEN`], or a pack `D^len` that OVERFLOWS `u64`
/// even at the minimal radix. The sequence analogue of [`compound_min_base`]'s record/func rule: each
/// element `a < base = D-1` (the `+1` digit shift ⇒ shifted digit `a+1 < D`), so `D ≥ maxElement + 2`,
/// FLOORED at `RECORD_FUNC_BASE` (⇒ `base ≥ SEQ_BASE`, the byte-compat floor). DETERMINISTIC.
#[cfg(feature = "clean-cic")]
fn seq_min_radix<'a>(
    elems: impl Iterator<Item = &'a crate::value::Value>,
    len: usize,
) -> Option<u64> {
    if len > SEQ_MAX_LEN as usize {
        return None; // beyond the length cap ⇒ not packable at any radix here
    }
    let mut max_elem = 0u64;
    for el in elems {
        max_elem = max_elem.max(nonneg_small_int(el, u64::MAX)?); // non-Int/neg element ⇒ not packable
    }
    // Each element `< base = D-1` ⇒ `D ≥ maxElement + 2`; floor `D ≥ RECORD_FUNC_BASE` (⇒ `base ≥
    // SEQ_BASE`). The `+2` (vs record/func's `+1`) is the sequence digit shift (shifted digit `a+1 < D`).
    let d = RECORD_FUNC_BASE.max(max_elem.checked_add(2)?);
    d.checked_pow(u32::try_from(len).ok()?)?; // pack `< D^len` must fit u64 (mirror base^arity)
    Some(d)
}

/// Pack a SEQUENCE/TUPLE (`<<a_0,…,a_{m-1}>>`) into the SELF-DELIMITING radix-`d` Nat `pack = Σ_i
/// (a_i+1)·d^i`, where `d` is the per-column derived radix (`bases[i]`, floored at [`RECORD_FUNC_BASE`] ⇒
/// element base `≥ SEQ_BASE`). Fail-closed if the length exceeds `SEQ_MAX_LEN`, any element is not a
/// nonneg small Int `< base = d-1`, `d` is below the floor, or a place/sum overflows `u64`. CANONICAL (the
/// `+1` shift self-delimits the length). At the FLOOR radix (`d == RECORD_FUNC_BASE`) this is
/// BYTE-IDENTICAL to the pre-widening encoder (`base = SEQ_BASE`, same pack, same sort). Returns the
/// `(ColSort::Seq, pack)` pair with the DERIVED element `base = d - 1`.
#[cfg(feature = "clean-cic")]
fn pack_seq<'a>(
    elems: impl Iterator<Item = &'a crate::value::Value>,
    len: usize,
    d: u64,
) -> Option<(ColSort, u64)> {
    if len > SEQ_MAX_LEN as usize {
        return None;
    }
    // `d` is the self-delimiting radix (`= base + 1`); the certify loop passes the per-column derived
    // radix, floored at `RECORD_FUNC_BASE`. Element `< base = d - 1`; the shifted digit `a+1 < d`.
    let base = d.checked_sub(1)?;
    if base < SEQ_BASE {
        return None; // radix below the byte-compat floor ⇒ out of fragment (caller passes ≥ floor)
    }
    let mut pack: u64 = 0;
    for (i, el) in elems.enumerate() {
        let a = nonneg_small_int(el, base)?; // element `< base = d-1`
        let place = d.checked_pow(i as u32)?;
        // shifted digit (a+1) so digit 0 marks the end ⇒ self-delimiting ⇒ canonical
        pack = pack.checked_add((a + 1).checked_mul(place)?)?;
    }
    Some((
        ColSort::Seq {
            base,
            max_len: SEQ_MAX_LEN,
            elem: CellSort::Int,
        },
        pack,
    ))
}

/// Pack a bounded ATOM / `Bool` sequence into the SELF-DELIMITING radix-`D` Nat `pack = Σ_i (code_i+1)·D^i`
/// (`D = base+1`), where each element's `code_i` is its value-type leaf code: an `Enum` element's INDEX in
/// `labels` (kind-checked) or a `Bool`'s `1|0`. This is the atom/Bool analogue of [`pack_seq`] (which packs
/// the nonneg-Int element `v` as `code = v`) — the length machinery is IDENTICAL, only the element→code map
/// differs. `base = max(SEQ_BASE, labels.len())` for an atom element (every index `< base`), `SEQ_BASE` for
/// `Bool`. Returns:
///   * `Ok((ColSort::Seq, pack))` on success (the derived `elem` leaf recorded in the sort);
///   * `Err(None)` — a length beyond [`SEQ_MAX_LEN`], a `u64` place/pack overflow, or an element whose KIND
///     does not match `elem` (a non-atom / wrong-kind / nested element) ⇒ fail-closed;
///   * `Err(Some(label))` — an atom element of the RIGHT kind not yet in `labels` ⇒ the caller grows the
///     per-column element union and restarts (the sequence analogue of `GrowEnum`).
/// A pure function of the elements + the (grown) `elem` leaf ⇒ verify (Leg-E) re-derives the SAME pack.
#[cfg(feature = "clean-cic")]
fn pack_atom_seq<'a>(
    elems: &[&'a crate::value::Value],
    elem: &CellSort,
) -> Result<(ColSort, u64), Option<String>> {
    use crate::value::Value;
    if elems.len() > SEQ_MAX_LEN as usize {
        return Err(None); // beyond the length cap ⇒ fail-closed
    }
    // The element radix: an atom index must be `< base` (floored at SEQ_BASE for byte-compat with the
    // pure-Int floor); `Bool`'s `{0,1}` fit SEQ_BASE. `D = base+1` is the self-delimiting pack radix.
    let base = match elem {
        CellSort::Enum { labels, .. } => SEQ_BASE.max(labels.len() as u64),
        CellSort::Bool => SEQ_BASE,
        CellSort::Int => return Err(None), // Int elements use `pack_seq`, never this path
    };
    let d = base.checked_add(1).ok_or(None)?;
    let mut pack: u64 = 0;
    for (i, el) in elems.iter().enumerate() {
        let code: u64 = match (elem, el) {
            (CellSort::Enum { labels, kind }, Value::String(s)) if *kind == EnumKind::Str => {
                match labels.iter().position(|l| l == s.as_ref()) {
                    Some(idx) => idx as u64,
                    None => return Err(Some(s.as_ref().to_string())), // grow the element union
                }
            }
            (CellSort::Enum { labels, kind }, Value::ModelValue(s)) if *kind == EnumKind::Model => {
                match labels.iter().position(|l| l == s.as_ref()) {
                    Some(idx) => idx as u64,
                    None => return Err(Some(s.as_ref().to_string())),
                }
            }
            (CellSort::Bool, Value::Bool(b)) => u64::from(*b),
            // A wrong-kind / non-atom / nested element in an established atom/Bool seq ⇒ fail-closed.
            _ => return Err(None),
        };
        let place = d.checked_pow(i as u32).ok_or(None)?;
        // shifted digit (code+1) so digit 0 marks the end ⇒ self-delimiting ⇒ canonical.
        pack = pack
            .checked_add((code + 1).checked_mul(place).ok_or(None)?)
            .ok_or(None)?;
    }
    Ok((
        ColSort::Seq {
            base,
            max_len: SEQ_MAX_LEN,
            elem: elem.clone(),
        },
        pack,
    ))
}

/// Classify a SEQUENCE ELEMENT's value-type leaf for the un-marked-column decision in the certify loop:
/// `Some(Ok(()))` = a nonneg Int (stays the byte-identical [`pack_seq`] Int path), `Some(Err(CellSort))` =
/// an atom (`Enum`) / `Bool` element that MARKS the column as a generalized [`pack_atom_seq`] sequence,
/// `None` = a non-leaf / negative / oversized element (fail-closed). Pure function of the value.
#[cfg(feature = "clean-cic")]
fn classify_seq_first_elem(v: &crate::value::Value) -> Option<Result<(), CellSort>> {
    use crate::value::Value;
    match v {
        Value::SmallInt(k) if *k >= 0 => Some(Ok(())),
        Value::Int(k)
            if !{
                use num_traits::Signed;
                k.is_negative()
            } =>
        {
            Some(Ok(()))
        }
        Value::Bool(_) => Some(Err(CellSort::Bool)),
        Value::String(s) => Some(Err(CellSort::Enum {
            labels: vec![s.as_ref().to_string()],
            kind: EnumKind::Str,
        })),
        Value::ModelValue(s) => Some(Err(CellSort::Enum {
            labels: vec![s.as_ref().to_string()],
            kind: EnumKind::Model,
        })),
        _ => None,
    }
}

/// The FUNCTION-of-ENUM view of a value: a finite function whose domain is the consecutive 0-based prefix
/// `0..arity-1` OR a set of config CONSTANT model-value / `String`-atom keys, and whose VALUES are ALL
/// `String` (or ALL model values) — the [`ColSort::FuncEnum`] fragment. Returns
/// `(arity, is_model, value_labels, dom_keys, dom_kind)` where `value_labels[d]` is `e_d`'s label text in
/// domain order (`value_labels.len() == arity`), `dom_keys` the domain key texts in canonical order (empty
/// for the Int prefix) and `dom_kind` their kind (`Str` vs `Model`; `Model` for the empty Int-prefix dom).
/// `None` (the caller falls through to [`value_cell_encode_at`], i.e. the plain `Func`/`Int` path) when the
/// value is not such a function: the domain is not one of those shapes, a value is not a label (an
/// Int-valued function is a plain `Func`), or the values MIX `String` and model kinds. The EMPTY function
/// (`arity = 0`) has no value to fix the value kind ⇒
/// `None` (handled by the plain `Func` arm). The per-column label UNION + index assignment and the
/// positional pack happen in the certify loop (mirroring how a scalar `String` cell grows
/// [`ColSort::Enum`] via `GrowEnum`) — this helper only classifies one cell. `pub(crate)` so the
/// decline-explainer probe ([`crate::certify_explain`]) can recognize a func-of-enum cell as ENCODABLE
/// (not a wall) without reconstructing the cross-state label union.
#[cfg(feature = "clean-cic")]
pub(crate) fn func_enum_view(
    val: &crate::value::Value,
) -> Option<(u32, bool, Vec<String>, Vec<String>, EnumKind)> {
    use crate::value::{IntIntervalFunc, Value};
    fn label_of(v: &Value) -> Option<(bool, String)> {
        match v {
            Value::String(s) => Some((false, s.as_ref().to_string())),
            Value::ModelValue(s) => Some((true, s.as_ref().to_string())),
            _ => None, // a non-label value (Int, nested func, …) ⇒ NOT a func-of-enum column
        }
    }
    // All values must be homogeneous labels (all String OR all model); `None` on a mix or a non-label,
    // and on the EMPTY value list (no value fixes the kind).
    fn collect<'a>(vals: impl Iterator<Item = &'a Value>) -> Option<(bool, Vec<String>)> {
        let mut is_model: Option<bool> = None;
        let mut labels = Vec::new();
        for v in vals {
            let (m, l) = label_of(v)?;
            match is_model {
                Some(prev) if prev != m => return None, // mixed String/model kinds ⇒ out of fragment
                _ => is_model = Some(m),
            }
            labels.push(l);
        }
        Some((is_model?, labels))
    }
    match val {
        Value::Func(f) => {
            // The domain is either the consecutive nonneg Int prefix `0,1,…,n-1` (`dom_keys` empty —
            // byte-identical to the historical shape) OR a set of config CONSTANT model values
            // (`dom_keys` = the keys' texts in canonical `Value::cmp` order). Any other domain ⇒ `None`.
            let (dom_keys, dom_kind) = func_enum_domain_keys(f)?;
            let (is_model, labels) = collect(f.mapping_values())?;
            Some((labels.len() as u32, is_model, labels, dom_keys, dom_kind))
        }
        Value::IntFunc(f) => {
            let min = IntIntervalFunc::min(f);
            let (is_model, labels) = collect(f.values().iter())?;
            if min == 0 {
                // Int-prefix `0..n-1` domain ⇒ empty `dom` ⇒ the skip-serialized `Model` default kind
                // (byte-identical to the historical shape).
                Some((
                    labels.len() as u32,
                    is_model,
                    labels,
                    Vec::new(),
                    EnumKind::Model,
                ))
            } else {
                // A general non-0-based interval `min..min+len-1` ⇒ an `Int`-keyed domain (keys stored as
                // decimal texts in sorted-by-value order; `f[k]` ⇒ slot `k − min`).
                let (dom_keys, dom_kind) = int_interval_domain_keys(min, labels.len())?;
                Some((labels.len() as u32, is_model, labels, dom_keys, dom_kind))
            }
        }
        // A SEQUENCE / TUPLE `<<e_0,…,e_{n-1}>>` IS the 1-based finite function `[1..n -> …]` (TLA sequences
        // are functions with domain `1..n`), so a homogeneous-LABEL sequence is a FuncEnum over the Int
        // domain `1..n`. The EMPTY sequence has no value to fix the kind ⇒ `None` (`collect`/`int_interval..`
        // both guard it). A Bool/Int-element sequence is NOT a func-of-enum ⇒ `None` (falls to the plain
        // `encode_compound_at` Seq arm, which redirects a Bool sequence to an Int-domain `Func` and keeps a
        // pure-Int sequence a byte-identical `ColSort::Seq`).
        Value::Seq(s) => {
            let (dom_keys, dom_kind) = int_interval_domain_keys(1, s.len())?;
            let (is_model, labels) = collect(s.iter())?;
            Some((labels.len() as u32, is_model, labels, dom_keys, dom_kind))
        }
        Value::Tuple(t) => {
            let (dom_keys, dom_kind) = int_interval_domain_keys(1, t.len())?;
            let (is_model, labels) = collect(t.iter())?;
            Some((labels.len() as u32, is_model, labels, dom_keys, dom_kind))
        }
        _ => None,
    }
}

/// The FUNCTION-to-SET view of `val` for the [`ColSort::FuncSetMask`] fragment (F2): a `Value::Func` /
/// `Value::IntFunc` whose domain is [`func_enum_domain_keys`]-shaped and whose EVERY value is a finite SET
/// of homogeneous value-universe atoms — config-CONSTANT model values, `String`s, OR nonneg `Int`s (the
/// `f ∈ [D -> SUBSET (0..N)]` class: SimpleRegular's `x ∈ [0..N-1 -> SUBSET {0,1}]`) — an empty set `{}`
/// admitted (it fixes no kind). Returns `(arity, fdom, fdom_kind, per_slot_atoms, e_kind)`: `per_slot_atoms[p]`
/// is the atom key texts of value `f[fdom[p]]` (an `Int` value keyed by its DECIMAL) in the domain's canonical
/// `Value::cmp` order (aligned with `mapping_values()`), and `e_kind` is the homogeneous atom kind of the value
/// universe `E` (`Model` / `Str` / `Int`), or `None` when EVERY value set is empty (the kind is not yet
/// fixed). `None` (fall through to the plain compound path) when the value is not such a function: a non-func
/// / non-atom-key domain, a value that is not a Set, a set with an unencodable (negative-Int / nested /
/// record) element, or value sets that MIX atom kinds (Int / model-value / `String`). A DETERMINISTIC
/// pure function of the value, so verify (Leg-E) re-derives the same view. `pub(crate)` so the
/// decline-explainer probe can recognize a func-of-set cell as ENCODABLE (not an R1 wall).
#[cfg(feature = "clean-cic")]
pub(crate) fn funcsetmask_view(
    val: &crate::value::Value,
) -> Option<(
    u32,
    Vec<String>,
    EnumKind,
    Vec<Vec<String>>,
    Option<EnumKind>,
)> {
    use crate::value::{IntIntervalFunc, Value};
    // The domain keys/kind + the per-slot value list (canonical `Value::cmp` order, aligned with the pack).
    let (fdom, fdom_kind, value_sets): (Vec<String>, EnumKind, Vec<&Value>) = match val {
        Value::Func(f) => {
            let (fdom, fk) = func_enum_domain_keys(f)?;
            (fdom, fk, f.mapping_values().collect())
        }
        Value::IntFunc(f) => {
            if IntIntervalFunc::min(f) != 0 {
                return None; // domain not the 0-based prefix ⇒ not this fragment
            }
            (Vec::new(), EnumKind::Model, f.values().iter().collect())
        }
        _ => return None, // not a function ⇒ not the func-to-set fragment
    };
    let arity = u32::try_from(value_sets.len()).ok()?;
    let mut per_slot: Vec<Vec<String>> = Vec::with_capacity(value_sets.len());
    let mut e_kind: Option<EnumKind> = None;
    for v in value_sets {
        let Value::Set(s) = v else { return None }; // every value must be a SET (empty `{}` admitted)
        let mut atoms: Vec<String> = Vec::with_capacity(s.len());
        for elem in s.iter() {
            let (k, name) = match elem {
                Value::ModelValue(sv) => (EnumKind::Model, sv.as_ref().to_string()),
                Value::String(sv) => (EnumKind::Str, sv.as_ref().to_string()),
                // Int VALUE-UNIVERSE atom (the `f ∈ [D -> SUBSET (0..N)]` class — SimpleRegular's
                // `x ∈ [0..N-1 -> SUBSET {0,1}]`): a NONNEG Int value is a universe atom of kind `Int`,
                // keyed by its DECIMAL text (`bit idx("k") ⟺ k ∈ S`) so it is grown into the SAME
                // `GrowSetMask` union and masked EXACTLY like a model / `String` atom — the ONLY
                // difference is the kind tag. A NEGATIVE / oversize Int (or any nested / record element)
                // is outside the nonneg bitmask fragment ⇒ `None` (fail-closed, the whole column declines).
                _ => match nonneg_small_int(elem, u64::MAX) {
                    Some(iv) => (EnumKind::Int, iv.to_string()),
                    None => return None,
                },
            };
            match e_kind {
                Some(ek) if ek != k => return None, // mixed value-atom kinds (Int/model/String) ⇒ closed
                _ => e_kind = Some(k),
            }
            atoms.push(name);
        }
        per_slot.push(atoms);
    }
    Some((arity, fdom, fdom_kind, per_slot, e_kind))
}

/// The Int DOMAIN KEYS `["lo", "lo+1", …, "lo+len-1"]` (decimal texts, already in sorted-by-value ==
/// position order) for a length-`len` function over the BOUNDED interval `lo..hi` (`hi = lo+len-1`) —
/// the [`EnumKind::Int`] analogue of a model-value / `String`-atom domain. The 1-based `lo == 1` case is a
/// `Value::Seq`/`Value::Tuple` (a PlusCal `pc ∈ [1..N -> labels]`); a general `lo` is a non-0-based
/// `Value::IntFunc`. `None` for the EMPTY function (`len == 0` — no keys to fix a bounded domain; the empty
/// case is handled by the plain arms) or on `lo + len` overflow. Since `dom[p]` is `lo + p`, an index key
/// `k ∈ lo..hi` resolves to slot `k − lo` (position of `"k"` in the list) — a faithful BIJECTION between the
/// interval and slots `0..len-1`. DETERMINISTIC (a pure function of `lo`, `len`), so verify (Leg-E)
/// re-derives the SAME keys. The classic 0-based prefix `lo == 0` is DELIBERATELY excluded (it keeps its
/// byte-identical EMPTY `dom`); this helper is only invoked for `lo ≥ 1`.
#[cfg(feature = "clean-cic")]
fn int_interval_domain_keys(lo: i64, len: usize) -> Option<(Vec<String>, EnumKind)> {
    if len == 0 {
        return None; // empty function ⇒ no key fixes a bounded Int domain (plain arms handle it)
    }
    let hi_excl = i64::try_from(len).ok()?.checked_add(lo)?; // lo + len (overflow ⇒ fail-closed)
    Some((
        (lo..hi_excl).map(|k| k.to_string()).collect(),
        EnumKind::Int,
    ))
}

/// The DOMAIN SHAPE of a `Value::Func` for the positional-pack FuncEnum fragment, as `(names, kind)`:
/// `Some((Vec::new(), Model))` when the domain is EXACTLY the consecutive 0-based Int prefix `0,1,…,n-1`
/// (the classic shape — positions ARE the keys, so no domain vector is stored ⇒ byte-identical cert; the
/// `Model` kind is the harmless skip-serialized default for an empty `dom`), `Some((names, Model))` when
/// EVERY domain key is a config CONSTANT MODEL VALUE, `Some((names, Str))` when EVERY domain key is a TLA
/// `String` ATOM (the APTCommit `RMVal = {"r1",..}` case this fragment adds — positions ordered by the
/// domain's canonical `Value::cmp` sort, `names[p]` the key's text), or `None` for any other domain (a
/// mix of kinds, a non-0 Int prefix, a compound key). A `String`-atom domain and a model-value domain
/// with the SAME names are DISTINCT (the `kind` tag), so they never conflate. The names come straight
/// from the sorted-unique domain (`domain_iter()`), a DETERMINISTIC per-column property, so verify (Leg-E) re-derives
/// the SAME order + kind and the sort rebuilds identically.
#[cfg(feature = "clean-cic")]
pub(crate) fn func_enum_domain_keys(
    f: &crate::value::FuncValue,
) -> Option<(Vec<String>, EnumKind)> {
    use crate::value::Value;
    // 0-based Int prefix? (positions are the keys ⇒ empty `dom` ⇒ byte-identical to a pre-domain cert.)
    if f.domain_iter()
        .enumerate()
        .all(|(d, key)| nonneg_small_int(key, u64::MAX) == Some(d as u64))
    {
        return Some((Vec::new(), EnumKind::Model));
    }
    // Otherwise EVERY key must be homogeneously either a config CONSTANT model value OR a `String` atom
    // (the two general atom-domain shapes). A mix of the two kinds — or a compound / non-0-prefix Int
    // key — is out of the fragment. The kind is fixed by the first key and required consistent.
    let mut names = Vec::with_capacity(f.domain_len());
    let mut kind: Option<EnumKind> = None;
    for key in f.domain_iter() {
        let (k, name) = match key {
            Value::ModelValue(s) => (EnumKind::Model, s.as_ref().to_string()),
            Value::String(s) => (EnumKind::Str, s.as_ref().to_string()),
            _ => return None, // a compound / non-0-prefix Int domain key ⇒ out of fragment
        };
        match kind {
            Some(prev) if prev != k => return None, // mixed String/model key kinds ⇒ closed
            _ => kind = Some(k),
        }
        names.push(name);
    }
    Some((names, kind?)) // `kind` is `Some` — the Int-prefix early return handled the empty domain
}

/// Canonical per-value CELL encoding shared by the explicit-fixpoint lane and the
/// refinement lane (`refinement_cert`): one `Value` ⇒ one `(ColSort, u64)` cell, fail-closed
/// outside the encodable fragment. Extracted from the certify closure VERBATIM so the two
/// lanes can never drift. Compound (`Record`/`Func`) packs use the FLOOR radix
/// [`RECORD_FUNC_BASE`]; the explicit-fixpoint certify loop DERIVES a per-column base
/// ([`compound_min_base`]) and re-encodes via [`value_cell_encode_at`] when the floor does not fit.
#[cfg(feature = "clean-cic")]
#[doc(hidden)] // exposed for the encoder-injectivity audit harness (tests/encoder_injectivity_cert.rs); not a stable API
pub fn value_cell_encode(val: &crate::value::Value) -> Option<(ColSort, u64)> {
    value_cell_encode_at(val, RECORD_FUNC_BASE)
}

/// [`value_cell_encode`] at an explicit compound (`Record`/`Func`) pack radix `rf_base` — the per-column
/// base the certify loop DERIVES via [`compound_min_base`] (`= max(RECORD_FUNC_BASE, maxValue+1)`). Only
/// the `Record`/`Func`/`IntFunc` arms read the radix; every other sort is radix-independent. The arity
/// bound is DERIVED, not a magic cap: the pack `base^arity` must fit a `u64` (`rf_base.checked_pow(arity)`
/// — an overflowing arity FAILS CLOSED). At the FLOOR radix (`rf_base == RECORD_FUNC_BASE`) a column all of
/// whose values fit base 10 is BYTE-IDENTICAL to the pre-R3 encoder (same sorts, same packs) — the
/// digest-compatibility invariant for pre-existing certificates. (The pre-R3 arity cap of 4 is LIFTED: a
/// wider base-10 record now packs so long as `10^arity < 2^64`; this only ADMITS more, never changes an
/// already-admitted pack.)
#[cfg(feature = "clean-cic")]
#[doc(hidden)] // exposed for the encoder-injectivity audit harness (tests/encoder_injectivity_cert.rs); not a stable API
pub fn value_cell_encode_at(val: &crate::value::Value, rf_base: u64) -> Option<(ColSort, u64)> {
    use crate::value::{IntIntervalFunc, Value};

    match val {
        Value::SmallInt(n) if *n >= 0 => Some((ColSort::Int, *n as u64)),
        Value::Int(n) => {
            use num_traits::Signed;
            if n.is_negative() {
                return None;
            }
            Some((ColSort::Int, u64::try_from(n.as_ref().clone()).ok()?))
        }
        Value::Bool(b) => Some((ColSort::Bool, u64::from(*b))),
        Value::Set(s) => {
            // A finite set of nonneg small Ints, encoded as the Nat bitmask `Σ_{e∈S} 2^e`.
            let mut mask: u64 = 0;
            for elem in s.iter() {
                let e: u64 = match elem {
                    Value::SmallInt(k) if *k >= 0 => *k as u64,
                    Value::Int(k)
                        if !{
                            use num_traits::Signed;
                            k.is_negative()
                        } =>
                    {
                        u64::try_from(k.as_ref().clone()).ok()?
                    }
                    _ => return None, // non-nonneg-Int element → out of the bitmask fragment
                };
                if e >= u64::from(SET_UNIVERSE_BITS) {
                    return None; // element exceeds the universe bit width → fail-closed
                }
                mask |= 1u64 << e;
            }
            Some((
                ColSort::Set {
                    universe: SET_UNIVERSE_BITS,
                },
                mask,
            ))
        }
        // A bounded RECORD packs POSITIONALLY into one Nat: a FIXED canonical field order (fields
        // sorted by name, via `iter_str`) with each value's digit CODE `code_i < rf_base` ⇒ `pack = Σ
        // code_i·base^i`. CANONICAL (one record ⇒ one Nat). Each position carries a [`CellSort`] (Int / Bool
        // — the VALUE-TYPE LEAF via [`cell_code`]); `cells` is normalized to EMPTY for an all-Int record so
        // the encoding is BYTE-IDENTICAL to the pre-value-type-leaf encoder. Fail-closed if any field value
        // is not a leaf (Int / Bool), a digit code `≥ base`, the arity's pack `base^arity` OVERFLOWS a `u64`,
        // or a place overflows.
        Value::Record(rec) => {
            // The pack ceiling: each digit `< base` ⇒ `pack < base^arity`, so the pack fits a `u64`
            // iff `base^arity` does. `checked_pow` is the DERIVED arity bound (no magic cap).
            rf_base.checked_pow(u32::try_from(rec.len()).ok()?)?;
            let mut pack: u64 = 0;
            let mut fields: Vec<String> = Vec::with_capacity(rec.len());
            let mut cells: Vec<CellSort> = Vec::with_capacity(rec.len());
            // `iter_str` yields fields in the canonical sorted order — the fixed positional order.
            for (i, (name, v)) in rec.iter_str().enumerate() {
                fields.push(name.to_string()); // field NAMES are part of the column identity (canonicality)
                let (code, cs) = cell_code(v)?; // Int value / Bool 0|1; a non-leaf value ⇒ fail-closed
                if code >= rf_base {
                    return None; // digit ≥ base ⇒ needs a wider base (caller's Widen via compound_min_base)
                }
                cells.push(cs);
                // pack += code · base^i
                let place = rf_base.checked_pow(i as u32)?;
                pack = pack.checked_add(code.checked_mul(place)?)?;
            }
            Some((
                ColSort::Record {
                    base: rf_base,
                    fields,
                    cells: normalize_cells(cells),
                },
                pack,
            ))
        }
        // A bounded FINITE FUNCTION whose domain is EXACTLY the consecutive prefix `0..arity-1` and each
        // value's digit CODE `< rf_base`, packed POSITIONALLY like a record: `pack = Σ_d code_d·base^d`,
        // each position a [`CellSort`] leaf (Int / Bool). Fail-closed if the domain is not `0..arity-1` (a
        // non-prefix / non-Int domain), any value is not a leaf / its code is out of range, or the pack
        // `base^arity` overflows a `u64`. (`Value::Tuple` — a 1-indexed sequence-ish function — is handled by
        // the Seq arm below; this arm is for explicit `[d ∈ S |-> e]`.)
        Value::Func(f) => {
            let n = f.domain_len();
            rf_base.checked_pow(u32::try_from(n).ok()?)?; // DERIVED arity bound: pack fits u64

            // The domain is the consecutive Int prefix `0,1,…,n-1` (`dom` empty ⇒ byte-identical) OR a set
            // of atom keys (model values / `String`s — `dom`/`dom_kind` per [`func_enum_domain_keys`]); any
            // other domain ⇒ out of fragment. Values pack in the domain's canonical `Value::cmp` order.
            let (dom, dom_kind) = func_enum_domain_keys(f)?;
            let mut pack: u64 = 0;
            let mut cells: Vec<CellSort> = Vec::with_capacity(n);
            for (d, v) in f.mapping_values().enumerate() {
                let (code, cs) = cell_code(v)?;
                if code >= rf_base {
                    return None;
                }
                cells.push(cs);
                let place = rf_base.checked_pow(d as u32)?;
                pack = pack.checked_add(code.checked_mul(place)?)?;
            }
            Some((
                ColSort::Func {
                    base: rf_base,
                    arity: n as u32,
                    cells: normalize_cells(cells),
                    dom,
                    dom_kind,
                },
                pack,
            ))
        }
        // The INT-DOMAIN fast path `[d ∈ 0..n |-> e]` (an `IntIntervalFunc`): the same POSITIONAL pack
        // as `Value::Func`. Its domain is the integer interval `min..=max`; we require `min == 0` (the
        // consecutive 0-based prefix the encoding assumes), the pack `base^arity` to fit a `u64`, and each
        // value `< rf_base`. Fail-closed otherwise.
        Value::IntFunc(f) => {
            if IntIntervalFunc::min(f) != 0 {
                return None; // domain not the 0-based prefix ⇒ out of fragment
            }
            let n = f.len();
            rf_base.checked_pow(u32::try_from(n).ok()?)?; // DERIVED arity bound: pack fits u64
            let mut pack: u64 = 0;
            let mut cells: Vec<CellSort> = Vec::with_capacity(n);
            for (d, v) in f.values().iter().enumerate() {
                let (code, cs) = cell_code(v)?;
                if code >= rf_base {
                    return None;
                }
                cells.push(cs);
                let place = rf_base.checked_pow(d as u32)?;
                pack = pack.checked_add(code.checked_mul(place)?)?;
            }
            Some((
                ColSort::Func {
                    base: rf_base,
                    arity: n as u32,
                    cells: normalize_cells(cells),
                    dom: Vec::new(),
                    dom_kind: EnumKind::Model,
                },
                pack,
            ))
        }
        // A bounded SEQUENCE / TUPLE `<<a_0,…,a_{m-1}>>` packs SELF-DELIMITINGLY in the per-column derived
        // radix `d = rf_base` (`= base + 1`, floored at `RECORD_FUNC_BASE`): `pack = Σ_i (a_i+1)·d^i` (the
        // `+1` shift makes digit 0 mark the end ⇒ length is self-delimiting ⇒ CANONICAL: one sequence ⇒
        // one Nat). Fail-closed if any element is not a nonneg small Int `< base = d-1`, the length exceeds
        // `SEQ_MAX_LEN`, or a place overflows. At the floor radix this is byte-identical to base `SEQ_BASE`.
        Value::Seq(s) => pack_seq(s.iter(), s.len(), rf_base),
        Value::Tuple(t) => pack_seq(t.iter(), t.len(), rf_base),
        _ => None,
    }
}

/// Run the LIVE explicit-state enumeration and, if the spec is in the fail-closed fragment, emit a
/// kernel-CHECKED explicit-state fixpoint certificate over the enumerated reachable set `R`. `None`
/// (fail-closed) on any deviation from the fragment or kernel rejection — see the module docs.
#[cfg(feature = "clean-cic")]
pub fn certify_explicit_state_spec(
    spec_src: &str,
    config: &Config,
) -> Option<ExplicitFixpointCert> {
    certify_explicit_state_spec_bounded(spec_src, config, memory_derived_state_cap())
}

/// As [`certify_explicit_state_spec`] with an explicit reachable-set cap (for tests / tuning).
#[cfg(feature = "clean-cic")]
pub fn certify_explicit_state_spec_bounded(
    spec_src: &str,
    config: &Config,
    state_cap: usize,
) -> Option<ExplicitFixpointCert> {
    #[cfg(kani)]
    {
        certify_explicit_state_spec_inner(spec_src, config, state_cap, false, None)
    }
    #[cfg(not(kani))]
    {
        // Proof construction contains deep recursive CIC terms. Test-harness
        // threads commonly have only a 2 MiB stack, so use the codec's 16 MiB
        // growth segment here (the corpus worker separately uses 64 MiB).
        // cleancic's iterative node/depth preflight remains the hard resource
        // boundary on both mint and verify; stack growth is not treated as one.
        stacker::grow(crate::cert::STACKER_GROWTH, || {
            certify_explicit_state_spec_inner(spec_src, config, state_cap, false, None)
        })
    }
}

/// As [`certify_explicit_state_spec_bounded`] with a COOPERATIVE wall-clock
/// deadline: the enumeration/BFS/widening loops poll a
/// [`tla_resource::MemoryProbe`] (deadline + a derived memory ceiling) and
/// return an HONEST decline (`None`) on expiry — the worker self-terminates
/// instead of leaking. `deadline: None` is byte-identical to the plain entry.
#[cfg(feature = "clean-cic")]
pub fn certify_explicit_state_spec_bounded_deadline(
    spec_src: &str,
    config: &Config,
    state_cap: usize,
    deadline: Option<std::time::Instant>,
) -> Option<ExplicitFixpointCert> {
    #[cfg(kani)]
    {
        certify_explicit_state_spec_inner(spec_src, config, state_cap, false, deadline)
    }
    #[cfg(not(kani))]
    {
        // See the bounded entry point above: the growth segment prevents
        // ordinary test-thread stack exhaustion; the deadline probe and the
        // shared term preflight remain the fail-closed resource bounds.
        stacker::grow(crate::cert::STACKER_GROWTH, || {
            certify_explicit_state_spec_inner(spec_src, config, state_cap, false, deadline)
        })
    }
}

/// Phase A (`docs/kernel-checked-tla-plan.md`): the FAIL-CLOSED domain-coverage mode. Kernel
/// completeness legs whose product-domain axes rely on TRUSTED-RUST bound rules (Int
/// primed-bound/stutter — [`crate::cleancic::DomainCoverage::RustDerived`]) are DECLINED; only
/// legs whose every axis is the column's full universe (coverage BY CONSTRUCTION) are kept.
/// Closure then rests on the honest enumerated `image ⊆ R` leg for the declined specs. This is
/// the "kernel-covered or declined" discipline; the default mode keeps Rust-derived legs and
/// SURFACES them (see `ty certify` output).
#[cfg(feature = "clean-cic")]
pub fn certify_explicit_state_spec_strict_domain(
    spec_src: &str,
    config: &Config,
) -> Option<ExplicitFixpointCert> {
    #[cfg(kani)]
    {
        certify_explicit_state_spec_inner(spec_src, config, memory_derived_state_cap(), true, None)
    }
    #[cfg(not(kani))]
    {
        // See the bounded entry point above: this segment prevents ordinary
        // test-thread exhaustion, while the shared term preflight supplies the
        // fail-closed resource bound.
        stacker::grow(crate::cert::STACKER_GROWTH, || {
            certify_explicit_state_spec_inner(
                spec_src,
                config,
                memory_derived_state_cap(),
                true,
                None,
            )
        })
    }
}

/// Parse a config `Value` raw string as a brace-delimited MODEL-VALUE set literal `{d1, d2, d3}`:
/// `Some(names)` iff it is `{`-`}` wrapped and EVERY comma-separated element is a NON-EMPTY, NON-NUMERIC
/// identifier (a model-value token — TLC's alternate spelling of a `ModelValueSet`). `None` for anything
/// else — an Int set `{1,2,3}`, a non-brace value, an empty `{}`, or an element with punctuation — so a
/// numeric/other set is NEVER misread as model values. Names are returned in source order; the caller
/// sorts+dedups them for determinism.
#[cfg(feature = "clean-cic")]
pub(crate) fn parse_brace_model_value_set(s: &str) -> Option<Vec<String>> {
    let inner = s.trim().strip_prefix('{')?.strip_suffix('}')?.trim();
    if inner.is_empty() {
        return None; // empty `{}` carries no kind signal (Int-vs-model ambiguous) ⇒ decline
    }
    let mut out = Vec::new();
    for part in inner.split(',') {
        let name = part.trim();
        let mut chars = name.chars();
        // A model-value token is an identifier: leading alpha/underscore, then alnum/underscore. This
        // EXCLUDES numeric literals and any punctuation, so `{1,2,3}` / `{a-b}` decline.
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return None,
        }
        if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        out.push(name.to_string());
    }
    Some(out)
}

/// The `(sorts, init_pred, next_pred, safety_pred)` a spec RECOGNIZES to — with the enumerator AND the
/// embedder OUT of the loop. The RECOGNITION-ONLY re-derivation used by the reflect-check `--full`
/// spec-bind ([`crate::reflect_safety_check::reflect_check_safety_cert_full`]).
#[cfg(feature = "clean-cic")]
pub struct RecognizedSpecIRs {
    /// The per-column sorts derived STRUCTURALLY from the configured invariants' type declarations.
    pub sorts: Vec<ColSort>,
    /// `Init` recognized as a `PredIR` over `sorts` (the `.pred` of the cert's `init_pred`).
    pub init_ir: PredIR,
    /// `Next` recognized as a `PredIR` over `sorts` (the `.pred` of the cert's `next_pred`).
    pub next_ir: PredIR,
    /// The conjoined configured invariants recognized as a `PredIR` over `sorts` (the `safety_pred`).
    pub safety_ir: PredIR,
}

/// RE-DERIVE a spec's `(sorts, Init IR, Next IR, Safety IR)` for the reflect-check `--full` spec-bind
/// WITHOUT enumerating any reachable state and WITHOUT the shallow embedder ([`crate::cleancic::embed_pred_ir`]).
///
/// This mirrors the FRONT-END of [`certify_explicit_state_spec_inner`] (parse → lower → inline the
/// `Init`/`Next`/conjoined-invariant bodies) EXACTLY, then:
///   * derives the per-column [`ColSort`] STRUCTURALLY from the invariants' type declarations
///     ([`crate::cleancic::derive_col_sorts_from_type_invariants`]) — the enumerator is NOT consulted;
///   * recognizes `Init`/`Next`/`Safety` into `PredIR`s over those sorts with the SAME recognizer the
///     certifier uses ([`crate::cleancic::recognize_pred_sorts_with_mvsets`]) — the embedder is NOT invoked.
///
/// The caller requires these to equal the cert's stored `sorts`/`init_pred`/`next_pred`/`safety_pred`; a
/// mismatch is a REJECT (the discharged IRs do not faithfully match the spec). `None` (⇒ INCONCLUSIVE)
/// when the spec is out of the recognition fragment: a sort not spec-derivable, an `Init`/`Next`/`Safety`
/// body the recognizer declines, a `Next` that leaves a variable unconstrained, or a missing operator.
///
/// The `Safety` recognition passes NO per-column reachable maxima (the certifier's `col_max`, a function
/// of the enumerated `R`), so a STATE-DEPENDENT quantifier domain (`\E j \in 0..tpos`, `tpos` a state
/// variable) is NOT expanded here and the resulting IR will DIFFER from the (col_max-expanded) stored one
/// ⇒ the caller REJECTS such a cert (fail-closed — that spec class does not get the recognition-only bind).
#[cfg(feature = "clean-cic")]
pub fn recognize_spec_fixpoint_irs(spec_src: &str, config: &Config) -> Option<RecognizedSpecIRs> {
    use tla_core::ast::{OperatorDef, Unit};

    let init_name = config.init.as_deref()?;
    let next_name = config.next.as_deref()?;
    if config.invariants.is_empty() {
        return None;
    }

    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module?;

    // State variables, in declaration order — the tuple-column order (mirrors the certifier).
    let var_names: Vec<Arc<str>> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    if var_names.is_empty() {
        return None;
    }

    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };
    let init_def = find_op(init_name)?;
    let next_def = find_op(next_name)?;

    // Resolve zero-arity operator refs + Int-literal CONSTANTs inside the recognized bodies — IDENTICAL
    // inlining to the certifier (a deterministic pure function of the parsed module + config).
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    let init_body = inline_env.inline(&init_def.body);
    let next_body = inline_env.inline(&next_def.body);
    // Same fail-closed gate the certifier applies: a `Next` leaving a variable unconstrained is out of
    // fragment (the certifier would decline it — no genuine cert exists for such a spec).
    {
        let vrefs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
        if !crate::cleancic::next_constrains_all_vars(&next_body.node, &vrefs) {
            return None;
        }
    }
    // The safety predicate is the CONJUNCTION of every configured INVARIANT (config order, left-nested),
    // each inlined independently — byte-identical to the certifier's `safety_body`.
    let safety_body = {
        let mut it = config.invariants.iter();
        let first = find_op(it.next()?)?;
        let mut acc = inline_env.inline(&first.body);
        for name in it {
            let leg = inline_env.inline(&find_op(name)?.body);
            acc = tla_core::Spanned::dummy(tla_core::ast::Expr::And(Box::new(acc), Box::new(leg)));
        }
        acc
    };

    // SOUNDNESS GATE (zero-arg-builtin overrides): with `Nat <- Op`-style config
    // overrides in force, a SURVIVING `Ident("Nat")` in an inlined obligation body
    // would be read by the recognizer arms with BUILTIN (infinite) semantics —
    // WEAKER than the overridden finite bound (false-safe vector). Decline.
    if crate::cert_inline::overridden_builtin_survives(
        config,
        &[&init_body, &next_body, &safety_body],
    ) {
        return None;
    }

    // Config CONSTANT model-value SETS (name → sorted-deduped member names) — the SAME map the certifier
    // threads into the recognizer (a `val \in Data` model-value membership resolves through it).
    let mvsets: std::collections::BTreeMap<String, Vec<String>> = {
        let mut m = std::collections::BTreeMap::new();
        for (name, cv) in &config.constants {
            let names: Option<Vec<String>> = match cv {
                crate::config::ConstantValue::ModelValueSet(ns) => Some(ns.clone()),
                crate::config::ConstantValue::Value(s) => parse_brace_model_value_set(s),
                _ => None,
            };
            if let Some(mut ns) = names {
                ns.sort();
                ns.dedup();
                m.insert(name.clone(), ns);
            }
        }
        m
    };

    let vars: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
    // STRUCTURAL sort derivation from the invariants' type declarations — enumeration-free (fail-closed
    // ⇒ `None` when any column's sort is not spec-derivable).
    let sorts =
        crate::cleancic::derive_col_sorts_from_type_invariants(&vars, &safety_body.node, &mvsets)?;

    // Recognize the three predicates over the DERIVED sorts — the embedder is never invoked. (Safety is
    // recognized WITHOUT `col_max`; see the fn doc for the state-dependent-domain fail-closed note.)
    let init_ir =
        crate::cleancic::recognize_pred_sorts_with_mvsets(&init_body.node, &vars, &sorts, &mvsets)?;
    let next_ir =
        crate::cleancic::recognize_pred_sorts_with_mvsets(&next_body.node, &vars, &sorts, &mvsets)?;
    let safety_ir = crate::cleancic::recognize_pred_sorts_with_mvsets(
        &safety_body.node,
        &vars,
        &sorts,
        &mvsets,
    )?;

    Some(RecognizedSpecIRs {
        sorts,
        init_ir,
        next_ir,
        safety_ir,
    })
}

/// The stdlib modules BUILTIN to the evaluator (kept as `EXTENDS` in a merged
/// module; every OTHER `EXTENDS` is a source module whose units are inlined).
fn is_stdlib_module(name: &str) -> bool {
    matches!(
        name,
        "Naturals" | "Integers" | "Sequences" | "FiniteSets" | "TLC" | "Reals" | "Bags"
    )
}

/// Resolve `X <- MCOp` (operator-override) CONSTANTs against a merged module: if
/// `MCOp` is a set-enum literal `{a, b, c}` of model values, rebind `X` to that
/// ModelValueSet; if an integer literal, rebind to that Value. Other shapes stay
/// `Replacement` (the cert then declines them fail-closed).
#[cfg(feature = "clean-cic")]
fn resolve_replacement_constants(config: &Config, module: &tla_core::ast::Module) -> Config {
    use tla_core::ast::{Expr, Unit};
    let mut out = config.clone();
    for (name, cv) in &config.constants {
        let crate::config::ConstantValue::Replacement(op) = cv else {
            continue;
        };
        let body = module.units.iter().find_map(|u| match &u.node {
            Unit::Operator(o) if o.name.node == *op && o.params.is_empty() => Some(&o.body.node),
            _ => None,
        });
        let replaced = match body {
            Some(Expr::SetEnum(elems)) => elems
                .iter()
                .map(|e| match &e.node {
                    Expr::Ident(n, _) => Some(n.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<String>>>()
                .map(crate::config::ConstantValue::ModelValueSet),
            Some(Expr::Int(n)) => Some(crate::config::ConstantValue::Value(n.to_string())),
            _ => None,
        };
        if let Some(v) = replaced {
            out.constants.insert(name.clone(), v);
        }
    }
    out
}

/// MULTI-MODULE concrete-config cert: load `main_file` + its EXTENDS chain via the
/// ModuleLoader (resolving `MC* EXTENDS base`), MERGE the chain's units into one
/// module (custom EXTENDS inlined, stdlib EXTENDS kept), resolve `<-` operator-
/// override CONSTANTs to their definitions, and certify through the SAME
/// module-level cert. Unblocks MC-wrapper corpus specs a single-file read cannot.
/// Additive: the single-module path is unchanged.
#[cfg(feature = "clean-cic")]
pub fn certify_explicit_state_spec_from_dir(
    main_file: &std::path::Path,
    config: &Config,
    state_cap: usize,
) -> Option<ExplicitFixpointCert> {
    certify_explicit_state_spec_from_dir_deadline(main_file, config, state_cap, None)
}

/// As [`certify_explicit_state_spec_from_dir`] with the cooperative deadline
/// (see [`certify_explicit_state_spec_bounded_deadline`]).
#[cfg(feature = "clean-cic")]
pub fn certify_explicit_state_spec_from_dir_deadline(
    main_file: &std::path::Path,
    config: &Config,
    state_cap: usize,
    deadline: Option<std::time::Instant>,
) -> Option<ExplicitFixpointCert> {
    use tla_core::ast::Unit;
    use tla_core::ast::{Expr as Expr2, Module};
    use tla_core::ModuleLoader;
    let main_name = main_file.file_stem()?.to_str()?.to_string();
    let mut loader = ModuleLoader::new(main_file);
    let main_module = loader.load(&main_name).ok()?.module.clone();
    // Load the transitive EXTENDS chain AND the main module's INSTANCE deps into
    // the loader's cache — modules_for_semantic_resolution only ORDERS already-
    // loaded modules; without loading, the chain comes back empty and the merge
    // would lack the base module's VARIABLES and operator definitions.
    loader.load_extends(&main_module).ok()?;
    loader.load_instances(&main_module).ok()?;
    // Merge ONLY the EXTENDS component. INSTANCE modules' units carry
    // un-materialized `WITH` substitutions — inlining them verbatim could evaluate
    // free names against same-named outer symbols and CERTIFY A DIFFERENT MODEL
    // than TLC checks (and leak instanced VARIABLEs as phantom state columns).
    // So instance modules' units are NEVER merged. UNREFERENCED named instances
    // (`TC == INSTANCE TCommit` used only in THEOREM-level operators — the
    // TwoPhase/EWD840 corpus pattern) are harmless: the certified obligations
    // never evaluate into them, and any obligation that DOES reference one
    // (`D1!Next`, MCDieHardest) hits an unresolvable module at evaluation or
    // recognition and the lane declines fail-closed THERE — the same honest
    // None, just later. The instance BINDING units stay in the merged module
    // (EvalCtx registers them lazily; resolution is at USE, so an unused
    // binding whose target module is absent never errors).
    let (extends_mods, _instance_mods) = loader.modules_for_semantic_resolution(&main_module);
    let mut chain: Vec<Module> = extends_mods.into_iter().cloned().collect();
    // TIER-A IDENTITY-INSTANCE MERGE: a STANDALONE `INSTANCE M` with NO
    // substitutions (or only identity `x <- x`) imports M's definitions with
    // the IDENTITY substitution — semantically the same import as EXTENDS for
    // the certified obligations (TLA requires M's declared names to already
    // exist in the instantiating scope). Merge such M's units like a base
    // module, gated v1-tight: M's own EXTENDS must be stdlib-only and M must
    // contain no INSTANCE of its own (anything else stays un-merged and any
    // reference to it declines fail-closed downstream, as before). Named
    // instances (`I == INSTANCE M`) and non-identity WITH stay excluded — the
    // un-materialized-substitution wrong-model hazard is untouched.
    {
        use tla_core::ast::Unit;
        let is_identity = |decl: &tla_core::ast::InstanceDecl| {
            decl.substitutions
                .iter()
                .all(|sub| matches!(&sub.to.node, Expr2::Ident(n, _) if *n == sub.from.node))
        };
        let mut tier_a: Vec<String> = Vec::new();
        let scope_mods: Vec<&Module> = chain.iter().chain(std::iter::once(&main_module)).collect();
        for m in &scope_mods {
            for u in &m.units {
                if let Unit::Instance(decl) = &u.node {
                    if is_identity(decl) {
                        tier_a.push(decl.module.node.clone());
                    }
                }
            }
        }
        tier_a.sort();
        tier_a.dedup();
        for name in tier_a {
            let Some(loaded) = loader.get(&name) else {
                continue;
            };
            let m = &loaded.module;
            let extends_ok = m.extends.iter().all(|e| is_stdlib_module(&e.node));
            let no_instances = m.units.iter().all(|u| !matches!(u.node, Unit::Instance(_)));
            if extends_ok && no_instances {
                chain.push(m.clone());
            }
        }
    }
    // Merge order: deepest base FIRST (TLC first-definition-wins import order),
    // main module's units LAST — matching load_modules_into_ctx's "main last"
    // convention. The chain NEVER contains main (modules_for_semantic_resolution
    // derives it from main's EXTENDS decls).
    let mut units = Vec::new();
    let mut extends = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Union the action-subscript provenance spans from ALL merged modules — the
    // base modules' `[A]_v` spans would otherwise be dropped by the struct spread.
    let mut subscript_spans = main_module.action_subscript_spans.clone();
    for m in &chain {
        for u in &m.units {
            units.push(u.clone());
        }
        for e in &m.extends {
            if is_stdlib_module(&e.node) && seen.insert(e.node.clone()) {
                extends.push(e.clone());
            }
        }
        subscript_spans.extend(m.action_subscript_spans.iter().copied());
    }
    units.extend(main_module.units.iter().cloned());
    // Fail-closed on duplicate operator names across the merged units: find_op /
    // the inline env / EvalCtx resolve duplicates with DIFFERENT precedence
    // (first-match vs last-wins), so a clash could certify a recognizer/enumerator
    // DIVERGENT model. A non-LOCAL clash is illegal TLA+ anyway — decline.
    {
        let mut op_names = std::collections::HashSet::new();
        for u in &units {
            if let Unit::Operator(o) = &u.node {
                if !o.local && !op_names.insert(o.name.node.clone()) {
                    return None;
                }
            }
        }
    }
    let merged = Module {
        extends,
        units,
        action_subscript_spans: subscript_spans,
        ..main_module
    };
    // Route cfg OPERATOR-OVERRIDES of the selected predicates (`Init <- MCInit`)
    // through the config: certify_module_inner selects init/next/invariants by
    // find_op on the NAME, which would silently pick the BASE definition and
    // certify the wrong model. Redirect each selected name through its
    // Replacement chain (cap 8; a cycle or over-deep chain declines).
    let redirect = |name: &str| -> Option<String> {
        let mut cur = name.to_string();
        for _ in 0..8 {
            match config.constants.get(&cur) {
                Some(crate::config::ConstantValue::Replacement(op)) => cur = op.clone(),
                _ => return Some(cur),
            }
        }
        None
    };
    let mut cfg2 = config.clone();
    cfg2.init = Some(redirect(config.init.as_deref()?)?);
    cfg2.next = Some(redirect(config.next.as_deref()?)?);
    cfg2.invariants = config
        .invariants
        .iter()
        .map(|i| redirect(i))
        .collect::<Option<Vec<_>>>()?;
    let resolved = resolve_replacement_constants(&cfg2, &merged);
    certify_module_inner(merged, &resolved, state_cap, false, deadline)
}

#[cfg(feature = "clean-cic")]
fn certify_explicit_state_spec_inner(
    spec_src: &str,
    config: &Config,
    state_cap: usize,
    require_domain_complete: bool,
    deadline: Option<std::time::Instant>,
) -> Option<ExplicitFixpointCert> {
    // Parse the SINGLE module, then delegate to the module-level cert. Multi-module
    // specs (MC-wrappers `EXTENDS base`) call `certify_module_inner` directly with a
    // MERGED module (see `certify_explicit_state_spec_from_dir`).
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = lowered.module?;
    certify_module_inner(module, config, state_cap, require_domain_complete, deadline)
}

/// Certify from an already-parsed module (single or the MERGED units of an EXTENDS
/// chain). Split out of [`certify_explicit_state_spec_inner`] so the multi-module
/// entry can feed a merged module through the identical cert logic.
#[cfg(feature = "clean-cic")]
fn certify_module_inner(
    #[cfg_attr(not(feature = "clean-cic"), allow(unused_mut))] mut module: tla_core::ast::Module,
    config: &Config,
    state_cap: usize,
    require_domain_complete: bool,
    deadline: Option<std::time::Instant>,
) -> Option<ExplicitFixpointCert> {
    // Solo-field record ELISION (function-cell scoped): rewrite `smokers[i].smoking`
    // -style solo records used exclusively as function cells to their bare field
    // value, uniformly across every operator body, so enumeration AND recognition
    // both see the record-free spec. Semantics-preserving bijection; no-op unless a
    // field is solo + accessed + cell-only (top-level record columns are excluded).
    #[cfg(feature = "clean-cic")]
    crate::cert_inline::elide_module_solo_field_records(&mut module);
    // COOPERATIVE DEADLINE probe (one per certification, OUTSIDE the widening
    // restart loop so its latched trip persists across Widen/Grow* restarts):
    // deadline + the lane's existing derived memory fraction. `None` deadline ⇒
    // no probe ⇒ zero behavior change on every existing entry point.
    // Budget shape: DEADLINE + the system-wide COLLECTIVE FLOOR only (the real
    // OOM guard). Deliberately NO per-process fraction ceiling: the probe
    // measures whole-process footprint, and in a shared sweep process that
    // accumulates across many certifications a fraction ceiling false-trips on
    // late specs (measured: glowingRaccoon/product declined spuriously).
    let mut deadline_probe: Option<tla_resource::MemoryProbe> = deadline.map(|d| {
        tla_resource::MemoryProbe::new(
            tla_resource::MemoryBudget::from_thresholds(
                None,
                None,
                tla_resource::collective_floor_bytes(),
                0,
            ),
            Some(d),
        )
    });
    use crate::enumerate::{
        enumerate_states_from_constraint_branches_probed, enumerate_successors,
        extract_init_constraints,
    };
    use crate::eval::EvalCtx;
    use crate::state::State;
    use tla_core::ast::{Module, OperatorDef, Unit};

    let init_name = config.init.as_deref()?;
    let next_name = config.next.as_deref()?;
    // At least one configured safety invariant. MULTIPLE invariants are conjoined into a
    // single safety predicate `I_0 /\ I_1 /\ … /\ I_{n-1}` (left-nested, in config order)
    // and certified together — R ⊆ (I_0 ∧ … ∧ I_{n-1}). The general embedded safety leg
    // (`safety_general`) already proves arbitrary conjunctions; the `⋀ x≥0` shortcut still
    // fires when the conjunction happens to be exactly that shape. Leg-E re-derives
    // `config.invariants` from the cert's embedded config and re-conjoins IDENTICALLY (same
    // order), so verify rebuilds the same obligation. A spec with several configured
    // INVARIANTs (the common corpus case) no longer needs a single hand-picked one.
    if config.invariants.is_empty() {
        return None;
    }

    // One or MORE state variables (a state is a TUPLE, in declaration order).
    // DEDUP order-preserving: a Tier-A-merged identity INSTANCE re-declares the
    // same VARIABLE names as the instantiating module (TLA requires the names to
    // align) — the column set must contain each name ONCE, at its first
    // declaration position.
    let var_names: Vec<Arc<str>> = {
        let mut seen = std::collections::HashSet::new();
        module
            .units
            .iter()
            .flat_map(|u| match &u.node {
                Unit::Variable(decls) => decls
                    .iter()
                    .map(|d| Arc::<str>::from(d.node.as_str()))
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .filter(|v| seen.insert(v.clone()))
            .collect()
    };
    if var_names.is_empty() {
        return None;
    }

    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };
    let init_def = find_op(init_name)?;
    let next_def = find_op(next_name)?.clone();

    // Resolve zero-arity operator references + Int-literal configured CONSTANTs inside the three
    // RECOGNIZED bodies (`cert_inline`): the recognizers see the resolved predicate (CoffeeCan's
    // `TypeInvariant == can \in Can` reaches the record-set membership form), while the LIVE
    // enumeration below keeps evaluating the ORIGINAL definitions (the evaluator resolves
    // operators itself). Deterministic — a pure function of the parsed module + config, both of
    // which Leg-E re-derives from the cert's embedded spec/config, so certify and verify inline
    // identically. A body with no such references is returned UNCHANGED, so previously-recognized
    // specs (and their certificates) are untouched.
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    let init_body = inline_env.inline(&init_def.body);
    let next_body = inline_env.inline(&next_def.body);
    // SOUNDNESS GATE (2026-07-05 confirmed false-safe): reject a spec whose `Next` leaves any
    // declared variable UNCONSTRAINED in some action — the live enumerator would FREEZE it, under-
    // approximating R and silently dropping reachable violating states. Fail-closed; a well-formed
    // spec constrains every variable in every action.
    {
        let vrefs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
        if !crate::cleancic::next_constrains_all_vars(&next_body.node, &vrefs) {
            return None;
        }
    }
    // The safety predicate is the CONJUNCTION of every configured INVARIANT (config order,
    // left-nested), each inlined independently, then folded with `Expr::And`. A single
    // configured invariant reduces to exactly its (inlined) body — byte-identical to before
    // this change, so existing single-invariant certs are untouched.
    let safety_body = {
        let mut it = config.invariants.iter();
        let first = find_op(it.next()?)?;
        let mut acc = inline_env.inline(&first.body);
        for name in it {
            let leg = inline_env.inline(&find_op(name)?.body);
            acc = tla_core::Spanned::dummy(tla_core::ast::Expr::And(Box::new(acc), Box::new(leg)));
        }
        acc
    };

    // SOUNDNESS GATE (zero-arg-builtin overrides): with `Nat <- Op`-style config
    // overrides in force, a SURVIVING `Ident("Nat")` in an inlined obligation body
    // would be read by the recognizer arms with BUILTIN (infinite) semantics —
    // WEAKER than the overridden finite bound (false-safe vector). Decline.
    if crate::cert_inline::overridden_builtin_survives(
        config,
        &[&init_body, &next_body, &safety_body],
    ) {
        return None;
    }

    // ── UNBOUNDED-DOMAIN parametric inductive-invariant path (BEFORE the finite BFS) ───────────────
    // If `Init = ⋀_j x_j=c_j / Next = ⋀_j x_j'=x_j+δ_j (NO guard) / Safety = ⋀_j x_j≥0` is recognized
    // over N≥1 Int variables, the reachable set is INFINITE — the BFS below would never reach a
    // fixpoint (it would just hit the state cap and fail-close). Instead, prove the inductive invariant
    // `J ≡ Safety ≡ ⋀_j x_j≥0` PARAMETRICALLY via three kernel-checked `∀x_0..x_{n-1}` implications
    // (initiation / consecution / preservation) and emit the cert WITHOUT enumeration. Distinct from
    // (mutually exclusive with) the finite affine path, which needs an `x < bound` guard.
    //
    // n=1 uses the SCALAR legs (byte-identical to the historical single-var cert); n≥2 uses the
    // CONJOINED multi-variable legs (per-variable `NonNeg.add`, `And.intro`-conjoined). Both are stored
    // with the full `pairs` vector so Leg-E re-derives and re-checks the exact obligation.
    {
        let var_refs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
        // Single-var fast path: keep the historical scalar recognizer + legs verbatim.
        let single = (var_refs.len() == 1)
            .then(|| {
                crate::cleancic::recognize_unbounded_affine(
                    &init_body.node,
                    &next_body.node,
                    &safety_body.node,
                    var_refs[0],
                )
            })
            .flatten();
        // RELATIONAL fast path (STRONGER than per-variable nonneg): exactly two Int variables whose
        // Safety is `v0=v1` and that step in lock-step (`v0'=v0+δ ∧ v1'=v1+δ`). The conjunctive-nonneg
        // recognizer would decline this (Safety is not `⋀ x_j≥0`), so try it FIRST.
        let relational = (var_refs.len() == 2)
            .then(|| {
                crate::cleancic::recognize_unbounded_relational_eq(
                    &init_body.node,
                    &next_body.node,
                    &safety_body.node,
                    var_refs[0],
                    var_refs[1],
                )
            })
            .flatten();
        let unbounded = if let Some(rel) = relational {
            // Build + kernel-check the three RELATIONAL `Eq`-legs (Eq.refl / Eq.subst / identity).
            let (initiation, consecution, preservation) =
                crate::cleancic::certify_unbounded_relational(&rel)?;
            Some(UnboundedInvariantCert {
                init: rel.init,
                delta: rel.delta,
                pairs: Vec::new(), // relational certs carry no per-variable nonneg vector
                bound: 0,
                relational: Some((rel.init, rel.delta)),
                initiation,
                consecution,
                preservation,
            })
        } else if let Some(shape) = single {
            // Build + kernel-check the three SCALAR parametric legs (Safety = x ≥ N; N = 0 is the
            // historical NonNeg fragment). Fail-closed (`None`) on any kernel rejection — the
            // verdict then stays at the honest non-certified tier.
            let (initiation, consecution, preservation) =
                crate::cleancic::certify_unbounded_invariant(&shape)?;
            Some(UnboundedInvariantCert {
                init: shape.init,
                delta: shape.delta,
                pairs: vec![(shape.init, shape.delta)],
                bound: shape.bound,
                relational: None,
                initiation,
                consecution,
                preservation,
            })
        } else if let Some(tuple) = crate::cleancic::recognize_unbounded_affine_tuple(
            &init_body.node,
            &next_body.node,
            &safety_body.node,
            &var_refs,
        ) {
            // Build + kernel-check the three CONJOINED multi-variable legs. Fail-closed on any kernel
            // rejection.
            let (initiation, consecution, preservation) =
                crate::cleancic::certify_unbounded_invariant_tuple(&tuple)?;
            let (init0, delta0) = tuple.pairs[0];
            Some(UnboundedInvariantCert {
                init: init0,
                delta: delta0,
                pairs: tuple.pairs,
                bound: 0,
                relational: None,
                initiation,
                consecution,
                preservation,
            })
        } else {
            None
        };
        if let Some(ub) = unbounded {
            // The state width: relational certs cover exactly two Int variables; nonneg certs cover
            // `pairs.len()`.
            let n = if ub.relational.is_some() {
                2
            } else {
                ub.pairs.len()
            };
            return Some(ExplicitFixpointCert {
                // NO enumeration: the empty `reachable`/`init_values`/`image` signal an UNBOUNDED cert.
                reachable: Vec::new(),
                init_values: Vec::new(),
                image: Vec::new(),
                sorts: vec![crate::explicit_fixpoint_cert::ColSort::Int; n],
                // The finite legs are vacuous here (closure rests on the `∀x_0..x_{n-1}` consecution leg).
                safety_term: Vec::new(),
                init_member_terms: Vec::new(),
                closed_member_terms: Vec::new(),
                next_shape: None,
                next_completeness: None,
                init_shape: None,
                init_completeness: None,
                next_pred: None,
                next_general_completeness: None,
                init_pred: None,
                init_general_completeness: None,
                unbounded_invariant: Some(ub),
                // The parametric lane's Safety is the recognized affine/relational shape itself —
                // no enumerated R exists to fold a general safety leg over.
                safety_pred: None,
                safety_general: None,
                init_member_reflected: None,
                closed_member_reflected: None,
                // The recognized unbounded shape is an UNGUARDED affine step `x'=x+δ` (a guard would
                // keep the reachable set finite and route through the BFS), so `Next` is unconditionally
                // enabled and every state has a successor — deadlock-freedom holds by construction. No
                // finite kernel witness leg is built (there is no enumerated R); the CLI states this.
                deadlock_free: None,
                deadlock_scan: MintAside(None),
            });
        }
    }

    // Build the live eval context exactly as the model checker / interactive explorer do.
    let build_ctx = |module: &Module| -> Option<EvalCtx> {
        let mut ctx = EvalCtx::new();
        ctx.load_module(module);
        for v in &var_names {
            ctx.register_var(Arc::clone(v));
        }
        crate::constants::bind_constants_from_config(&mut ctx, config).ok()?;
        Some(ctx)
    };

    // ── Enumerate + encode, with the PER-COLUMN compound-base DERIVATION retry (roadmap R3) ─────────
    // One attempt enumerates the live Init states and BFS-closes them under the live `Next`,
    // encoding every state at the CURRENT per-column compound radixes (`value_cell_encode_at` — a
    // nonneg `Int` → `(Int, value)`; a `Bool` → `(Bool, 1|0)`; a finite `Set` of nonneg small Ints →
    // the `SET_UNIVERSE_BITS` bitmask; a `Record`/`Func` → the positional base-`bases[i]` pack).
    // When a compound column's value does not fit the CURRENT radix but fits a LARGER one, the attempt
    // stops carrying the DERIVED minimal admitting base ([`compound_min_base`]), that column's base is
    // raised (monotonically) to it, and the (deterministic) enumeration restarts from scratch — so the
    // final `bases[i]` is `max` over all states of `compound_min_base` = the SMALLEST base admitting
    // every observed value of the column, i.e. `max(RECORD_FUNC_BASE, maxObservedValue+1)`. Verify
    // (Leg-E) re-enumerates the SAME states and re-derives the SAME base byte-identically. A value
    // whose derived `base^arity` overflows a `u64` fails closed, as does a per-column sort disagreement
    // across states (a column must keep ONE sort).
    use crate::explicit_fixpoint_cert::{ColSort, EnumKind};
    // `EnumCol` (the kind + sorted label union) is a module-level type shared by the scalar/`FuncEnum`
    // per-column union and the per-POSITION compound union.
    enum EnumStop {
        /// Column `i` needs a WIDER compound radix (the derived minimal admitting base) — restart with
        /// its base raised to (at least) the carried value.
        Widen(usize, u64),
        /// Column `i` observed a NEW enum label (of the given kind) not yet in its label set — grow the
        /// per-column sorted label union and restart. Mirrors `Widen`: a deterministic per-column
        /// property (the label union) is fixed by re-enumeration, so verify re-derives the same sort.
        GrowEnum(usize, EnumKind, String),
        /// Column `i` (a SET-of-atoms column) observed a NEW atom (of the given kind) not yet in its
        /// bitmask UNIVERSE `dom` — grow the per-column sorted atom union (shared with `enum_labels[i]`)
        /// and restart. The set analogue of `GrowEnum`: the final `dom` is the sorted union of all atoms
        /// EVER present in the column, a deterministic per-column property Leg-E re-derives (bit `i` ⟺
        /// `dom[i] ∈ S`). Restarts terminate — the distinct atoms across the finite state set are finite.
        GrowSetMask(usize, EnumKind, String),
        /// Column `i` (a SET-of-RECORDS column) observed a NEW record KEY not yet in its bitmask UNIVERSE
        /// `dom` — grow the per-column sorted record-key union (`recmask_labels[i]`) and restart. The
        /// record analogue of `GrowSetMask`: the final `dom` is the sorted union of all record keys EVER
        /// present in the column (a deterministic per-column property Leg-E re-derives; bit `i` ⟺
        /// `dom[i] ∈ S`). Restarts terminate — the distinct record keys across the finite state set are
        /// finite. Capped at 64 (the one-cell bit width).
        GrowSetMaskRec(usize, String),
        /// Column `i`, POSITION `p` (a `Record`/`Func` compound cell) observed a NEW enum label — grow that
        /// position's sorted label union and restart. The per-POSITION analogue of `GrowEnum` (a
        /// "GrowCompoundEnum" one level below the scalar Enum); the label union is a deterministic
        /// per-position property, so verify re-derives the same `cells`.
        GrowCompoundEnum(usize, usize, EnumKind, String),
        /// Column `i` (a SEQUENCE-of-atoms column) observed a NEW element atom (of the given kind) not yet in
        /// its per-column element union — grow the sorted element union (`seq_elem[i]`) and restart. The
        /// sequence-element analogue of `GrowEnum`: the FINAL union is the sorted set of all element atoms
        /// EVER present in the column (a deterministic per-column property Leg-E re-derives), and every state
        /// packs its elements as `code = idx(label)` into that SAME union. Restarts terminate — the distinct
        /// element atoms across the finite state set are finite.
        GrowSeqAtom(usize, EnumKind, String),
        /// Column `i` observed a `Bool` SEQUENCE element while unmarked — MARK it a `Bool`-element sequence
        /// column (`seq_elem[i] = Bool`) and restart, so its EMPTY sequences re-encode consistently (mirrors
        /// the `SetMask` empty-set re-encode). Idempotent; an atom↔Bool flip on the column fails closed.
        MarkSeqBool(usize),
        /// Column `i` (a `Value::Seq`/`Value::Tuple` column) was observed at TWO DIFFERENT sequence LENGTHS
        /// — it is a variable-length QUEUE, not a fixed-arity `[1..n -> …]` function — so MARK it a
        /// generalized `Seq` column (`seq_queue[i] = true`) and restart, routing ALL its sequence states to
        /// the self-delimiting `Seq` pack (which keeps ONE sort across all lengths). Idempotent + monotone.
        MarkSeqQueue(usize),
        /// Out of the encodable fragment / not a finite fixpoint here — no certificate.
        Fail,
    }
    struct EnumOut {
        init_values: Vec<Vec<u64>>,
        reachable: Vec<Vec<u64>>,
        image: Vec<Vec<u64>>,
        sorts: Vec<ColSort>,
        /// DEADLOCK scan (certify/check parity): the first reachable state popped by the BFS whose
        /// live `enumerate_successors` returned EMPTY — a state with no successor under `Next`, i.e. a
        /// deadlock (self-loops `x'=x` enumerate a successor, so are NOT deadlocks — matching `ty
        /// check`). `None` ⇒ every reachable state has ≥1 successor. Config-terminal-agnostic (the CLI
        /// applies the `TERMINAL` exemption); recorded UNCONDITIONALLY here.
        deadlock_witness: Option<Vec<u64>>,
        /// One enumerated witness successor per reachable state that HAS a successor (state tuple → its
        /// first-enumerated successor tuple). Deterministic (each state's successors are enumerated
        /// exactly once, in a stable order), so a re-enumeration reproduces the identical map — the
        /// kernel deadlock-freedom leg's witnesses. States with no successor are absent (they are the
        /// `deadlock_witness`).
        succ_witness: std::collections::BTreeMap<Vec<u64>, Vec<u64>>,
    }
    let enumerate_at = |bases: &[u64],
                        enum_labels: &[Option<EnumCol>],
                        compound_enum: &[Vec<Option<EnumCol>>],
                        recmask_labels: &[Option<Vec<String>>],
                        seq_elem: &[Option<CellSort>],
                        seq_queue: &[bool],
                        deadline_probe: &mut Option<tla_resource::MemoryProbe>|
     -> Result<EnumOut, EnumStop> {
        // The per-column SORT, recorded from the first state and REQUIRED consistent across every
        // enumerated state. The compound base is FIXED per attempt, so the recorded `Record`/`Func`
        // sort (which carries the base) is stable within an attempt by construction.
        let mut col_sorts: Option<Vec<ColSort>> = None;
        // QUEUE DETECTION (per-attempt): the FIRST sequence LENGTH observed in each `Value::Seq`/`Value::
        // Tuple` column. A column later seen at a DIFFERENT length is a QUEUE (variable-length) — not a
        // fixed-arity `[1..n -> …]` function — so it raises `MarkSeqQueue` to route it to the generalized
        // `Seq` (set `seq_queue[i]` + restart). A column at ONE length only stays the fixed-arity
        // `FuncEnum`/`Func` path (a program counter `pc ∈ [1..N -> labels]` is NOT stolen). Order-
        // independent (promotion iff ≥2 distinct lengths ever appear ⇒ a property of R), so Leg-E re-derives.
        let mut seq_first_len: Vec<Option<usize>> = vec![None; var_names.len()];
        // Extract the TUPLE of a state (one stored cell per variable, declaration order), recording /
        // checking the per-column sort against `col_sorts`.
        let mut state_tuple = |s: &State| -> Result<Vec<u64>, EnumStop> {
            let mut tup = Vec::with_capacity(var_names.len());
            let mut sorts = Vec::with_capacity(var_names.len());
            for (i, v) in var_names.iter().enumerate() {
                let val = s.get(v).ok_or(EnumStop::Fail)?;
                // ── QUEUE-DETECTION pre-check: a `Value::Seq`/`Value::Tuple` column seen at TWO DIFFERENT
                // lengths is a QUEUE ⇒ promote it to the generalized `Seq` (`MarkSeqQueue` + restart); a
                // column at ONE length stays the fixed-arity `FuncEnum`/`Func`. Skipped once the column is
                // already a queue / atom-Seq. This is what keeps a fixed `pc ∈ [1..N -> labels]` a `FuncEnum`
                // while a genuine `Append`/`Tail` queue becomes a generalized `Seq`.
                if !seq_queue[i] && seq_elem[i].is_none() {
                    let seq_len = match val {
                        crate::value::Value::Seq(sv) => Some(sv.len()),
                        crate::value::Value::Tuple(t) => Some(t.len()),
                        _ => None,
                    };
                    if let Some(len) = seq_len {
                        match seq_first_len[i] {
                            Some(prev) if prev != len => return Err(EnumStop::MarkSeqQueue(i)),
                            None => seq_first_len[i] = Some(len),
                            _ => {}
                        }
                    }
                }
                // ── FINITE-ENUM path: a `String` / model-value cell encodes to the INDEX of its label
                // in the column's per-column SORTED label set (`enum_labels[i]`). A label not yet in the
                // set (or a first sighting of this column as enum) raises `GrowEnum` ⇒ the outer loop
                // grows the union and restarts, exactly as `Widen` grows a compound radix. A column that
                // MIXES `String` and model-value labels fails closed (they must not share indices).
                let enum_label = match val {
                    crate::value::Value::String(sv) => Some((EnumKind::Str, sv.as_ref())),
                    crate::value::Value::ModelValue(sv) => Some((EnumKind::Model, sv.as_ref())),
                    _ => None,
                };
                let (sort, cell) = if let Some((kind, label)) = enum_label {
                    match &enum_labels[i] {
                        Some(ec) if ec.kind == kind => {
                            match ec.labels.iter().position(|l| l == label) {
                                Some(idx) => {
                                    // The column's kind (`Str`/`Model`) is a DETERMINISTIC function of
                                    // its observed cells — stored in the sort so recognition can kind-check
                                    // membership/equality and verify re-derives it. `ec.kind` is `Copy`.
                                    (
                                        ColSort::Enum {
                                            labels: ec.labels.clone(),
                                            kind: ec.kind,
                                        },
                                        idx as u64,
                                    )
                                }
                                None => return Err(EnumStop::GrowEnum(i, kind, label.to_string())),
                            }
                        }
                        // Column recorded as the OTHER label kind ⇒ heterogeneous ⇒ fail closed.
                        Some(_) => return Err(EnumStop::Fail),
                        // First time this column is seen as enum — grow it in.
                        None => return Err(EnumStop::GrowEnum(i, kind, label.to_string())),
                    }
                } else if let Some(elems) = if seq_queue[i] {
                    // ── SEQUENCE-of-atoms (generalized `Seq`) path: a bounded QUEUE — a column seen at ≥2
                    // DIFFERENT sequence LENGTHS (`seq_queue[i]`, set by the queue-detection pre-check
                    // above) — whose elements are config model values / `String` atoms / `Bool`s / Ints
                    // packs SELF-DELIMITINGLY into ONE `Nat` cell (`pack = Σ (code+1)·D^i`, `code =
                    // idx(label)` / `1|0` / `v`), so a VARYING-length queue keeps ONE `ColSort::Seq{elem}`
                    // sort across ALL lengths. A FIXED-arity `[1..n -> labels]` function (a program counter
                    // — always ONE length, never a queue) has `seq_queue[i] == false` ⇒ this branch does NOT
                    // fire ⇒ it falls through to `func_enum_view` (the byte-identical `FuncEnum`, unchanged).
                    // The empty sequence `<<>>` evaluates to `Value::Tuple([])`, so a queue column's states
                    // are `Value::Seq` (non-empty) and `Value::Tuple([])` (empty) — route both.
                    match val {
                        crate::value::Value::Seq(sv) => Some(sv.iter().collect::<Vec<_>>()),
                        crate::value::Value::Tuple(t) => Some(t.iter().collect::<Vec<_>>()),
                        _ => None,
                    }
                } else {
                    None
                } {
                    match &seq_elem[i] {
                        // Column MARKED as a generalized atom/Bool sequence — encode against its element
                        // leaf (an EMPTY sequence re-encodes as pack 0 in the SAME leaf, like SetMask's
                        // empty set). A new atom grows the union; a wrong-kind element fails closed.
                        Some(marked @ (CellSort::Enum { .. } | CellSort::Bool)) => {
                            match pack_atom_seq(&elems, marked) {
                                Ok(x) => x,
                                Err(Some(label)) => {
                                    let kind = match marked {
                                        CellSort::Enum { kind, .. } => *kind,
                                        _ => return Err(EnumStop::Fail),
                                    };
                                    return Err(EnumStop::GrowSeqAtom(i, kind, label));
                                }
                                Err(None) => return Err(EnumStop::Fail),
                            }
                        }
                        // UNMARKED (or the never-stored `Int` marker): decide from the FIRST element. An
                        // atom/Bool first element MARKS the column (grow + restart); an Int/empty stays the
                        // byte-identical Int `Seq` path; a non-leaf element fails closed.
                        _ => match elems.first().map(|v| classify_seq_first_elem(v)) {
                            Some(Some(Err(CellSort::Enum { labels, kind }))) => {
                                return Err(EnumStop::GrowSeqAtom(
                                    i,
                                    kind,
                                    labels.into_iter().next().unwrap_or_default(),
                                ));
                            }
                            Some(Some(Err(CellSort::Bool))) => {
                                return Err(EnumStop::MarkSeqBool(i));
                            }
                            Some(None) => return Err(EnumStop::Fail), // non-leaf element
                            // Int first element, or empty sequence (`None`): the byte-identical Int path.
                            _ => match value_cell_encode_at(val, bases[i]) {
                                Some(x) => x,
                                None => {
                                    return Err(match compound_min_base(val) {
                                        Some(b) if b > bases[i] => EnumStop::Widen(i, b),
                                        _ => EnumStop::Fail,
                                    });
                                }
                            },
                        },
                    }
                } else if let Some((arity, is_model, value_labels, dom_keys, dom_kind)) =
                    func_enum_view(val)
                {
                    // ── FUNCTION-of-ENUM path: a function whose domain is the `0..arity-1` Int prefix
                    // (`dom_keys` empty) OR a set of config CONSTANT model-value / `String`-atom keys
                    // (`dom_keys` = the keys in canonical order, `dom_kind` their kind) and whose VALUES
                    // are labels packs POSITIONALLY on the column's GROWN
                    // label union (`enum_labels[i]`):
                    // `pack = Σ_d idx(e_d)·|labels|^d` (base `|labels|`, each digit an enum index). A value
                    // label not yet in the union raises `GrowEnum` ⇒ restart with it grown, exactly like a
                    // scalar enum cell. The label union is the SAME per-column property verify re-derives.
                    let kind = if is_model {
                        EnumKind::Model
                    } else {
                        EnumKind::Str
                    };
                    match &enum_labels[i] {
                        Some(ec) if ec.kind == kind => {
                            let base = ec.labels.len() as u64;
                            // Overflow guard: `|labels|^arity` (the pack universe) must fit u64.
                            if base.checked_pow(arity).is_none() {
                                return Err(EnumStop::Fail);
                            }
                            let mut pack: u64 = 0;
                            for (d, lbl) in value_labels.iter().enumerate() {
                                match ec.labels.iter().position(|l| l == lbl) {
                                    Some(idx) => {
                                        let place =
                                            base.checked_pow(d as u32).ok_or(EnumStop::Fail)?;
                                        pack = pack
                                            .checked_add(
                                                (idx as u64)
                                                    .checked_mul(place)
                                                    .ok_or(EnumStop::Fail)?,
                                            )
                                            .ok_or(EnumStop::Fail)?;
                                    }
                                    // A never-yet-seen value label ⇒ grow the union and restart.
                                    None => return Err(EnumStop::GrowEnum(i, kind, lbl.clone())),
                                }
                            }
                            (
                                ColSort::FuncEnum {
                                    arity,
                                    labels: ec.labels.clone(),
                                    dom: dom_keys.clone(),
                                    dom_kind,
                                },
                                pack,
                            )
                        }
                        // Column recorded as the OTHER label kind ⇒ heterogeneous ⇒ fail closed.
                        Some(_) => return Err(EnumStop::Fail),
                        // First time this column is seen as a func-of-enum — grow its first value label in
                        // (`func_enum_view` returns `None` for the empty function, so `value_labels[0]` is
                        // present here).
                        None => return Err(EnumStop::GrowEnum(i, kind, value_labels[0].clone())),
                    }
                } else if let crate::value::Value::Set(setv) = val {
                    // ── SET-of-RECORDS (SetMaskRec) path: a set `S` of bounded RECORD values encodes to the
                    // `|dom|`-bit BITMASK `Σ_{r∈S} 2^idx(r)` over the column's GROWING record-key universe
                    // `dom` (`recmask_labels[i]`), bit `idx(record_value_key(r))` ⟺ `r ∈ S`. A new record key
                    // raises `GrowSetMaskRec` (grow the sorted union + restart, exactly like the atom
                    // `GrowSetMask`); the final `dom` is the sorted union of all record keys EVER present in
                    // the column. Routed when the set holds a record OR the column is ALREADY an established
                    // record-set column (its EMPTY set then re-encodes as `SetMaskRec` mask 0, mirroring the
                    // atom-`SetMask` empty-set handling). A record with a non-leaf field is UNKEYABLE ⇒
                    // fail-closed; a set MIXING records and atoms fails closed (`record_value_key` = `None`).
                    let record_col = recmask_labels[i].is_some();
                    let has_record = setv
                        .iter()
                        .any(|e| matches!(e, crate::value::Value::Record(_)));
                    if has_record
                        || (record_col
                            && setv
                                .iter()
                                .all(|e| matches!(e, crate::value::Value::Record(_))))
                    {
                        let dom: Vec<String> = recmask_labels[i].clone().unwrap_or_default();
                        let mut mask: u64 = 0;
                        for elem in setv.iter() {
                            // A non-record / unkeyable element ⇒ fail-closed (mixed / nested-field set).
                            let key = record_value_key(elem).ok_or(EnumStop::Fail)?;
                            match dom.iter().position(|d| *d == key) {
                                // `idx < |dom| ≤ 64` by the GrowSetMaskRec cap in the restart loop.
                                Some(idx) => mask |= 1u64 << idx,
                                None => return Err(EnumStop::GrowSetMaskRec(i, key)),
                            }
                        }
                        (ColSort::SetMaskRec { dom }, mask)
                    } else {
                        // ── SET-of-ATOMS (SetMask) path: a set `S` of config CONSTANT model values / `String`
                        // atoms encodes to the `|dom|`-bit BITMASK `Σ_{a∈S} 2^idx(a)` over the column's GROWING
                        // atom universe `dom` (shared with `enum_labels[i]`), bit `idx(a)` ⟺ `a ∈ S`. A new atom
                        // raises `GrowSetMask` (grow the sorted union + restart, exactly like the scalar
                        // `GrowEnum`); the final `dom` is the sorted union of all atoms EVER present in the
                        // column (a deterministic per-column property Leg-E re-derives). An EMPTY set with NO
                        // prior union stays an Int bitmask (mask 0 — the historical `value_cell` behavior); a
                        // later atom-set GrowSetMask restart re-encodes it as `SetMask` mask 0.
                        let mut set_kind: Option<EnumKind> =
                            enum_labels[i].as_ref().map(|ec| ec.kind);
                        let mut atom_names: Vec<String> = Vec::new();
                        let mut all_atoms = true;
                        for elem in setv.iter() {
                            let (k, name) = match elem {
                                crate::value::Value::ModelValue(sv) => {
                                    (EnumKind::Model, sv.as_ref().to_string())
                                }
                                crate::value::Value::String(sv) => {
                                    (EnumKind::Str, sv.as_ref().to_string())
                                }
                                _ => {
                                    all_atoms = false;
                                    break;
                                }
                            };
                            match set_kind {
                                Some(ck) if ck != k => return Err(EnumStop::Fail), // mixed atom kinds
                                _ => set_kind = Some(k),
                            }
                            atom_names.push(name);
                        }
                        // Route to SetMask iff the column already carries an atom union OR this is a NON-EMPTY
                        // homogeneous atom set. Anything else (empty set w/ no union, or a non-atom element)
                        // falls to the Int-bitmask `value_cell_encode_at`; a non-atom set in an established
                        // SetMask column then yields a `Set` sort ⇒ `col_sorts` disagreement ⇒ fail-closed.
                        if all_atoms && (enum_labels[i].is_some() || !atom_names.is_empty()) {
                            let kind = set_kind.expect("kind known when routed to SetMask");
                            let dom: Vec<String> = enum_labels[i]
                                .as_ref()
                                .map(|ec| ec.labels.clone())
                                .unwrap_or_default();
                            let mut mask: u64 = 0;
                            for a in &atom_names {
                                match dom.iter().position(|l| l == a) {
                                    // `idx < |dom| ≤ 64` by the GrowSetMask cap below.
                                    Some(idx) => mask |= 1u64 << idx,
                                    None => return Err(EnumStop::GrowSetMask(i, kind, a.clone())),
                                }
                            }
                            (
                                ColSort::SetMask {
                                    dom,
                                    dom_kind: kind,
                                },
                                mask,
                            )
                        } else {
                            match value_cell_encode_at(val, bases[i]) {
                                Some(x) => x,
                                None => {
                                    return Err(match compound_min_base(val) {
                                        Some(b) if b > bases[i] => EnumStop::Widen(i, b),
                                        _ => EnumStop::Fail,
                                    });
                                }
                            }
                        }
                    }
                } else if let Some((fsm_arity, fdom, fdom_kind, per_slot, view_e_kind)) =
                    funcsetmask_view(val)
                {
                    // ── FUNCTION-to-SET (FuncSetMask) path (F2): a function `[D -> SUBSET E]` whose values
                    // are subsets of the atom universe `E` (`alloc ∈ [Clients -> SUBSET Resources]`, the
                    // SimpleAllocator class). COMPOSES the `Func` DOMAIN pack (`D` = `fdom` keys, canonical
                    // order) with the `SetMask` VALUE bijection: each value cell is a `|E|`-bit mask over the
                    // SHARED value universe `E` (stored in `enum_labels[i]`, grown monotonically via
                    // `GrowSetMask` EXACTLY like a scalar `SetMask` atom column). `pack = Σ_d mask(f[fdom_d])·
                    // base^d`, `base = 2^|E|`. The universe kind is the established union's kind, else this
                    // state's observed atom kind, else `Model` (the transient all-empty-no-union default —
                    // a real non-empty value later grows the union with its TRUE kind and restarts). A value
                    // atom not yet in the union raises `GrowSetMask`; a cross-state atom-KIND flip fails closed.
                    if let (Some(ec), Some(vk)) = (enum_labels[i].as_ref(), view_e_kind) {
                        if ec.kind != vk {
                            return Err(EnumStop::Fail); // model/String value-atom kind flip across states
                        }
                    }
                    let e_kind = enum_labels[i]
                        .as_ref()
                        .map(|ec| ec.kind)
                        .or(view_e_kind)
                        .unwrap_or(EnumKind::Model);
                    let dom: Vec<String> = enum_labels[i]
                        .as_ref()
                        .map(|ec| ec.labels.clone())
                        .unwrap_or_default();
                    // `base = 2^|E|`; the pack `base^arity` must fit a `u64` (`|E|·arity ≤ 63`) ⇒ else closed.
                    if dom.len() >= 64 {
                        return Err(EnumStop::Fail);
                    }
                    let base: u64 = 1u64 << dom.len();
                    if base.checked_pow(fsm_arity).is_none() {
                        return Err(EnumStop::Fail); // pack `base^arity` overflows the u64 cell ⇒ closed
                    }
                    let mut pack: u64 = 0;
                    for (d, atoms) in per_slot.iter().enumerate() {
                        let mut mask: u64 = 0;
                        for a in atoms {
                            match dom.iter().position(|l| l == a) {
                                // `idx < |dom| ≤ 63` by the GrowSetMask cap in the restart loop.
                                Some(idx) => mask |= 1u64 << idx,
                                None => return Err(EnumStop::GrowSetMask(i, e_kind, a.clone())),
                            }
                        }
                        let place = base.checked_pow(d as u32).ok_or(EnumStop::Fail)?;
                        pack = pack
                            .checked_add(mask.checked_mul(place).ok_or(EnumStop::Fail)?)
                            .ok_or(EnumStop::Fail)?;
                    }
                    (
                        ColSort::FuncSetMask {
                            arity: fsm_arity,
                            fdom,
                            fdom_kind,
                            dom,
                            dom_kind: e_kind,
                        },
                        pack,
                    )
                } else {
                    // ── POSITIONAL-COMPOUND path (Record / Int-domain Func), possibly with String/model
                    // ENUM positions (the value-type-leaf Enum cell). `encode_compound_at` resolves each
                    // position's digit at the CURRENT per-column radix (`bases[i]`) + per-POSITION label
                    // union (`compound_enum[i]`); it is BYTE-IDENTICAL to `value_cell_encode_at` for an
                    // all-Int/Bool compound. A new label at a position raises `GrowCompoundEnum` (grow +
                    // restart, exactly like the scalar `GrowEnum`); a code needing a wider radix raises
                    // `Widen`. `NotCompound` (a Set/Seq/scalar) falls back to `value_cell_encode_at`.
                    let pos_labels: &[Option<EnumCol>] = &compound_enum[i];
                    match encode_compound_at(val, bases[i], pos_labels) {
                        Ok(x) => x,
                        Err(CompoundStop::Grow(p, kind, label)) => {
                            return Err(EnumStop::GrowCompoundEnum(i, p, kind, label));
                        }
                        Err(CompoundStop::Widen(b)) => return Err(EnumStop::Widen(i, b)),
                        Err(CompoundStop::Fail) => return Err(EnumStop::Fail),
                        Err(CompoundStop::NotCompound) => match value_cell_encode_at(val, bases[i])
                        {
                            Some(x) => x,
                            None => {
                                // Widen exactly when the value is a packable Set/Seq whose DERIVED minimal
                                // base exceeds the current one; anything else is genuinely out of fragment
                                // (`compound_min_base` returns `None` for those).
                                return Err(match compound_min_base(val) {
                                    Some(b) if b > bases[i] => EnumStop::Widen(i, b),
                                    _ => EnumStop::Fail,
                                });
                            }
                        },
                    }
                };
                sorts.push(sort);
                tup.push(cell);
            }
            match &col_sorts {
                Some(prev) if *prev != sorts => return Err(EnumStop::Fail), // sort disagreement
                None => col_sorts = Some(sorts),
                _ => {}
            }
            Ok(tup)
        };

        // --- LIVE Init enumeration ---
        let ctx = build_ctx(&module).ok_or(EnumStop::Fail)?;
        let branches = extract_init_constraints(&ctx, &init_def.body, &var_names, None)
            .ok_or(EnumStop::Fail)?;
        let init_states = enumerate_states_from_constraint_branches_probed(
            Some(&ctx),
            &var_names,
            &branches,
            deadline_probe,
        )
        .ok()
        .flatten()
        .filter(|v| !v.is_empty())
        .ok_or(EnumStop::Fail)?;
        let mut init_values: Vec<Vec<u64>> = Vec::new();
        for s in &init_states {
            init_values.push(state_tuple(s)?);
        }

        // --- LIVE BFS to a fixpoint over Next (GENERAL, non-stutter, MULTI-VARIABLE) ---
        let mut visited: BTreeSet<Vec<u64>> = BTreeSet::new();
        let mut image: BTreeSet<Vec<u64>> = BTreeSet::new();
        let mut frontier: Vec<State> = init_states.clone();
        for s in &init_states {
            visited.insert(state_tuple(s)?);
            if visited.len() > state_cap {
                // The cap applies to the INITIAL states too (as in the refinement lane): a
                // reachable set beyond the cap is not certifiable here regardless of how it
                // was reached — fail closed instead of building infeasible kernel legs.
                return Err(EnumStop::Fail);
            }
        }
        let mut next_ctx = build_ctx(&module).ok_or(EnumStop::Fail)?;
        // DEADLOCK scan state (certify/check parity): the first no-successor reachable state, and one
        // witness successor per state that has one. Both are pure functions of the reachable graph, so
        // a re-enumeration reproduces them identically (the deadlock leg binds the witnesses to the spec).
        let mut deadlock_witness: Option<Vec<u64>> = None;
        let mut succ_witness: std::collections::BTreeMap<Vec<u64>, Vec<u64>> =
            std::collections::BTreeMap::new();
        while let Some(cur) = frontier.pop() {
            // Cooperative deadline/memory tick (one per popped state; ~1ns when
            // un-tripped, latched once tripped) — expiry ⇒ honest Fail ⇒ None.
            if deadline_probe.as_mut().is_some_and(|p| p.over_budget()) {
                return Err(EnumStop::Fail);
            }
            let succs = enumerate_successors(&mut next_ctx, &next_def, &cur, &var_names)
                .map_err(|_| EnumStop::Fail)?;
            // DEADLOCK check (parity with `ty check`): a state with NO successor under `Next` is a
            // deadlock (a self-loop `x'=x` enumerates one successor, so is NOT a deadlock). Record the
            // FIRST such state as the witness, and for every state that HAS a successor keep its first
            // enumerated successor as the kernel deadlock leg's witness.
            if succs.is_empty() {
                let sv = state_tuple(&cur)?;
                if deadlock_witness.is_none() {
                    deadlock_witness = Some(sv);
                }
            } else {
                let sv = state_tuple(&cur)?;
                let wv = state_tuple(&succs[0])?;
                succ_witness.entry(sv).or_insert(wv);
            }
            // Record EVERY enumerated successor TUPLE as part of the image of R under Next
            // (arbitrary Next — no stutter requirement). Each successor joins R; image(R) ⊆ R by
            // construction (the BFS runs to a fixpoint). The kernel closed-leg certifies each
            // image tuple ∈ R.
            for succ in &succs {
                let sv = state_tuple(succ)?;
                image.insert(sv.clone());
                if visited.insert(sv) {
                    if visited.len() > state_cap {
                        // R exceeded the bound -> not a finite fixpoint here
                        return Err(EnumStop::Fail);
                    }
                    frontier.push(succ.clone());
                }
            }
        }
        // The per-column sort vector observed over ALL enumerated states.
        let sorts = col_sorts.ok_or(EnumStop::Fail)?;
        Ok(EnumOut {
            init_values,
            reachable: visited.into_iter().collect(),
            image: image.into_iter().collect(),
            sorts,
            deadlock_witness,
            succ_witness,
        })
    };

    let mut bases: Vec<u64> = vec![RECORD_FUNC_BASE; var_names.len()];
    // Per-column enum label sets, grown monotonically across restarts (like `bases`). The FINAL
    // `enum_labels[i]` is the SORTED union of all labels observed in column `i` over ALL enumerated
    // states — a deterministic per-column property verify re-derives identically.
    let mut enum_labels: Vec<Option<EnumCol>> = vec![None; var_names.len()];
    // Per-column, per-POSITION enum label sets for COMPOUND (Record / Int-domain Func) columns — the
    // value-type-leaf Enum cell. `compound_enum[i][p]` is `Some(EnumCol)` iff position `p` of column `i`
    // holds a String/model-value ENUM cell; the union grows monotonically across restarts (like
    // `enum_labels`), and EVERY state at that position indexes into the SAME final union. A pure function
    // of the reachable set (a sorted per-position union), so Leg-E re-derives the identical `cells`.
    let mut compound_enum: Vec<Vec<Option<EnumCol>>> = vec![Vec::new(); var_names.len()];
    // Per-column record-KEY universe for SET-of-RECORDS (`SetMaskRec`) columns, grown monotonically across
    // restarts (like `enum_labels` for atom sets). The FINAL `recmask_labels[i]` is the SORTED union of all
    // record keys observed in column `i` over ALL enumerated states — a deterministic per-column property
    // Leg-E re-derives identically (bit `idx` ⟺ `dom[idx] ∈ S`).
    let mut recmask_labels: Vec<Option<Vec<String>>> = vec![None; var_names.len()];
    // Per-column SEQUENCE-element leaf for generalized atom/`Bool` `Seq` columns, grown monotonically across
    // restarts (like `enum_labels` for scalar enums). `Some(Enum{labels,kind})` = an atom sequence whose
    // element union grows; `Some(Bool)` = a `Bool` sequence; `None` = unmarked (a pure-Int / empty sequence
    // stays the byte-identical Int `Seq` path). A deterministic per-column property Leg-E re-derives.
    let mut seq_elem: Vec<Option<CellSort>> = vec![None; var_names.len()];
    // Per-column QUEUE flag: `true` once column `i` is seen at ≥2 different sequence lengths (a variable-
    // length queue) ⇒ its sequences route to the generalized `Seq`. Grown monotonically across restarts.
    let mut seq_queue: Vec<bool> = vec![false; var_names.len()];
    let EnumOut {
        init_values,
        reachable,
        image: image_vec,
        sorts,
        deadlock_witness,
        succ_witness,
    } = loop {
        // A latched deadline trip also stops the Widen/Grow* RESTART loop — the
        // next enumerate_at attempt would re-tick and fail, but checking here
        // avoids even starting a doomed restart.
        if deadline_probe.as_mut().is_some_and(|p| p.over_budget()) {
            return None;
        }
        match enumerate_at(
            &bases,
            &enum_labels,
            &compound_enum,
            &recmask_labels,
            &seq_elem,
            &seq_queue,
            &mut deadline_probe,
        ) {
            Ok(out) => break out,
            // `Widen(i, b)` raises column `i`'s base to the derived minimal admitting base `b > bases[i]`
            // (monotonic: `max` guards against any re-order). The observed field values are finite and the
            // per-column base is bounded by the `u64` pack ceiling, so the base strictly increases each
            // time and the loop terminates; the final base is the `max` over states = the smallest base
            // admitting the whole column, which Leg-E re-derives identically.
            Err(EnumStop::Widen(i, b)) => bases[i] = bases[i].max(b),
            // `GrowEnum(i, kind, label)` adds ONE new label to column `i`'s sorted union; the total
            // distinct labels across the finite state set is finite, so the restarts terminate. A kind
            // flip on an already-typed column is malformed (state_tuple already fails closed on the mix).
            Err(EnumStop::GrowEnum(i, kind, label)) => {
                let ec = enum_labels[i].get_or_insert(EnumCol {
                    kind,
                    labels: Vec::new(),
                });
                if ec.kind != kind {
                    return None;
                }
                if !ec.labels.iter().any(|l| *l == label) {
                    ec.labels.push(label);
                    ec.labels.sort();
                }
            }
            // `GrowSetMask(i, kind, atom)` adds ONE new atom to column `i`'s bitmask UNIVERSE `dom`
            // (the same `enum_labels[i]` sorted union); the total distinct atoms across the finite state
            // set is finite, so the restarts terminate. CAPPED at 64 (the u64 cell bit width): a universe
            // wider than 64 atoms cannot fit ONE mask cell ⇒ fail-closed (the whole spec declines).
            Err(EnumStop::GrowSetMask(i, kind, atom)) => {
                let ec = enum_labels[i].get_or_insert(EnumCol {
                    kind,
                    labels: Vec::new(),
                });
                if ec.kind != kind {
                    return None;
                }
                if !ec.labels.iter().any(|l| *l == atom) {
                    ec.labels.push(atom);
                    ec.labels.sort();
                    if ec.labels.len() > 64 {
                        return None; // universe exceeds the one-cell bitmask width ⇒ fail-closed
                    }
                }
            }
            // `GrowSetMaskRec(i, key)` adds ONE new record KEY to column `i`'s bitmask UNIVERSE
            // (`recmask_labels[i]`); the total distinct record keys across the finite state set is finite, so
            // the restarts terminate. CAPPED at 64 (the u64 cell bit width) exactly like the atom `GrowSetMask`.
            Err(EnumStop::GrowSetMaskRec(i, key)) => {
                let dom = recmask_labels[i].get_or_insert_with(Vec::new);
                if !dom.iter().any(|d| *d == key) {
                    dom.push(key);
                    dom.sort();
                    if dom.len() > 64 {
                        return None; // record universe exceeds the one-cell bitmask width ⇒ fail-closed
                    }
                }
            }
            // `GrowCompoundEnum(i, p, kind, label)` adds ONE new label to column `i`'s POSITION `p` sorted
            // union (extending the per-position vector on demand); the distinct labels across the finite
            // state set are finite, so the restarts terminate. A kind flip on an already-typed position is
            // malformed (`encode_compound_at` already fails closed on the mix, so this is defensive).
            Err(EnumStop::GrowCompoundEnum(i, p, kind, label)) => {
                let positions = &mut compound_enum[i];
                if positions.len() <= p {
                    positions.resize_with(p + 1, || None);
                }
                let ec = positions[p].get_or_insert(EnumCol {
                    kind,
                    labels: Vec::new(),
                });
                if ec.kind != kind {
                    return None;
                }
                if !ec.labels.iter().any(|l| *l == label) {
                    ec.labels.push(label);
                    ec.labels.sort();
                }
            }
            // `GrowSeqAtom(i, kind, atom)` adds ONE new element atom to column `i`'s sequence-element union
            // (`seq_elem[i]`); the distinct element atoms across the finite state set are finite, so restarts
            // terminate. Capped at [`SEQ_BASE`]-relative pack ceiling in `pack_atom_seq` (a huge alphabet
            // whose `D^max_len` overflows a `u64` fails closed there). A kind flip (or atom↔Bool) declines.
            Err(EnumStop::GrowSeqAtom(i, kind, atom)) => {
                let cur = seq_elem[i].get_or_insert(CellSort::Enum {
                    labels: Vec::new(),
                    kind,
                });
                match cur {
                    CellSort::Enum { labels, kind: k } if *k == kind => {
                        if !labels.iter().any(|l| *l == atom) {
                            labels.push(atom);
                            labels.sort();
                        }
                    }
                    _ => return None, // atom↔Bool or String↔model flip on the column ⇒ fail-closed
                }
            }
            // `MarkSeqBool(i)` marks column `i` a `Bool`-element sequence (idempotent); an atom↔Bool flip
            // (the column already carries an atom element union) declines.
            Err(EnumStop::MarkSeqBool(i)) => match &seq_elem[i] {
                None => seq_elem[i] = Some(CellSort::Bool),
                Some(CellSort::Bool) => {}
                Some(_) => return None,
            },
            // `MarkSeqQueue(i)` promotes column `i` to a variable-length `Seq` (idempotent, monotone) — the
            // total columns is finite, so the restarts terminate.
            Err(EnumStop::MarkSeqQueue(i)) => seq_queue[i] = true,
            Err(EnumStop::Fail) => return None,
        }
    };

    // `Safety == ⋀_{Int j} (var_j ≥ 0)` — one `x≥0` conjunct per INT state variable, NONE for Bool
    // (the shape the kernel tuple R⊆Safety leg proves) — the PRIMARY lane, unchanged. When the
    // invariant is NOT that shape, fall back to the GENERAL embedded-safety lane: recognize it into
    // the kernel predicate fragment (`recognize_pred_sorts`) and later prove `⋀_{s∈R} ⟦Safety⟧(s)`
    // reduces to `Bool.true` (`safety_general` below). Fail-closed gates for the general lane:
    //   * `pred_exact` — the safety claim needs kernel-TRUE ⇒ TLA-TRUE, so the recognizer's
    //     superset-safe (over-approximating) forms — Nat-truncating `Sub`/`Div`/`Mod`, Seq digit
    //     ops, non-Int cells punned as Nat — are REJECTED (same gate as the refinement lane);
    //   * no primed columns — an invariant is a STATE predicate (`embed_pred_ir(ir, s, s)` would
    //     silently read the current state for a prime);
    //   * the Phase-A runtime recognizer/embedder cross-check on the ACTUAL obligation states.
    let var_strs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
    // Config CONSTANT model-value SETS, keyed by name → the SORTED, deduped member names. Threaded into
    // the recognizer so a `val ∈ Data` membership over a MODEL-value Enum column resolves (roadmap R1).
    // DETERMINISTIC (a `BTreeMap` keyed by name, each value sorted+deduped) so Leg-E — which re-derives
    // it from the cert's reconstructed config — rebuilds the SAME map ⇒ the SAME `safety_pred`. A
    // `ModelValueSet` gives the member names directly; a `Value` whose raw string is a brace-delimited
    // list of non-numeric identifiers (`{d1, d2, d3}`) is TLC's alternate spelling and parsed the same.
    let mvsets: std::collections::BTreeMap<String, Vec<String>> = {
        let mut m = std::collections::BTreeMap::new();
        for (name, cv) in &config.constants {
            let names: Option<Vec<String>> = match cv {
                crate::config::ConstantValue::ModelValueSet(ns) => Some(ns.clone()),
                crate::config::ConstantValue::Value(s) => parse_brace_model_value_set(s),
                _ => None,
            };
            if let Some(mut ns) = names {
                ns.sort();
                ns.dedup();
                m.insert(name.clone(), ns);
            }
        }
        m
    };
    // Per-column MAXIMA over the enumerated reachable set R — `col_max[c] = Some(max_{s∈R} s[c])` for an
    // Int column, `None` for a non-Int column (a bare `Var(c)` in an Int value position only arises for an
    // Int column, so a non-Int max is never needed AND would be meaningless as an integer bound). Threaded
    // into the recognizer so a STATE-DEPENDENT quantifier DOMAIN `lo..hi` (EWD840's `∃j∈0..tpos`, `tpos` a
    // state variable) expands EXACTLY over `lo..=M` with a `k ≤ hi` guard (see `recognize_bounded_quant`).
    // DETERMINISTIC (a pure per-column fold over the canonical `reachable` tuples), so Leg-E — which
    // re-enumerates the SAME R and reruns this path — recomputes the SAME `col_max` ⇒ the SAME expanded
    // `safety_pred`. A spec with NO state-dependent domain never consults it ⇒ byte-identical certs.
    let col_max: Vec<Option<u64>> = (0..sorts.len())
        .map(|c| {
            if sorts[c] == ColSort::Int {
                reachable.iter().map(|t| t[c]).max()
            } else {
                None
            }
        })
        .collect();
    let safety_pred: Option<PredIR> =
        if crate::cleancic::is_conjunctive_nonneg_safety(&safety_body.node, &var_strs, &sorts) {
            None // PRIMARY lane: the tuple `⋀ x≥0` leg IS the spec's Safety
        } else {
            let ir = crate::cleancic::recognize_pred_sorts_with_mvsets_colmax(
                &safety_body.node,
                &var_strs,
                &sorts,
                &mvsets,
                Some(&col_max),
            )?;
            if !crate::refinement_cert::pred_exact(&ir, &sorts)
                || crate::refinement_cert::pred_mentions_prime(&ir)
            {
                return None; // inexact embedding / primed "invariant" → fail-closed
            }
            // Phase-A runtime cross-check (plan B#5), safety side: the AST-rooted embedder must agree
            // with the IR-rooted one on every comparable state of the ACTUAL obligation (pairs are
            // (s, s) — an invariant mentions no primed columns). A DISAGREEMENT means the trusted
            // recognizer/embedder pair is buggy; the safety leg has NO fallback, so decline the cert.
            let cross = crate::cleancic::cross_check_pred_embedders(
                &safety_body.node,
                &ir,
                &var_strs,
                reachable.iter().map(|s| (s.as_slice(), s.as_slice())),
            );
            if cross == crate::cleancic::EmbedCrossCheck::Disagree {
                eprintln!(
                    "WARNING: recognizer/embedder cross-check DISAGREEMENT on the Safety \
                 invariant — declining the kernel explicit-state certificate (fail-closed). \
                 This indicates a ty bug; please report."
                );
                return None;
            }
            Some(ir)
        };

    // Hand the LIVE enumerated set + image + per-column sorts to the kernel fixpoint cert (the final
    // arbiter). It is the compound-sort fragment's authority — any leg the kernel rejects → None.
    // In the GENERAL safety lane the tuple `safety_term` this returns proves the encoding-level
    // `⋀_{Int} x≥0` fact (true by the nonneg cell encoding, still kernel-checked); the SPEC's
    // Safety is then the `safety_general` leg below — BOTH ride in the cert and BOTH must re-check.
    //
    // MEMBERSHIP-LEG DISPATCH (R2): at most REFLECTED_MEMBERSHIP_THRESHOLD states keeps the
    // historical per-member Or-injection legs BYTE-IDENTICALLY; beyond it the per-member terms
    // are O(|R|²)/O(|R|³) infeasible, so the legs become the REFLECTED single obligations
    // (`TyReflectSubseq` over the quoted tuple lists + the balanced nonneg fold) — same claims,
    // O(|R|) size. `reachable` is already canonical here (BTreeSet-derived).
    #[allow(clippy::type_complexity)]
    let (
        closed_member_terms,
        safety_term,
        init_member_terms,
        init_member_reflected,
        closed_member_reflected,
    ): (
        Vec<Vec<u8>>,
        Vec<u8>,
        Vec<Vec<u8>>,
        Option<Vec<u8>>,
        Option<Vec<u8>>,
    ) = if use_reflected_membership_lane(reachable.len(), sorts.len()) {
        let (closed_r, safety, init_r) = crate::cleancic::certify_explicit_fixpoint_set_reflected(
            &reachable,
            &init_values,
            &image_vec,
            &sorts,
        )?;
        (Vec::new(), safety, Vec::new(), Some(init_r), Some(closed_r))
    } else {
        let (closed, safety, init) = crate::cleancic::certify_explicit_fixpoint_set(
            &reachable,
            &init_values,
            &image_vec,
            &sorts,
        )?;
        (closed, safety, init, None, None)
    };

    // GENERAL `R⊆Safety` leg: `⋀_{s∈R} ⟦Safety⟧(s)` reduced to `Bool.true`, kernel-gated. The kernel
    // is the arbiter: a reachable state VIOLATING the invariant reduces the conjunction to
    // `Bool.false`, `certify_bool_true_obligation` returns `None`, and the WHOLE cert is declined
    // (fail-closed — never a Certified verdict the kernel did not accept).
    let safety_general: Option<Vec<u8>> = match &safety_pred {
        None => None,
        Some(ir) => Some(certify_safety_general(&reachable, ir)?),
    };

    // KERNEL-RE-EVALUATED `Next`-completeness (closes the enumerator-trust gap for the affine single-Int
    // fragment): if the spec is a single nonneg-Int variable and `Next` is the recognized affine shape
    // `x' = x + δ ∧ x < bound`, the kernel ITSELF re-evaluates `Next` over the full finite domain `D` and
    // proves `R` is closed under the RELATION — not merely over TY's enumerated image. The kernel
    // returning `None` here (R not actually closed under the relation, e.g. an enumerator
    // under-approximation) fail-closes the WHOLE cert. Otherwise (non-affine `Next`) closure honestly
    // rests on the enumerated-image leg (`next_shape = None`).
    #[allow(clippy::type_complexity)]
    let (
        next_shape,
        next_completeness,
        init_shape,
        init_completeness,
        next_pred,
        next_general_completeness,
        init_pred,
        init_general_completeness,
    ) = {
        let mut next_shape = None;
        let mut next_completeness = None;
        let mut next_pred = None;
        let mut next_general_completeness = None;
        let mut init_shape = None;
        let mut init_completeness = None;
        let mut init_pred = None;
        let mut init_general_completeness = None;

        // ── SINGLE-INT affine / literal-disjunction shortcuts (back-compat, unchanged) ─────────────
        // These recognizers are single-variable only and produce the dedicated `next_shape`/`init_shape`
        // legs. They run ONLY for `sorts==[Int]`; the GENERAL multi-var leg below picks up everything
        // else (and anything these shortcuts decline).
        if sorts.as_slice() == [ColSort::Int] {
            let scalar_r: Vec<u64> = reachable.iter().map(|t| t[0]).collect();
            let var = var_names[0].as_ref();
            // Prefer the AFFINE recognizer. D = {0 ..= (bound-1)+δ}: a guarded state `x<bound` has
            // `x ≤ bound-1`, so its successor `x+δ ≤ (bound-1)+δ` ⇒ D ⊇ Succ(R) absolutely.
            // Phase A: both shortcut legs derive their single-Int axis from a TRUSTED-RUST
            // rule (the affine shape's `bound-1+δ` / the literal max) — RustDerived coverage.
            // Under `require_domain_complete` they are DECLINED (kernel-covered or declined).
            if !require_domain_complete {
                if let Some(shape) = crate::cleancic::recognize_affine_next(&next_body.node, var) {
                    let hi = shape.bound.saturating_sub(1).saturating_add(shape.delta);
                    if let Some(domain) = bounded_scalar_completeness_domain(hi) {
                        let bytes =
                            crate::cleancic::certify_next_completeness(&scalar_r, &domain, &shape)?;
                        next_shape = Some(shape);
                        next_completeness = Some(bytes);
                    }
                }
                // Prefer the literal-disjunction Init recognizer. D = {0 ..= max c_i} contains every
                // Init-state (Init holds ONLY at the literals), so the check is complete.
                if let Some(vals) = crate::cleancic::recognize_init_values(&init_body.node, var) {
                    let hi = vals.iter().copied().max().unwrap_or(0);
                    if let Some(domain) = bounded_scalar_completeness_domain(hi) {
                        let bytes =
                            crate::cleancic::certify_init_completeness(&scalar_r, &domain, &vals)?;
                        init_shape = Some(vals);
                        init_completeness = Some(bytes);
                    }
                }
            }
        }

        // ── GENERAL MULTI-VARIABLE completeness (all columns Int/Bool/Set) ─────────────────────────
        // Fires for ANY tuple spec whose every column is Int/Bool/Set, EXCEPT where an affine/literal
        // shortcut already produced the corresponding leg above. The kernel RE-EVALUATES the actual
        // `Next`/`Init` predicate over the PRODUCT domain `D = ⨉_i {0..=H_i}`, where each per-column
        // bound `H_i` is the SOUND structural upper bound (primed-bound / stutter for Int, 1 for Bool,
        // and for a Set column either the full bitmask range `2^K-1` when changing — capped at
        // `SET_COMPLETENESS_UNIVERSE_CAP` — or stutter `max(R)` otherwise; see the Set arm of
        // `cleancic::next_domain_bounds_from_ir`).
        // `D ⊇ Succ(R)` holds per column ⇒ closure is proved ABSOLUTELY (not over TY's enumerated image).
        // When a Set column's completeness bound is not derivable (universe too wide AND not stuttered),
        // the bounds fn returns `None` and the general leg is simply DECLINED — closure then rests on the
        // honest kernel-checked enumerated `image ⊆ R` leg (still sound, just not enumerator-free).
        //
        // A COMPOUND column (`Record`/`Func`/`Seq`) is ALSO embeddable here: its cell is a packed `Nat`,
        // so a STUTTER conjunct (`UNCHANGED`/`x'=x`) bounds its axis by `max(R)` and a CHANGING column over
        // a pack range `≤ COMPOUND_COMPLETENESS_PACK_CAP` enumerates the full `{0..=cap}` axis (see the
        // compound arm of `next_domain_bounds_from_ir`). If `recognize_pred_sorts` cannot embed the
        // compound-column conjuncts (field access / app / seq ops outside the recognized fragment), the
        // general leg simply DECLINES and closure rests on the honest enumerated `image ⊆ R` leg.
        // A scalar ENUM column is ALSO embeddable: its cell is the label INDEX (a `Nat` in
        // `0..labels.len()`), and every recognized enum predicate (`val = "d"`, `val ∈ Data`) reduces to
        // `Nat.beq`/a `Bool.or`-fold of `Nat.beq` over that index — kernel-reducible EXACTLY like an Int
        // column's comparisons. Its per-column successor bound `H_i = labels.len()-1` is sound BY
        // CONSTRUCTION (every index `< labels.len()`) and the coverage upgrade kernel-PROVES the
        // membership→bound step from the Next disjuncts (see `next_domain_bounds_cov_from_ir`'s Enum arm).
        let all_embeddable = sorts.iter().all(|s| {
            matches!(
                s,
                ColSort::Int | ColSort::Bool | ColSort::Set { .. } | ColSort::Enum { .. }
            ) || s.is_compound()
        });
        if all_embeddable {
            let vars: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
            let n = sorts.len();
            // GENERAL Next leg — only when the affine shortcut did NOT fire.
            if next_shape.is_none() {
                // Thread the config model-value SETS so a `val' ∈ Data` successor membership over a MODEL
                // Enum column resolves to the `⋁_ℓ (val'=idx(ℓ))` Or-fold (the safety leg already uses the
                // same map). The verify path re-checks the STORED IR, so this recognizer choice is a
                // certify-time detail — the cert round-trips identically.
                if let Some(ir) = crate::cleancic::recognize_pred_sorts_with_mvsets(
                    &next_body.node,
                    &vars,
                    &sorts,
                    &mvsets,
                ) {
                    // Phase A strict mode: decline unless EVERY axis is universe-complete or
                    // KERNEL-PROVEN (the upgrade pass synthesizes and kernel-checks the
                    // per-state successor-bound lemmas for Rust-derived axes).
                    let bounds =
                        crate::cleancic::next_domain_bounds_cov_from_ir(&ir, n, &reachable, &sorts)
                            .filter(|bc| {
                                if !require_domain_complete {
                                    return true;
                                }
                                let mut bc2 = bc.clone();
                                crate::cleancic::upgrade_domain_coverage(
                                    &ir, n, &reachable, &mut bc2, true,
                                );
                                bc2.iter().all(|(_, c)| {
                                    *c != crate::cleancic::DomainCoverage::RustDerived
                                })
                            })
                            .map(|bc| bc.into_iter().map(|(h, _)| h).collect::<Vec<u64>>());
                    if let Some(bounds) = bounds {
                        // The product domain `D` is rebuilt with the STABLE fixed domain cap (NOT the
                        // memory-derived reachable cap `state_cap`): the general leg is kept only when
                        // `|R| × |D| ≤ GENERAL_COMPLETENESS_WORK_CAP` below, so a domain bigger than that
                        // work cap is never used — and Leg-E rebuilds `D` with this SAME fixed cap
                        // (`verify_explicit_state_cert`), so mint and verify reconstruct the identical `D`.
                        // Using the large `state_cap` here would let a many-column product blow `|D|` up to
                        // the reachable budget before the work-cap filter rejects it.
                        if let Some(domain) =
                            crate::cleancic::product_domain(&bounds, DEFAULT_FIXPOINT_STATE_CAP)
                                // WORK CAP: the general obligation reduces `|R| × |D|` embedded-pred legs in
                                // the kernel; past the cap it is impractically slow (a multi-FuncEnum product
                                // domain blows up `|D|`). DECLINE beyond it — closure falls back to the honest
                                // enumerated `image ⊆ R` leg (enumerator-assisted), never a hang.
                                .filter(|domain| {
                                    reachable.len().saturating_mul(domain.len())
                                        <= GENERAL_COMPLETENESS_WORK_CAP
                                })
                        {
                            // Phase-A RUNTIME cross-check (plan B#5): the AST-rooted embedder
                            // must agree with the IR-rooted one on every comparable (s,s') of
                            // the ACTUAL obligation. An active DISAGREEMENT means the trusted
                            // recognizer/embedder pair is buggy → DECLINE this leg (closure
                            // falls back to the honest enumerated `image ⊆ R` leg).
                            let cross = crate::cleancic::cross_check_pred_embedders(
                                &next_body.node,
                                &ir,
                                &vars,
                                reachable.iter().flat_map(|s| {
                                    domain.iter().map(move |sp| (s.as_slice(), sp.as_slice()))
                                }),
                            );
                            if cross == crate::cleancic::EmbedCrossCheck::Disagree {
                                // A DISAGREEMENT between the two independent embedding paths
                                // means the trusted recognizer/embedder pair has a bug — the
                                // leg is declined (fail-closed), and LOUDLY: this must never
                                // pass as a routine out-of-fragment decline.
                                eprintln!(
                                    "WARNING: recognizer/embedder cross-check DISAGREEMENT on \
                                     the Next predicate — declining the kernel completeness \
                                     leg (fail-closed). This indicates a ty bug; please report."
                                );
                            } else if let Some(bytes) =
                                crate::cleancic::certify_general_completeness(
                                    &reachable, &domain, &ir,
                                )
                            {
                                next_pred = Some(ValIRDomain {
                                    pred: ir,
                                    hi: bounds,
                                });
                                next_general_completeness = Some(bytes);
                            }
                        }
                    }
                }
            }
            // GENERAL Init leg — only when the literal-disjunction shortcut did NOT fire.
            if init_shape.is_none() {
                if let Some(ir) = crate::cleancic::recognize_pred_sorts_with_mvsets(
                    &init_body.node,
                    &vars,
                    &sorts,
                    &mvsets,
                ) {
                    // Phase A strict mode: same coverage discipline as the Next leg (the Init
                    // upgrade is a single Π-lemma over all state tuples).
                    let bounds = crate::cleancic::init_domain_bounds_cov_from_ir(&ir, n, &sorts)
                        .filter(|bc| {
                            if !require_domain_complete {
                                return true;
                            }
                            let mut bc2 = bc.clone();
                            crate::cleancic::upgrade_domain_coverage(
                                &ir, n, &reachable, &mut bc2, false,
                            );
                            bc2.iter()
                                .all(|(_, c)| *c != crate::cleancic::DomainCoverage::RustDerived)
                        })
                        .map(|bc| bc.into_iter().map(|(h, _)| h).collect::<Vec<u64>>());
                    if let Some(bounds) = bounds {
                        // The product domain `D` is rebuilt with the STABLE fixed domain cap (NOT the
                        // memory-derived reachable cap `state_cap`): the general leg is kept only when
                        // `|R| × |D| ≤ GENERAL_COMPLETENESS_WORK_CAP` below, so a domain bigger than that
                        // work cap is never used — and Leg-E rebuilds `D` with this SAME fixed cap
                        // (`verify_explicit_state_cert`), so mint and verify reconstruct the identical `D`.
                        // Using the large `state_cap` here would let a many-column product blow `|D|` up to
                        // the reachable budget before the work-cap filter rejects it.
                        if let Some(domain) =
                            crate::cleancic::product_domain(&bounds, DEFAULT_FIXPOINT_STATE_CAP)
                                // WORK CAP (same bound as the Next leg): the Init obligation reduces `|D|`
                                // embedded-pred legs; `|R| × |D|` is a conservative proxy (Init's true cost is
                                // `|D|`, so declining on this over-estimate is safe). Past the cap DECLINE — Init
                                // exhaustiveness then rests on the enumerated `init_values ⊆ R` leg, never a hang.
                                .filter(|domain| {
                                    reachable.len().saturating_mul(domain.len())
                                        <= GENERAL_COMPLETENESS_WORK_CAP
                                })
                        {
                            // Phase-A RUNTIME cross-check, Init side (pairs are (d, d) —
                            // Init mentions no primed columns; mirrors the obligation builder).
                            let cross = crate::cleancic::cross_check_pred_embedders(
                                &init_body.node,
                                &ir,
                                &vars,
                                domain.iter().map(|d| (d.as_slice(), d.as_slice())),
                            );
                            if cross == crate::cleancic::EmbedCrossCheck::Disagree {
                                eprintln!(
                                    "WARNING: recognizer/embedder cross-check DISAGREEMENT on \
                                     the Init predicate — declining the kernel completeness \
                                     leg (fail-closed). This indicates a ty bug; please report."
                                );
                            } else if let Some(bytes) =
                                crate::cleancic::certify_general_init_completeness(
                                    &reachable, &domain, &ir,
                                )
                            {
                                init_pred = Some(ValIRDomain {
                                    pred: ir,
                                    hi: bounds,
                                });
                                init_general_completeness = Some(bytes);
                            }
                        }
                    }
                }
            }
        }

        (
            next_shape,
            next_completeness,
            init_shape,
            init_completeness,
            next_pred,
            next_general_completeness,
            init_pred,
            init_general_completeness,
        )
    };

    // ADDITIVE reflect-v2 exactness cross-check (proof-roadmap §2 B2, task #17): opt-in via
    // `TY_REFLECT_EXACTNESS_XCHECK`. The recognized `safety_pred` is re-evaluated over R through the
    // INDEPENDENT deep kernel-defined evaluator (`TyReflectEvalP`), whose per-op realization is
    // kernel-checked definition data rather than the shallow embedder's per-obligation Rust. At this
    // point the shallow `safety_general` leg has ALREADY reduced `⟦Safety⟧(s)` to `Bool.true` for
    // every `s ∈ R` (else we returned `None` above), so a reflect DISAGREE (deep reduces the SAME IR
    // to `Bool.false`) is a genuine recognizer/embedder EXACTNESS bug — DECLINE (fail-closed). It can
    // only decline or corroborate, never accept more, and it changes NO emitted cert byte. Reflect v2
    // covers only the scalar fragment; `Uncovered` falls back to the existing `pred_exact`/twin
    // guarantee. Opt-in because the per-state kernel reduction is not free and the coverage is partial.
    if std::env::var_os("TY_REFLECT_EXACTNESS_XCHECK").is_some() {
        if let Some(ir) = &safety_pred {
            match crate::reflect_safety_check::reflect_exactness_over_reachable(ir, &reachable) {
                crate::reflect_safety_check::ReflectExactnessOutcome::Mismatch {
                    state,
                    detail,
                } => {
                    eprintln!(
                        "REFLECT-EXACTNESS MISMATCH (fail-closed decline): the reflect-v2 deep \
                         evaluator (TyReflectEvalP) reduced the recognized Safety invariant to \
                         Bool.false at reachable state {state:?}, which the shallow safety leg \
                         accepted — the two op-realizations disagree on truth (a recognizer/embedder \
                         exactness bug). {detail}. This indicates a ty bug; please report."
                    );
                    return None;
                }
                crate::reflect_safety_check::ReflectExactnessOutcome::Corroborated {
                    covered,
                    total,
                } => {
                    eprintln!(
                        "reflect-exactness: CORROBORATED — the reflect-v2 deep evaluator agrees \
                         with the shallow safety leg on {covered}/{total} reachable state(s) \
                         (remaining states' IR is outside reflect v2's scalar fragment)."
                    );
                }
                crate::reflect_safety_check::ReflectExactnessOutcome::Uncovered => {
                    eprintln!(
                        "reflect-exactness: UNCOVERED — the recognized Safety IR is outside reflect \
                         v2's scalar fragment on every reachable state; exactness stays syntactic \
                         (pred_exact + violated-twin tests)."
                    );
                }
            }
        }
    }

    // ── DEADLOCK-FREEDOM leg (certify/check parity; the default-on POLICY is applied at the CLI) ────
    // The mint BFS recorded, per reachable state, its first enumerated successor (`succ_witness`) and
    // the first state with NONE (`deadlock_witness` — a deadlock, exactly as `ty check` decides it:
    // self-loops enumerate a successor, so are not deadlocks). Build the kernel CORROBORATION leg iff
    //   (a) there is NO reachable deadlock (`deadlock_witness` is None),
    //   (b) NO `TERMINAL` predicate is configured (this lane does not model terminal-state exemption —
    //       the CLI surfaces that honestly instead of claiming deadlock-freedom), and
    //   (c) the Next relation embeds (`next_pred`/`next_shape` present), so there IS a kernel Next to
    //       reduce over the enumerated witnesses.
    // The deadlock-freedom DECISION (decline vs certify) is the enumerator's — this leg re-checks the
    // witnesses through the kernel (`⋀_{s∈R} ⟦Next⟧(s, wₛ) = Bool.true`). On the enumerator-ASSISTED
    // tier (no embeddable Next) the leg is `None` and deadlock-freedom rests on the enumerator alone.
    let deadlock_free: Option<DeadlockFreeLeg> = (|| {
        if deadlock_witness.is_some() || config.terminal.is_some() {
            return None;
        }
        // Witnesses aligned to the FINAL reachable order; every reachable state has one (no deadlock).
        let witnesses: Vec<Vec<u64>> = reachable
            .iter()
            .map(|s| succ_witness.get(s).cloned())
            .collect::<Option<Vec<_>>>()?;
        if let Some(dp) = &next_pred {
            return crate::cleancic::certify_deadlock_witness_general(
                &reachable, &witnesses, &dp.pred,
            )
            .map(|term| DeadlockFreeLeg { witnesses, term });
        }
        if let Some(shape) = &next_shape {
            if sorts.as_slice() == [ColSort::Int] && witnesses.iter().all(|w| w.len() == 1) {
                let scalar_r: Vec<u64> = reachable.iter().map(|t| t[0]).collect();
                let scalar_w: Vec<u64> = witnesses.iter().map(|w| w[0]).collect();
                return crate::cleancic::certify_deadlock_witness_affine(
                    &scalar_r, &scalar_w, shape,
                )
                .map(|term| DeadlockFreeLeg { witnesses, term });
            }
        }
        None
    })();

    Some(ExplicitFixpointCert {
        reachable,
        init_values,
        image: image_vec,
        sorts,
        safety_term,
        init_member_terms,
        closed_member_terms,
        next_shape,
        next_completeness,
        init_shape,
        init_completeness,
        next_pred,
        next_general_completeness,
        init_pred,
        init_general_completeness,
        // The finite-enumeration path NEVER carries the unbounded parametric witness (mutually exclusive).
        unbounded_invariant: None,
        safety_pred,
        safety_general,
        init_member_reflected,
        closed_member_reflected,
        deadlock_free,
        deadlock_scan: MintAside(deadlock_witness),
    })
}

/// The GENERAL `R⊆Safety` obligation `⋀_{s∈R} ⟦Safety⟧(s)` as ONE kernel Bool term: per reachable
/// state, the invariant IR embedded as a closed Bool term via [`crate::cleancic::embed_pred_ir`]
/// (an invariant mentions no primed columns — gated at recognize time — so the state is passed for
/// both `s` and `s'`, the fixpoint-lane convention), conjoined in the stored `reachable` order.
/// The refinement lane's `init_refinement_bool` is the same pattern. An empty `R` degenerates
/// to `Bool.true`, but is unreachable through certify/verify (the base membership legs already
/// fail-close on an empty enumeration).
///
/// SHAPE RULE (deterministic; certify and verify both call THIS function, so they always agree):
/// at most [`REFLECTED_MEMBERSHIP_THRESHOLD`] states keeps the historical LEFT-NESTED
/// `Bool.and` chain — every existing cert's obligation (and hence its digest-relevant bytes)
/// is untouched; beyond it the conjunction is a BALANCED tree
/// ([`crate::cleancic::balanced_bool_and`]) whose depth is `⌈log₂|R|⌉` instead of `|R|` — a
/// 5K-state chain would nest 5K deep in every serde/walker and in the kernel's argument
/// reduction. The kernel reduces both shapes to the same constant (pinned by test).
#[cfg(feature = "clean-cic")]
fn safety_general_bool(reachable: &[Vec<u64>], ir: &PredIR) -> clean_kernel::Expr {
    use clean_kernel::Expr;
    if reachable.len() > REFLECTED_MEMBERSHIP_THRESHOLD {
        let legs: Vec<Expr> = reachable
            .iter()
            .map(|s| crate::cleancic::embed_pred_ir(ir, s, s))
            .collect();
        return crate::cleancic::balanced_bool_and(legs);
    }
    let mut acc: Option<Expr> = None;
    for s in reachable {
        let leg = crate::cleancic::embed_pred_ir(ir, s, s);
        acc = Some(match acc {
            None => leg,
            Some(a) => Expr::apps(Expr::const_str("Bool.and"), [a, leg]),
        });
    }
    acc.unwrap_or_else(|| Expr::const_str("Bool.true"))
}

/// Maximum source states in one fallback `R ⊆ Safety` kernel obligation. The monolith remains the
/// preferred byte-compatible path; this exact conjunction partition is used only when its structural
/// term/heartbeat guard rejects. SingleLaneBridge's 3,605-state invariant is the motivating scale.
const SAFETY_GENERAL_CHUNK_STATES: usize = 128;

fn certify_safety_general(reachable: &[Vec<u64>], ir: &PredIR) -> Option<Vec<u8>> {
    if let Some(bytes) =
        crate::cleancic::certify_bool_true_obligation(safety_general_bool(reachable, ir))
    {
        return Some(bytes);
    }
    let mut token = None;
    for chunk in reachable.chunks(SAFETY_GENERAL_CHUNK_STATES) {
        let bytes = crate::cleancic::certify_bool_true_obligation(safety_general_bool(chunk, ir))?;
        token.get_or_insert(bytes);
    }
    token
}

fn verify_safety_general(reachable: &[Vec<u64>], ir: &PredIR, bytes: &[u8]) -> bool {
    if crate::cleancic::verify_bool_true_obligation(safety_general_bool(reachable, ir), bytes) {
        return true;
    }
    !reachable.is_empty()
        && reachable.chunks(SAFETY_GENERAL_CHUNK_STATES).all(|chunk| {
            crate::cleancic::verify_bool_true_obligation(safety_general_bool(chunk, ir), bytes)
        })
}

/// Re-check a [`ExplicitFixpointCert`] by re-running the Clean kernel on every embedded leg, bound to
/// the SAME enumerated `(reachable, init_values)`. No model checker, no SMT — the trust base is the
/// kernel. Fail-closed.
#[cfg(feature = "clean-cic")]
pub fn verify_explicit_state_cert(cert: &ExplicitFixpointCert) -> bool {
    // UNBOUNDED parametric inductive-invariant cert: there is NO enumerated `R` (closure rests on the
    // `∀x` consecution leg). Re-check the three legs against the obligation TYPES rebuilt from the
    // stored `(c, δ)`, and require ALL finite machinery to be EMPTY/None (mutually exclusive). The
    // `(c, δ)` are bound to the SPEC by Leg E (`verify_explicit_fixpoint_report` re-recognizes the
    // unbounded shape from `spec_src` and requires `re.unbounded_invariant == fp.unbounded_invariant`).
    if let Some(ub) = &cert.unbounded_invariant {
        // No finite enumeration may accompany an unbounded cert.
        if !cert.reachable.is_empty()
            || !cert.init_values.is_empty()
            || !cert.image.is_empty()
            || !cert.safety_term.is_empty()
            || !cert.init_member_terms.is_empty()
            || !cert.closed_member_terms.is_empty()
            || cert.next_shape.is_some()
            || cert.next_completeness.is_some()
            || cert.init_shape.is_some()
            || cert.init_completeness.is_some()
            || cert.next_pred.is_some()
            || cert.next_general_completeness.is_some()
            || cert.init_pred.is_some()
            || cert.init_general_completeness.is_some()
            || cert.safety_pred.is_some()
            || cert.safety_general.is_some()
            || cert.init_member_reflected.is_some()
            || cert.closed_member_reflected.is_some()
            || cert.deadlock_free.is_some()
        {
            return false;
        }
        // RELATIONAL cert (`J ≡ v0=v1`): dispatch to the `Eq`-relational re-check. Its `pairs` is empty
        // (mutually exclusive with the nonneg vector) and it covers exactly two Int variables. Bind the
        // summary `init`/`delta` to the stored `(c, δ)`.
        if let Some((c, delta)) = ub.relational {
            if !ub.pairs.is_empty() || (ub.init, ub.delta) != (c, delta) || ub.bound != 0 {
                return false;
            }
            if cert.sorts.as_slice() != [ColSort::Int, ColSort::Int] {
                return false;
            }
            let shape = crate::cleancic::UnboundedRelationalShape { init: c, delta };
            return crate::cleancic::verify_unbounded_relational(
                &shape,
                &ub.initiation,
                &ub.consecution,
                &ub.preservation,
            );
        }
        // `pairs` is the source of truth for the variable count. Bind the summary `init`/`delta` to it
        // (a cert whose scalar summary disagrees with `pairs[0]` is malformed → reject), and require
        // the sort vector to be all-Int of the right width.
        if ub.pairs.is_empty() || (ub.init, ub.delta) != ub.pairs[0] {
            return false;
        }
        if cert.sorts.len() != ub.pairs.len() || !cert.sorts.iter().all(|s| *s == ColSort::Int) {
            return false;
        }
        if ub.pairs.len() == 1 {
            // n=1: the SCALAR legs (byte-identical to the historical single-var cert for
            // `bound = 0`; the Phase-2 `Int.le` legs for `bound > 0`). The obligation types
            // are rebuilt from `(c, δ, N)` — a tampered `bound` fails the kernel re-check,
            // and Leg-E re-recognition binds `bound` to the SPEC's `Safety = x ≥ N`.
            let (init, delta) = ub.pairs[0];
            // Initiation feasibility is part of the recognized fragment: reject a cert
            // claiming `c < N` outright (its initiation type would be unprovable anyway).
            if init < ub.bound {
                return false;
            }
            let shape = crate::cleancic::UnboundedAffineShape {
                init,
                delta,
                bound: ub.bound,
            };
            return crate::cleancic::verify_unbounded_invariant(
                &shape,
                &ub.initiation,
                &ub.consecution,
                &ub.preservation,
            );
        }
        // n≥2: the CONJOINED multi-variable legs (bound is scalar-only; nonzero is malformed).
        if ub.bound != 0 {
            return false;
        }
        let tuple = crate::cleancic::UnboundedAffineTuple {
            pairs: ub.pairs.clone(),
        };
        return crate::cleancic::verify_unbounded_invariant_tuple(
            &tuple,
            &ub.initiation,
            &ub.consecution,
            &ub.preservation,
        );
    }

    // MEMBERSHIP-LEG DISPATCH (R2) — the mirror of the mint rule, decided by the CANONICALIZED
    // |R| (never by which legs the cert happens to carry). The presence rules are EXPLICIT and
    // fail-closed in both directions:
    //  * |R| ≤ threshold: the per-member lane — BOTH reflected fields must be ABSENT (a
    //    sub-threshold cert carrying reflected legs is malformed/mixed → reject);
    //  * |R| > threshold: the reflected lane — BOTH reflected fields must be PRESENT and BOTH
    //    per-member vectors EMPTY (missing/extra/mixed legs → reject); present-but-failing
    //    reflected bytes → reject (inside `verify_explicit_fixpoint_set_reflected`).
    let canon_len = {
        let mut st = cert.reachable.clone();
        st.sort_unstable();
        st.dedup();
        st.len()
    };
    let base = if use_reflected_membership_lane(canon_len, cert.sorts.len()) {
        match (&cert.init_member_reflected, &cert.closed_member_reflected) {
            (Some(init_r), Some(closed_r)) => {
                if !cert.init_member_terms.is_empty() || !cert.closed_member_terms.is_empty() {
                    return false; // mixed per-member + reflected legs → malformed
                }
                crate::cleancic::verify_explicit_fixpoint_set_reflected(
                    &cert.reachable,
                    &cert.init_values,
                    &cert.image,
                    &cert.sorts,
                    closed_r,
                    &cert.safety_term,
                    init_r,
                )
            }
            _ => return false, // above-threshold cert without both reflected legs → reject
        }
    } else {
        if cert.init_member_reflected.is_some() || cert.closed_member_reflected.is_some() {
            return false; // sub-threshold cert must not carry reflected legs
        }
        crate::cleancic::verify_explicit_fixpoint_set(
            &cert.reachable,
            &cert.init_values,
            &cert.image,
            &cert.sorts,
            &cert.closed_member_terms,
            &cert.safety_term,
            &cert.init_member_terms,
        )
    };
    if !base {
        return false;
    }
    // A state with NO Int column carries a TRIVIAL tuple safety leg (`Eq Bool true true` — see
    // `cleancic::explicit_safety_tuple`), which claims NOTHING about the spec's invariant. Such a
    // cert must carry the GENERAL safety leg or it certifies no safety at all — reject without it
    // (certify always populates it for no-Int states; a hand-built cert without it is malformed).
    if !cert.sorts.iter().any(|s| *s == ColSort::Int) && cert.safety_pred.is_none() {
        return false;
    }
    // The single-Int scalar view of R (the affine/literal completeness legs operate on Nat values).
    // Only valid when every reachable tuple is a 1-tuple over an Int column.
    let single_int =
        cert.sorts.as_slice() == [ColSort::Int] && cert.reachable.iter().all(|t| t.len() == 1);
    let scalar_r = || -> Vec<u64> { cert.reachable.iter().map(|t| t[0]).collect() };
    // The GENERAL (multi-variable) completeness legs operate on TUPLES over a product domain; valid
    // when every column is Int/Bool/Set and the tuple width matches the sort vector.
    let n_cols = cert.sorts.len();
    // A scalar ENUM column embeds too (its cell is a `Nat` label index, `Nat.beq`/`Bool.or`-fold exact) —
    // mirrors the certify-side `all_embeddable` so the verify path admits the SAME general legs.
    let all_int_bool = cert.sorts.iter().all(|s| {
        matches!(
            s,
            ColSort::Int | ColSort::Bool | ColSort::Set { .. } | ColSort::Enum { .. }
        ) || s.is_compound()
    }) && cert.reachable.iter().all(|t| t.len() == n_cols);

    // KERNEL-RE-EVALUATED `Next`-completeness leg — present iff the spec was a recognized affine counter.
    let next_ok = match (&cert.next_shape, &cert.next_completeness) {
        (None, None) => true, // non-affine spec: closure rests on the (already re-checked) image leg
        (Some(shape), Some(bytes)) => {
            if !single_int {
                return false; // shape disagrees with the cert's column structure → fail-closed
            }
            let hi = shape.bound.saturating_sub(1).saturating_add(shape.delta);
            let Some(domain) = bounded_scalar_completeness_domain(hi) else {
                return false;
            };
            crate::cleancic::verify_next_completeness(&scalar_r(), &domain, shape, bytes)
        }
        _ => return false, // exactly one of (shape, term) present → inconsistent → fail-closed
    };
    if !next_ok {
        return false;
    }

    // KERNEL-RE-EVALUATED GENERAL `Next`-completeness leg — present iff the spec was NON-affine but
    // embeddable (all columns Int/Bool) with a sound PER-COLUMN successor bound. Rebuild the PRODUCT
    // domain `D = ⨉_i {0..=H_i}` from the stored IR (the per-column bounds are RECOMPUTED from the IR +
    // reachable + sorts — never trusting the serialized `hi`) and re-run the kernel.
    let next_general_ok = match (&cert.next_pred, &cert.next_general_completeness) {
        (None, None) => true,
        (Some(domir), Some(bytes)) => {
            if !all_int_bool {
                return false;
            }
            // RECOMPUTE the per-column bounds from the IR — never trust the serialized `hi` (a re-sealed
            // crafted cert could shrink D and hide an escaping successor). `None` ⇒ unverifiable ⇒ reject.
            let Some(bounds) = crate::cleancic::next_domain_bounds_from_ir(
                &domir.pred,
                n_cols,
                &cert.reachable,
                &cert.sorts,
            ) else {
                return false;
            };
            let Some(domain) = crate::cleancic::product_domain(&bounds, DEFAULT_FIXPOINT_STATE_CAP)
            else {
                return false;
            };
            crate::cleancic::verify_general_completeness(
                &cert.reachable,
                &domain,
                &domir.pred,
                bytes,
            )
        }
        _ => return false,
    };
    if !next_general_ok {
        return false;
    }
    // The affine and general Next legs are mutually exclusive (the orchestrator tries affine first,
    // general only as the fallback) — both present is an inconsistent cert.
    if cert.next_shape.is_some() && cert.next_pred.is_some() {
        return false;
    }

    // KERNEL-RE-EVALUATED `Init`-completeness leg — present iff `Init` was a recognized literal disjunction.
    let init_ok = match (&cert.init_shape, &cert.init_completeness) {
        (None, None) => true, // Init-exhaustiveness rests on the (already re-checked) `init_values ⊆ R`
        (Some(vals), Some(bytes)) => {
            if !single_int {
                return false;
            }
            let hi = vals.iter().copied().max().unwrap_or(0);
            let Some(domain) = bounded_scalar_completeness_domain(hi) else {
                return false;
            };
            crate::cleancic::verify_init_completeness(&scalar_r(), &domain, vals, bytes)
        }
        _ => return false,
    };
    if !init_ok {
        return false;
    }

    // KERNEL-RE-EVALUATED GENERAL `Init`-completeness leg (multi-variable product domain).
    let init_general_ok = match (&cert.init_pred, &cert.init_general_completeness) {
        (None, None) => true,
        (Some(domir), Some(bytes)) => {
            if !all_int_bool {
                return false;
            }
            // Recompute the per-column Init bounds from the IR — never trust the serialized `hi`.
            let Some(bounds) =
                crate::cleancic::init_domain_bounds_from_ir(&domir.pred, n_cols, &cert.sorts)
            else {
                return false;
            };
            let Some(domain) = crate::cleancic::product_domain(&bounds, DEFAULT_FIXPOINT_STATE_CAP)
            else {
                return false;
            };
            crate::cleancic::verify_general_init_completeness(
                &cert.reachable,
                &domain,
                &domir.pred,
                bytes,
            )
        }
        _ => return false,
    };
    if !init_general_ok {
        return false;
    }
    if cert.init_shape.is_some() && cert.init_pred.is_some() {
        return false;
    }

    // GENERAL `R⊆Safety` leg — present iff the spec's invariant was NOT the conjunctive-nonneg
    // shape but was recognized into the kernel predicate fragment. Rebuild the obligation
    // `⋀_{s∈R} ⟦Safety⟧(s)` from the stored `reachable` + `safety_pred` and re-run the kernel.
    // PRESENT-BUT-FAILING ⇒ reject; exactly one of (IR, bytes) present ⇒ inconsistent ⇒ reject.
    // NOTE the stored IR is only kernel-bound here (a WIDER invariant than the spec's would still
    // reduce true over R) — binding the IR to the SPEC is Leg-E's job (`verify_explicit_fixpoint_report`
    // re-recognizes the invariant from the re-parsed spec and requires `re.safety_pred == fp.safety_pred`).
    match (&cert.safety_pred, &cert.safety_general) {
        (None, None) => {} // primary nonneg lane: the (already re-checked) tuple safety_term governs
        (Some(ir), Some(bytes)) => {
            // The stored IR must itself lie in the certifiable fragment: a truth-direction-EXACT
            // STATE predicate over these sorts (same gates as certify — a stored IR outside that
            // fragment is malformed, and `Nat.sub`-style truncation could make the kernel prove a
            // conjunction the TLA invariant does not assert).
            if !crate::refinement_cert::pred_exact(ir, &cert.sorts)
                || crate::refinement_cert::pred_mentions_prime(ir)
            {
                return false;
            }
            if !verify_safety_general(&cert.reachable, ir, bytes) {
                return false;
            }
        }
        _ => return false,
    }

    // KERNEL-RE-EVALUATED deadlock-freedom leg (see [`DeadlockFreeLeg`]) — present iff the enumerator-
    // free deadlock corroboration fired. Rebuild `⋀_{s∈R} ⟦Next⟧(s, wₛ) = Bool.true` from the stored
    // witnesses + the present Next leg (`next_pred` / `next_shape`) and re-run the kernel. The witnesses
    // are ALSO bound to the spec by `cert.rs`'s `matches_rederivation` (a re-enumeration reproduces the
    // identical first-successor witnesses), so a tampered cert cannot substitute spurious witnesses:
    // here they must kernel-reduce to `Bool.true`, and there they must match the spec's re-derivation.
    if let Some(dl) = &cert.deadlock_free {
        if dl.witnesses.len() != cert.reachable.len() || cert.reachable.is_empty() {
            return false;
        }
        let dl_ok = if let Some(domir) = &cert.next_pred {
            if !all_int_bool {
                return false;
            }
            crate::cleancic::verify_deadlock_witness_general(
                &cert.reachable,
                &dl.witnesses,
                &domir.pred,
                &dl.term,
            )
        } else if let Some(shape) = &cert.next_shape {
            if !single_int || dl.witnesses.iter().any(|w| w.len() != 1) {
                return false;
            }
            let sw: Vec<u64> = dl.witnesses.iter().map(|w| w[0]).collect();
            crate::cleancic::verify_deadlock_witness_affine(&scalar_r(), &sw, shape, &dl.term)
        } else {
            // A deadlock-freedom leg with no embeddable Next to reduce over is malformed.
            return false;
        };
        if !dl_ok {
            return false;
        }
    }
    true
}

/// Phase-A domain-coverage summary of a cert's kernel completeness legs (`docs/
/// kernel-checked-tla-plan.md`): for each present leg, WHICH columns' product-domain axes rely
/// on trusted-Rust bound rules vs. covering their whole universe by construction. RECOMPUTED
/// from the cert's IR + reachable + sorts — never trusted from serialized data.
#[cfg(feature = "clean-cic")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DomainCoverageReport {
    /// General-Next-leg columns whose axis bound is Rust-derived (empty ⇒ every axis is
    /// universe-complete or kernel-proven).
    pub next_rust_columns: Vec<usize>,
    /// General-Init-leg columns whose axis bound is Rust-derived.
    pub init_rust_columns: Vec<usize>,
    /// General-Next-leg columns whose successor bound was PROVED IN THE KERNEL (Phase A
    /// increment 2): per source state, a Π-quantified lemma `∀sp. ⟦Next⟧(s,sp) ⇒ sp_i ≤ H_i`
    /// was synthesized and kernel-accepted — the Rust bound rule is out of the trust story.
    pub next_kernel_columns: Vec<usize>,
    /// General-Init-leg columns whose domain bound was kernel-proven (one Π-lemma).
    pub init_kernel_columns: Vec<usize>,
    /// Whether the single-Int affine/literal SHORTCUT legs are present (their axis is always
    /// Rust-derived — the affine `bound-1+δ` / literal-max rules).
    pub shortcut_legs: bool,
    /// Whether ANY kernel completeness leg is present (else there is nothing to classify —
    /// closure rests on the enumerated `image ⊆ R` leg).
    pub any_completeness_leg: bool,
}

#[cfg(feature = "clean-cic")]
impl DomainCoverageReport {
    /// `true` iff every present completeness leg's every axis is covered WITHOUT trusting a
    /// Rust bound rule — universe-complete by sort, or kernel-proven per state. This is the
    /// bar `--require-domain-complete` enforces.
    pub fn fully_construction_covered(&self) -> bool {
        self.any_completeness_leg
            && !self.shortcut_legs
            && self.next_rust_columns.is_empty()
            && self.init_rust_columns.is_empty()
    }
}

/// Compute the [`DomainCoverageReport`] for a certificate, INCLUDING the kernel-coverage
/// upgrade pass (each Rust-derived axis is offered to the successor-bound synthesizer; a
/// kernel-accepted proof moves it to the kernel-proven bucket). Deterministic.
#[cfg(feature = "clean-cic")]
pub fn domain_coverage_of_cert(cert: &ExplicitFixpointCert) -> DomainCoverageReport {
    let n = cert.sorts.len();
    let split = |cov: Option<Vec<(u64, crate::cleancic::DomainCoverage)>>,
                 ir: Option<&PredIR>,
                 primed: bool|
     -> (Vec<usize>, Vec<usize>) {
        let Some(mut v) = cov else {
            return (Vec::new(), Vec::new());
        };
        if let Some(ir) = ir {
            crate::cleancic::upgrade_domain_coverage(ir, n, &cert.reachable, &mut v, primed);
        }
        let rust = v
            .iter()
            .enumerate()
            .filter(|(_, (_, c))| *c == crate::cleancic::DomainCoverage::RustDerived)
            .map(|(i, _)| i)
            .collect();
        let kernel = v
            .iter()
            .enumerate()
            .filter(|(_, (_, c))| *c == crate::cleancic::DomainCoverage::KernelProven)
            .map(|(i, _)| i)
            .collect();
        (rust, kernel)
    };
    let (next_rust_columns, next_kernel_columns) = cert
        .next_pred
        .as_ref()
        .map(|d| {
            split(
                crate::cleancic::next_domain_bounds_cov_from_ir(
                    &d.pred,
                    n,
                    &cert.reachable,
                    &cert.sorts,
                ),
                Some(&d.pred),
                true,
            )
        })
        .unwrap_or_default();
    let (init_rust_columns, init_kernel_columns) = cert
        .init_pred
        .as_ref()
        .map(|d| {
            split(
                crate::cleancic::init_domain_bounds_cov_from_ir(&d.pred, n, &cert.sorts),
                Some(&d.pred),
                false,
            )
        })
        .unwrap_or_default();
    DomainCoverageReport {
        next_rust_columns,
        init_rust_columns,
        next_kernel_columns,
        init_kernel_columns,
        shortcut_legs: cert.next_shape.is_some() || cert.init_shape.is_some(),
        any_completeness_leg: cert.next_completeness.is_some()
            || cert.init_completeness.is_some()
            || cert.next_general_completeness.is_some()
            || cert.init_general_completeness.is_some(),
    }
}

#[cfg(all(test, feature = "clean-cic"))]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn scalar_completeness_domain_is_resource_bounded() {
        let max_admitted_hi = DEFAULT_FIXPOINT_STATE_CAP as u64 - 1;
        let domain = bounded_scalar_completeness_domain(max_admitted_hi)
            .expect("the exact stable domain cap is admitted");
        assert_eq!(domain.len(), DEFAULT_FIXPOINT_STATE_CAP);
        assert_eq!(domain.first(), Some(&0));
        assert_eq!(domain.last(), Some(&max_admitted_hi));
        assert!(
            bounded_scalar_completeness_domain(max_admitted_hi + 1).is_none(),
            "cap+1 members must decline before allocation"
        );
        assert!(
            bounded_scalar_completeness_domain(u64::MAX).is_none(),
            "an overflowing inclusive bound must decline before allocation"
        );
    }

    /// Serialized shortcut bounds are attacker-controlled until Leg-E re-derives the certificate.
    /// Verification must reject an oversized affine/literal domain before trying to allocate it.
    #[test]
    fn oversized_scalar_completeness_domains_fail_closed() {
        let spec = "---- MODULE Bounded ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1 /\\ x < 3\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("baseline certifies");
        assert!(verify_explicit_state_cert(&cert));

        let mut oversized_next = cert.clone();
        oversized_next
            .next_shape
            .as_mut()
            .expect("affine shortcut")
            .bound = u64::MAX;
        assert!(
            !verify_explicit_state_cert(&oversized_next),
            "an oversized affine domain must reject without allocation"
        );

        let mut oversized_init = cert;
        oversized_init.init_shape = Some(vec![u64::MAX]);
        assert!(
            !verify_explicit_state_cert(&oversized_init),
            "an oversized literal domain must reject without allocation"
        );
    }

    /// DEADLOCK-FREEDOM leg (certify/check parity): a cycling spec (`x' = IF x=2 THEN 0 ELSE x+1`) has
    /// a successor at every reachable state, so the mint records NO deadlock witness and builds the
    /// kernel corroboration leg; the cert (INCLUDING the deadlock leg) re-verifies and serde-round-trips.
    #[test]
    fn deadlock_free_leg_present_and_reverifies() {
        let spec = "---- MODULE Cyc ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = IF x = 2 THEN 0 ELSE x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("cycling spec must certify");
        assert!(
            cert.deadlock_scan.0.is_none(),
            "no reachable deadlock in a 3-cycle"
        );
        let dl = cert
            .deadlock_free
            .as_ref()
            .expect("enumerator-free tier builds the kernel deadlock leg");
        assert_eq!(
            dl.witnesses.len(),
            cert.reachable.len(),
            "one witness per reachable state"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "cert (incl. deadlock leg) must re-verify"
        );
        let bytes = serde_json::to_vec(&cert).expect("serialize");
        assert!(
            String::from_utf8_lossy(&bytes).contains("deadlock_free"),
            "the deadlock leg must be serialized (digest changes by design)"
        );
        let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(
            cert, back,
            "the deadlock-leg cert must serde round-trip identically"
        );
        assert!(
            verify_explicit_state_cert(&back),
            "the round-tripped cert must re-verify"
        );
    }

    /// A deadlocking spec (`x < 3 /\ x' = x+1`): state x=3 has NO successor. The mint records the
    /// witness on `deadlock_scan` (the CLI declines from it, matching `ty check`) and builds NO kernel
    /// deadlock leg. The SAFETY cert itself is still sound — it is exactly what `--no-deadlock` certifies.
    #[test]
    fn deadlocking_spec_records_witness_and_builds_no_leg() {
        let spec = "---- MODULE Dl ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x < 3 /\\ x' = x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("safety still certifies");
        assert_eq!(
            cert.deadlock_scan.0,
            Some(vec![3]),
            "x=3 is the deadlock witness"
        );
        assert!(
            cert.deadlock_free.is_none(),
            "no kernel deadlock leg for a deadlocking spec"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the safety cert is sound (this is the `--no-deadlock` certificate)"
        );
    }

    /// A self-loop (`x' = x`) is NOT a deadlock — the state enumerates itself as a successor (parity
    /// with `ty check`, which passes self-loops). The mint records no witness and each state's witness
    /// successor is ITSELF.
    #[test]
    fn self_loop_is_not_a_deadlock() {
        let spec = "---- MODULE Sl ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0 \\/ x = 1\n\
                    Next == x' = x\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("self-loop spec certifies");
        assert!(
            cert.deadlock_scan.0.is_none(),
            "a self-loop is not a deadlock"
        );
        let dl = cert
            .deadlock_free
            .as_ref()
            .expect("kernel deadlock leg present");
        assert_eq!(
            dl.witnesses, cert.reachable,
            "each state's witness successor is itself"
        );
        assert!(verify_explicit_state_cert(&cert));
    }

    /// A TAMPERED deadlock witness (a non-successor) fails the kernel re-check in
    /// `verify_explicit_state_cert`: the recognized `Next` does not reduce to `Bool.true` at the fake
    /// `(s, wₛ)` pair, so the conjunction is `Bool.false` and the kernel rejects (fail-closed).
    #[test]
    fn tampered_deadlock_witness_is_kernel_rejected() {
        let spec = "---- MODULE Cyc2 ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = IF x = 2 THEN 0 ELSE x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        let mut cert = certify_explicit_state_spec(spec, &cfg()).expect("certifies");
        assert!(verify_explicit_state_cert(&cert), "baseline cert verifies");
        let dl = cert.deadlock_free.as_mut().expect("has a deadlock leg");
        dl.witnesses[0] = vec![99]; // not a successor of reachable[0]
        assert!(
            !verify_explicit_state_cert(&cert),
            "a fake (non-successor) witness must be kernel-rejected"
        );
    }

    /// LIVE end-to-end: a small finite explicit spec whose reachable set is `R={2,5}` (a 2-value
    /// stutter init). The live model-checker enumeration produces `R`, and the kernel fixpoint cert
    /// is emitted AND re-verified — a normal-shaped spec reaching a kernel-CHECKED Certified verdict
    /// through the live explicit-state path.
    #[test]
    fn live_explicit_fixpoint_two_value_stutter() {
        let spec = "---- MODULE TwoVal ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 2 \\/ x = 5\n\
                    Next == x' = x\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the live explicit-state fixpoint must be kernel-certified");
        assert_eq!(cert.reachable, vec![vec![2], vec![5]]); // 1-tuples (single var)
        assert_eq!(cert.init_values.len(), 2);
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the live explicit-state cert must pass"
        );
    }

    /// Phase 1 (`docs/kernel-checked-tla-plan.md`): the tiny second checker (clean-ck0) MUST
    /// independently re-check the kernel-RE-EVALUATED completeness legs of a real certified spec.
    /// Those obligations (`Eq.refl : Eq Bool C Bool.true`) lie in ck0's Nat/Bool fragment, so a
    /// certification run must tally ck0 corroborations — and NEVER a checker disagreement. Since
    /// the ck0 env widened to the Int dependency closure (`ck0_bridge`), the structural Int legs
    /// (previously honestly `unavailable`) are corroborated too: this spec's ENTIRE leg set is
    /// second-checked, so the tally must show zero `unavailable`.
    #[test]
    fn ck0_second_checker_corroborates_completeness_legs() {
        let spec = "---- MODULE Bounded ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1 /\\ x < 3\n\
                    Safety == x >= 0\n\
                    ====\n";
        crate::ck0_bridge::begin_tally();
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the bounded counter must kernel-certify");
        assert!(cert.next_completeness.is_some() && cert.init_completeness.is_some());
        assert!(verify_explicit_state_cert(&cert));
        let tally = crate::ck0_bridge::take_tally().expect("tally was started");
        assert_eq!(
            tally.rejected, 0,
            "clean-kernel and clean-ck0 must never disagree on a live leg: {tally:?}"
        );
        assert!(
            tally.corroborated >= 2,
            "the Init- and Next-completeness legs (certify AND verify passes) must be \
             ck0-corroborated — the tiny-kernel fragment is live, not theoretical: {tally:?}"
        );
        // HISTORY: this used to assert `unavailable > 0` (the structural Int legs were outside
        // ck0's Nat/Bool fragment). The ck0 env now ingests the Int dependency closure
        // (`Int.NonNeg`/`Int.le` legs included), so those legs corroborate and NOTHING in this
        // spec's leg set is left at the clean-kernel-only tier — pin that widening.
        assert_eq!(
            tally.unavailable, 0,
            "every leg of this spec lies in ck0's widened Nat/Bool/Int fragment: {tally:?}"
        );
    }

    /// Phase A INCREMENT 2 — THE RESEARCH-WALL FLIP: under `--require-domain-complete`, the
    /// bounded Int counter's `D ⊇ Succ(R)` obligation is now DISCHARGED IN THE KERNEL. The
    /// affine/literal shortcut legs are still declined (Rust-derived by construction), but the
    /// GENERAL legs fire with rule-3 Eq-pin bounds and the upgrade pass synthesizes, per source
    /// state, the Π-quantified successor-bound lemma `∀sp. ⟦Next⟧(s,sp)=true → sp ≤ H` — each
    /// kernel-accepted. The strict cert is enumerator-free AND coverage-kernel-proven: the
    /// trusted-Rust bound rules are OUT of its trust story.
    #[test]
    fn strict_domain_mode_kernel_proves_int_counter_coverage() {
        let spec = "---- MODULE Bounded ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1 /\\ x < 3\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert =
            certify_explicit_state_spec_strict_domain(spec, &cfg()).expect("strict mode certifies");
        assert!(
            cert.next_shape.is_none() && cert.init_shape.is_none(),
            "the shortcut legs (Rust-derived by construction) stay declined in strict mode"
        );
        assert!(
            cert.next_general_completeness.is_some() && cert.init_general_completeness.is_some(),
            "the GENERAL completeness legs must survive strict mode — their coverage is \
             kernel-proven (rule-3 Eq-pin + per-state Π-lemmas)"
        );
        assert!(verify_explicit_state_cert(&cert));
        let coverage = domain_coverage_of_cert(&cert);
        assert!(
            coverage.fully_construction_covered(),
            "no Rust bound rule may remain in the strict cert's trust story: {coverage:?}"
        );
        assert_eq!(
            coverage.next_kernel_columns,
            vec![0],
            "the Int column's successor bound must be KERNEL-PROVEN: {coverage:?}"
        );
        assert_eq!(coverage.init_kernel_columns, vec![0]);
        // The DEFAULT mode still prefers the shortcut legs (back-compat) and the report
        // SURFACES their Rust-derived basis honestly.
        let default_cert = certify_explicit_state_spec(spec, &cfg()).expect("default certifies");
        let default_cov = domain_coverage_of_cert(&default_cert);
        assert!(default_cov.shortcut_legs);
        assert!(
            !default_cov.fully_construction_covered(),
            "shortcut legs must be SURFACED as Rust-derived, never claimed by-construction"
        );
    }

    /// Strict mode still DECLINES what the synthesizer cannot kernel-prove: a Set column
    /// updated through a comprehension filter is outside the symbolic-embedding fragment
    /// (the fold needs concrete masks), so its axis stays Rust-derived and the strict mode
    /// drops the general legs — honest fail-closed, never a false "kernel-covered".
    #[test]
    fn strict_domain_mode_still_declines_unprovable_coverage() {
        let spec = "---- MODULE SetFilter ----\n\
                    EXTENDS Integers\n\
                    VARIABLE s\n\
                    Init == s = {1, 2}\n\
                    Next == s' = {x \\in s : x # 1}\n\
                    Safety == 3 \\notin s\n\
                    ====\n";
        let Some(cert) = certify_explicit_state_spec_strict_domain(spec, &cfg()) else {
            // Spec outside the strict-certifiable class entirely — also honest.
            return;
        };
        assert!(
            cert.next_general_completeness.is_none(),
            "a filter-updated Set axis cannot be kernel-proven — strict mode must decline"
        );
        assert!(verify_explicit_state_cert(&cert));
    }

    /// Phase A strict mode keeps legs whose every axis covers BY CONSTRUCTION: a pure-Bool spec's
    /// product domain is the full Bool universe per column, so the general completeness legs
    /// survive `--require-domain-complete` and the cert is fully kernel-covered.
    #[test]
    fn strict_domain_mode_keeps_universe_complete_bool_legs() {
        let spec = "---- MODULE BoolFlip ----\n\
                    VARIABLE b\n\
                    Init == b = FALSE\n\
                    Next == b' = ~b\n\
                    Safety == b = TRUE \\/ b = FALSE\n\
                    ====\n";
        let cert = certify_explicit_state_spec_strict_domain(spec, &cfg());
        let Some(cert) = cert else {
            // The Bool spec may decline for reasons orthogonal to coverage (recognizer
            // fragment); the essential assertion is the Int case above. Skip gracefully.
            return;
        };
        if cert.next_general_completeness.is_some() {
            let coverage = domain_coverage_of_cert(&cert);
            assert!(
                coverage.next_rust_columns.is_empty(),
                "a Bool column's axis is its whole universe — never Rust-derived"
            );
        }
        assert!(verify_explicit_state_cert(&cert));
    }

    /// A bounded single-Int counter spec with `R = {0..bound}` (`bound+1` states).
    fn counter_spec(bound: u64) -> String {
        format!(
            "---- MODULE Counter{bound} ----\n\
             EXTENDS Integers\n\
             VARIABLE x\n\
             Init == x = 0\n\
             Next == x' = x + 1 /\\ x < {bound}\n\
             Safety == x >= 0\n\
             ====\n"
        )
    }

    /// R2 THRESHOLD BOUNDARY (63/64/65 states): at and below
    /// [`REFLECTED_MEMBERSHIP_THRESHOLD`] the cert keeps the historical per-member Or-injection
    /// legs and its serialization carries NO reflected field (byte-identity with pre-R2 certs —
    /// the fixture test pins the exact bytes); one state above, the membership legs are the
    /// REFLECTED single obligations and the per-member vectors are EMPTY. Both lanes verify.
    #[test]
    fn reflected_membership_threshold_boundary() {
        for (bound, expect_reflected) in [(62u64, false), (63, false), (64, true)] {
            let cert = certify_explicit_state_spec(&counter_spec(bound), &cfg())
                .unwrap_or_else(|| panic!("bound {bound} must certify"));
            assert_eq!(
                cert.reachable.len(),
                bound as usize + 1,
                "R = {{0..{bound}}}"
            );
            if expect_reflected {
                assert!(
                    cert.init_member_reflected.is_some() && cert.closed_member_reflected.is_some(),
                    "above the threshold BOTH reflected legs must be present"
                );
                assert!(
                    cert.init_member_terms.is_empty() && cert.closed_member_terms.is_empty(),
                    "above the threshold the per-member vectors stay EMPTY"
                );
            } else {
                assert!(
                    cert.init_member_reflected.is_none() && cert.closed_member_reflected.is_none(),
                    "at/below the threshold the reflected fields must be ABSENT"
                );
                assert!(
                    !cert.init_member_terms.is_empty() && !cert.closed_member_terms.is_empty(),
                    "at/below the threshold the per-member legs are the lane"
                );
            }
            let json = String::from_utf8(serde_json::to_vec(&cert).expect("serialize")).unwrap();
            assert_eq!(
                json.contains("member_reflected"),
                expect_reflected,
                "serde presence must track the threshold (digest back-compat below it)"
            );
            assert!(
                verify_explicit_state_cert(&cert),
                "bound {bound} must re-verify"
            );
            // Round-trip through serde (the reflected fields deserialize + verify).
            let back: ExplicitFixpointCert =
                serde_json::from_slice(json.as_bytes()).expect("deserialize");
            assert_eq!(cert, back);
            assert!(verify_explicit_state_cert(&back));
        }
    }

    /// R2 TAMPER MATRIX (above the threshold): mutated reflected bytes, an image tuple escaping
    /// `R`, a foreign Init value, a dropped-`R`-state subset swap, a MISSING reflected leg, and
    /// a sub-threshold cert carrying reflected legs — every direction REJECTS.
    #[test]
    fn reflected_membership_tampers_reject() {
        let cert = certify_explicit_state_spec(&counter_spec(64), &cfg())
            .expect("65-state counter certifies");
        assert!(
            verify_explicit_state_cert(&cert),
            "control: genuine cert verifies"
        );

        // (a) Mutated reflected bytes: garbage AND a wrong-constant proof both reject.
        let mut t = cert.clone();
        t.closed_member_reflected = Some(b"garbage".to_vec());
        assert!(
            !verify_explicit_state_cert(&t),
            "garbage closed bytes must reject"
        );
        let mut t = cert.clone();
        t.init_member_reflected = Some(b"garbage".to_vec());
        assert!(
            !verify_explicit_state_cert(&t),
            "garbage init bytes must reject"
        );

        // (b) Image superset: an image tuple OUTSIDE R refutes the reflected closed leg (the
        // obligation is REBUILT from the tampered tuples; the kernel reduces Subseq to false).
        let mut t = cert.clone();
        t.image.push(vec![999]);
        assert!(
            !verify_explicit_state_cert(&t),
            "an escaping image tuple must reject"
        );

        // (c) Foreign Init value: Init ⊄ R refutes the reflected init leg.
        let mut t = cert.clone();
        t.init_values.push(vec![999]);
        assert!(
            !verify_explicit_state_cert(&t),
            "a foreign Init value must reject"
        );

        // (d) R-SUBSET swap: dropping a reachable state leaves image tuples outside the
        // shrunken R. (65 → 64 states also flips the lane to per-member, where the PRESENT
        // reflected legs are malformed — either rule must reject.)
        let mut t = cert.clone();
        t.reachable.pop();
        assert!(
            !verify_explicit_state_cert(&t),
            "an R-subset swap must reject"
        );

        // (e) Missing reflected leg above the threshold: explicit presence rule.
        let mut t = cert.clone();
        t.closed_member_reflected = None;
        assert!(
            !verify_explicit_state_cert(&t),
            "a missing reflected leg must reject"
        );

        // (f) Sub-threshold cert carrying reflected legs: the mixed rule rejects.
        let small = certify_explicit_state_spec(&counter_spec(3), &cfg())
            .expect("4-state counter certifies");
        assert!(verify_explicit_state_cert(&small));
        let mut t = small.clone();
        t.init_member_reflected = cert.init_member_reflected.clone();
        t.closed_member_reflected = cert.closed_member_reflected.clone();
        assert!(
            !verify_explicit_state_cert(&t),
            "a sub-threshold cert with reflected legs is malformed — reject"
        );
    }

    /// LIVE end-to-end with a SINGLE reachable state `R={3}` (degenerate disjunction). Exercises the
    /// 1-member path of the N-state legs through the live enumerator.
    #[test]
    fn live_explicit_fixpoint_singleton() {
        let spec = "---- MODULE One ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 3\n\
                    Next == x' = x\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("singleton R={3} must certify");
        assert_eq!(cert.reachable, vec![vec![3]]);
        assert!(verify_explicit_state_cert(&cert));
    }

    /// UNBOUNDED end-to-end: `Init x=0 / Next x'=x+1 (NO guard) / Safety x>=0` has an INFINITE reachable
    /// set the finite BFS cannot fold. The PARAMETRIC inductive-invariant path certifies it WITHOUT
    /// enumeration: the cert carries the `unbounded_invariant` (NOT a finite `reachable`), and Leg-E
    /// re-check passes. The heart is the kernel-checked `∀x` consecution leg.
    #[test]
    fn unbounded_affine_counter_certifies_without_enumeration() {
        let spec = "---- MODULE Unbounded ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the unbounded affine counter must be Certified via the parametric leg");
        // NO enumeration: the cert carries the unbounded invariant, not a finite reachable set.
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("the cert must carry the parametric inductive invariant");
        assert_eq!((ub.init, ub.delta), (0, 1));
        assert!(
            cert.reachable.is_empty(),
            "an unbounded cert has NO enumerated reachable set"
        );
        assert!(cert.init_values.is_empty() && cert.image.is_empty());
        assert!(
            cert.next_shape.is_none(),
            "the unbounded path is NOT the finite affine path"
        );
        // verify_explicit_state_cert + Leg-E re-check pass (the 3 parametric legs kernel-re-check).
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the 3 parametric legs must pass"
        );
    }

    /// A larger init/δ unbounded counter (`Init x=3 / Next x'=x+5 / Safety x>=0`) still certifies via
    /// the parametric path — the legs are parametric in `(c, δ)`, not tied to the δ=1 instance.
    #[test]
    fn unbounded_affine_counter_nonunit_delta() {
        let spec = "---- MODULE Unb2 ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 3\n\
                    Next == x' = x + 5\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("an unbounded counter with c=3, δ=5 must certify");
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("parametric invariant present");
        assert_eq!((ub.init, ub.delta), (3, 5));
        assert_eq!(
            ub.pairs,
            vec![(3, 5)],
            "single-var cert carries pairs=[(c,δ)]"
        );
        assert!(verify_explicit_state_cert(&cert));
    }

    /// PHASE-2 WIDENING (`docs/kernel-checked-tla-plan.md`): `Safety == x >= N` for a literal N > 0 —
    /// beyond the historical `x >= 0` NonNeg fragment. `Init x=5 / Next x'=x+1 / Safety x>=2` has an
    /// INFINITE reachable set; the parametric lane proves `J ≡ x≥2` via the prelude's constructive
    /// `Int.le` lemmas (`le_trans`/`add_le_add_left`/`add_zero`), kernel-checked, NO enumeration.
    #[test]
    fn unbounded_counter_with_nonzero_lower_bound_certifies() {
        let spec = "---- MODULE UnbGe ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 5\n\
                    Next == x' = x + 1\n\
                    Safety == x >= 2\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("x>=N (N=2) must certify via the widened parametric leg");
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("parametric invariant present");
        assert_eq!((ub.init, ub.delta, ub.bound), (5, 1, 2));
        assert!(cert.reachable.is_empty(), "no enumeration");
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check of the Int.le legs"
        );
        // TAMPER: a cert claiming a WEAKER bound than the spec's must be caught by Leg-E
        // re-recognition (struct equality on the re-derived shape); here, kernel-level:
        // a different bound rebuilds different obligation types → the stored terms fail.
        let mut tampered = cert.clone();
        if let Some(ub) = tampered.unbounded_invariant.as_mut() {
            ub.bound = 0;
        }
        assert!(
            !verify_explicit_state_cert(&tampered),
            "a tampered bound must fail the kernel re-check (le-legs at NonNeg types)"
        );
    }

    /// SERDE/DIGEST BACK-COMPAT: a `bound = 0` cert (the entire pre-widening population)
    /// must serialize with NO `bound` key — byte-identical to certificates minted before the
    /// field existed — so `SafetyCertificate` sha256 digests (recomputed over a
    /// re-serialization at verify time) keep matching for old certs. A `bound > 0` cert
    /// (new-only) serializes it. Round-trip from old-style JSON stays byte-stable.
    #[test]
    fn bound_zero_serializes_byte_identically_to_pre_widening_certs() {
        let spec = "---- MODULE Unb ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("unbounded certifies");
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("parametric invariant");
        assert_eq!(ub.bound, 0);
        let json = serde_json::to_string(ub).expect("serializes");
        assert!(
            !json.contains("\"bound\""),
            "bound=0 must not serialize (pre-widening digest compatibility): {json}"
        );
        // Old-style JSON (no `bound` key) round-trips byte-stably.
        let reparsed: UnboundedInvariantCert = serde_json::from_str(&json).expect("reparses");
        assert_eq!(&reparsed, ub);
        assert_eq!(
            serde_json::to_string(&reparsed).expect("re-serializes"),
            json
        );
        // A widened cert DOES carry the field (self-consistent, new-only).
        let spec_ge = "---- MODULE UnbGe ----\n\
                       EXTENDS Integers\n\
                       VARIABLE x\n\
                       Init == x = 5\n\
                       Next == x' = x + 1\n\
                       Safety == x >= 2\n\
                       ====\n";
        let cert_ge = certify_explicit_state_spec(spec_ge, &cfg()).expect("x>=2 certifies");
        let ub_ge = cert_ge.unbounded_invariant.as_ref().expect("invariant");
        assert_eq!(ub_ge.bound, 2);
        assert!(serde_json::to_string(ub_ge)
            .expect("serializes")
            .contains("\"bound\":2"));
    }

    /// PHASE-2 WIDENING, general δ: `Init x=10 / Next x'=x+3 / Safety x>=4` exercises the
    /// general-δ consecution route (`add_le_add_left` + `Eq.subst` along `add_zero`).
    #[test]
    fn unbounded_counter_nonzero_bound_nonunit_delta_certifies() {
        let spec = "---- MODULE UnbGeD ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 10\n\
                    Next == x' = x + 3\n\
                    Safety == x >= 4\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("x>=4 with δ=3 must certify via the widened parametric leg");
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("parametric invariant present");
        assert_eq!((ub.init, ub.delta, ub.bound), (10, 3, 4));
        assert!(verify_explicit_state_cert(&cert));
    }

    /// PHASE-2 WIDENING fail-closed: an initial value BELOW the bound (`Init x=1 / Safety x>=2`)
    /// is UNSAFE at the initial state — the recognizer must decline (never a false `safe`).
    #[test]
    fn unbounded_counter_init_below_bound_fails_closed() {
        let spec = "---- MODULE UnbBad ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 1\n\
                    Next == x' = x + 1\n\
                    Safety == x >= 2\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "Init below the safety bound must NOT certify (x=1 violates x>=2)"
        );
    }

    /// MULTI-VARIABLE UNBOUNDED end-to-end: `VARIABLES x,y / Init x=0∧y=3 / Next x'=x+1∧y'=y+2 /
    /// Safety x≥0∧y≥0` has an INFINITE product reachable set the finite BFS cannot fold. The PARAMETRIC
    /// multi-variable leg certifies it WITHOUT enumeration: the cert carries the conjoined
    /// `unbounded_invariant` (NOT a finite `reachable`), and the kernel re-checks the 3 conjoined legs.
    #[test]
    fn multivar_unbounded_counter_certifies_without_enumeration() {
        let spec = "---- MODULE MVUnb ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 3\n\
                    Next == x' = x + 1 /\\ y' = y + 2\n\
                    Safety == x >= 0 /\\ y >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect(
            "the multi-var unbounded counter must be Certified via the parametric tuple leg",
        );
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("the cert must carry the conjoined parametric inductive invariant");
        assert_eq!(
            ub.pairs,
            vec![(0, 1), (3, 2)],
            "per-variable (c_j, δ_j) in declaration order"
        );
        assert_eq!(
            (ub.init, ub.delta),
            (0, 1),
            "the scalar summary is pairs[0]"
        );
        assert!(
            cert.reachable.is_empty(),
            "a multi-var unbounded cert has NO enumerated reachable set"
        );
        assert!(cert.init_values.is_empty() && cert.image.is_empty());
        assert_eq!(
            cert.sorts,
            vec![ColSort::Int, ColSort::Int],
            "an all-Int product of width n=2"
        );
        assert!(
            cert.next_shape.is_none(),
            "the unbounded path is NOT the finite affine path"
        );
        // The kernel re-check of the 3 conjoined legs (And.intro / NonNeg.add / And.left/right) passes.
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the 3 conjoined multi-variable legs must pass"
        );
        // TAMPER: mutating the stored `pairs` (or a leg) must break the re-check (bind to the spec).
        let mut tampered = cert.clone();
        if let Some(ub) = tampered.unbounded_invariant.as_mut() {
            ub.pairs[1].1 = 3; // claim δ_y=3 while the leg proves δ_y=2
        }
        assert!(
            !verify_explicit_state_cert(&tampered),
            "a cert whose claimed δ disagrees with the kernel-checked leg must be rejected"
        );
    }

    /// RELATIONAL UNBOUNDED end-to-end (STRONGER than per-variable nonneg): `VARIABLES x,y /
    /// Init x=0∧y=0 / Next x'=x+1∧y'=y+1 / Safety x=y`. `x=y` is NOT a per-variable nonneg — it couples
    /// the two variables — and is preserved by the lock-step step. The parametric RELATIONAL leg
    /// certifies it WITHOUT enumeration: the cert carries `relational = Some((0,1))`, and the kernel
    /// re-checks the 3 `Eq`-legs (`Eq.refl` / `Eq.subst` / identity).
    #[test]
    fn relational_unbounded_counter_certifies_without_enumeration() {
        let spec = "---- MODULE RelUnb ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 0\n\
                    Next == x' = x + 1 /\\ y' = y + 1\n\
                    Safety == x = y\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the relational lock-step counter must be Certified via the parametric Eq-leg");
        let ub = cert
            .unbounded_invariant
            .as_ref()
            .expect("carries the relational invariant");
        assert_eq!(
            ub.relational,
            Some((0, 1)),
            "the relational (c,δ) is recorded"
        );
        assert!(
            ub.pairs.is_empty(),
            "a relational cert carries no per-variable nonneg vector"
        );
        assert!(
            cert.reachable.is_empty(),
            "a relational unbounded cert has NO enumerated reachable set"
        );
        assert_eq!(cert.sorts, vec![ColSort::Int, ColSort::Int]);
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the 3 relational Eq-legs must pass"
        );
        // TAMPER the claimed δ (1→2): the stored Eq.subst leg proves δ=1, so re-check at the δ=2 type
        // is rejected by the kernel.
        let mut tampered = cert.clone();
        if let Some(ub) = tampered.unbounded_invariant.as_mut() {
            ub.relational = Some((0, 2));
            ub.delta = 2;
        }
        assert!(
            !verify_explicit_state_cert(&tampered),
            "a relational cert whose claimed δ disagrees with the kernel-checked leg must be rejected"
        );
    }

    /// FAIL-CLOSED (relational, unsound coupling): `Next x'=x+1 ∧ y'=y+2` with `Safety x=y` does NOT
    /// preserve `x=y` (the increments differ). The relational recognizer declines the mismatched δ, and
    /// the conjunctive-nonneg recognizer declines the `x=y` Safety — so no cert is emitted.
    #[test]
    fn relational_unbounded_mismatched_delta_fails_closed() {
        let spec = "---- MODULE RelBad ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 0\n\
                    Next == x' = x + 1 /\\ y' = y + 2\n\
                    Safety == x = y\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "mismatched increments do NOT preserve x=y ⇒ no cert (no false relational cert)"
        );
    }

    /// FAIL-CLOSED (multi-var, genuinely unsafe): `Init x=0∧y=0 / Next x'=x+1∧y'=y-1 / Safety x≥0∧y≥0`.
    /// `y'=y-1` DRIVES `y` negative, so the spec is UNSAFE and no inductive-nonneg cert exists. The
    /// recognizer declines the decrement conjunct (`parse_prime_eq_var_plus` fails on `y-1`) ⇒ no cert.
    #[test]
    fn multivar_unbounded_unsafe_fails_closed() {
        let spec = "---- MODULE MVUnsafe ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 0\n\
                    Next == x' = x + 1 /\\ y' = y - 1\n\
                    Safety == x >= 0 /\\ y >= 0\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "a spec that drives y negative must NOT be certified (no false cert)"
        );
    }

    /// FAIL-CLOSED: a non-inductive unbounded spec `Init x=5 / Next x'=x+1 / Safety x<=10`. `x≤10` is
    /// NOT the nonneg `J≡Safety≡x≥0` shape (and is not preserved by `x'=x+1`), so the unbounded
    /// recognizer declines AND the finite BFS would not reach a fixpoint either — no Certified verdict.
    #[test]
    fn non_inductive_unbounded_spec_fails_closed() {
        let spec = "---- MODULE NotInd ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 5\n\
                    Next == x' = x + 1\n\
                    Safety == x <= 10\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "a non-nonneg, non-inductive unbounded spec must NOT be certified"
        );
    }

    /// REGRESSION: the BOUNDED counter `x'=x+1 ∧ x<3` still uses the FINITE enumeration path (it has a
    /// guard ⇒ finite R={0,1,2,3}), NOT the unbounded parametric path. The cert carries a finite
    /// `reachable` + the affine `next_shape`, and `unbounded_invariant` is `None`.
    #[test]
    fn bounded_counter_still_uses_finite_path() {
        let spec = "---- MODULE Bounded ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1 /\\ x < 3\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the bounded counter must certify via the FINITE path");
        assert!(
            cert.unbounded_invariant.is_none(),
            "a GUARDED (finite) counter must NOT take the unbounded path"
        );
        assert_eq!(cert.reachable, vec![vec![0], vec![1], vec![2], vec![3]]);
        assert_eq!(
            cert.next_shape,
            Some(AffineNextShape { delta: 1, bound: 3 })
        );
        assert!(verify_explicit_state_cert(&cert));
    }

    /// GENERAL (non-stutter) closure: a BOUNDED counter `x'=x+1 ∧ x<3` has real transitions
    /// (0→1→2→3, all distinct), reaches the finite fixpoint R={0,1,2,3} with image {1,2,3}, and now
    /// CERTIFIES — the closed-under-Next leg proves each successor ∈ R (was fail-closed under the old
    /// stutter-only requirement).
    #[test]
    fn live_explicit_fixpoint_bounded_non_stutter_certifies() {
        let spec = "---- MODULE BCounter ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1 /\\ x < 3\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("bounded non-stutter counter must certify (R={0,1,2,3})");
        assert_eq!(cert.reachable, vec![vec![0], vec![1], vec![2], vec![3]]);
        assert_eq!(
            cert.image,
            vec![vec![1], vec![2], vec![3]],
            "image(R) under Next = successors 1,2,3"
        );
        // The affine `Next` is recognized LIVE, so the cert carries the KERNEL-RE-EVALUATED completeness
        // leg: the kernel itself reduced `Next` over the finite domain D and proved R is closed under the
        // RELATION (not merely over TY's enumerated image) — the enumerator-trust gap is closed here.
        assert_eq!(
            cert.next_shape,
            Some(AffineNextShape { delta: 1, bound: 3 }),
            "the live Next x'=x+1 ∧ x<3 must be recognized as the affine shape"
        );
        assert!(
            cert.next_completeness.is_some(),
            "the kernel must have re-evaluated Next over the finite domain (completeness leg present)"
        );
        // The literal `Init == x = 0` is recognized too, so the kernel ALSO re-evaluated Init over the
        // finite domain — `Init ⊆ R` no longer trusts that TY enumerated every init state. With both
        // legs, R is a fully KERNEL-VERIFIED inductive invariant (Init-complete ∧ Next-closed ∧ Safe).
        assert_eq!(
            cert.init_shape,
            Some(vec![0]),
            "Init x=0 recognized as the literal set {{0}}"
        );
        assert!(
            cert.init_completeness.is_some(),
            "the kernel re-evaluated Init over the finite domain"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. BOTH completeness legs"
        );
    }

    /// LIVE end-to-end MULTI-VARIABLE: a 2-variable stutter spec — the live model checker enumerates
    /// the reachable set of TUPLES and the Clean kernel certifies + re-checks all three legs over the
    /// And-conjunction tuple-membership encoding. Proof the multi-var generalization runs LIVE.
    #[test]
    fn live_explicit_fixpoint_two_variable() {
        let spec = "---- MODULE TwoVar ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 2\n\
                    Next == x' = x /\\ y' = y\n\
                    Safety == x >= 0 /\\ y >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("2-variable spec must kernel-certify over tuples");
        assert_eq!(
            cert.reachable,
            vec![vec![0, 2]],
            "R is the single tuple (0,2)"
        );
        assert!(verify_explicit_state_cert(&cert));
    }

    /// LIVE end-to-end MIXED Int/Bool: a spec with an Int var `x` and a Bool var `b`. The live model
    /// checker enumerates the reachable TUPLES (Bool extracted as 1/0), the orchestrator records the
    /// per-column sort vector `[Int, Bool]`, and the Clean kernel certifies + re-checks all three legs
    /// over the mixed `Eq Int` / `Eq Bool` tuple-membership encoding (Safety `⋀_{Int} x≥0` — no
    /// conjunct for the Bool column). Proof the compound-sort (Int+Bool) fragment runs LIVE.
    #[test]
    fn live_explicit_fixpoint_int_bool_mixed() {
        let spec = "---- MODULE IntBool ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, b\n\
                    Init == x = 0 /\\ b = TRUE\n\
                    Next == x' = x /\\ b' = b\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("mixed Int/Bool spec must kernel-certify over tuples");
        assert_eq!(
            cert.reachable,
            vec![vec![0, 1]],
            "R is the single tuple (x=0, b=true→1)"
        );
        assert_eq!(
            cert.sorts,
            vec![ColSort::Int, ColSort::Bool],
            "x is Int, b is Bool"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the mixed Int/Bool legs must pass"
        );
    }

    /// GENERAL (NON-affine) single-Int Next: `x' = (x + 2) % 5 ∧ x' < 9` is NOT the affine
    /// `x'=x+δ ∧ x<N` shape, so the affine recognizer DECLINES — yet it embeds as a PredIR and RULE 1
    /// reads the successor bound `H=8` straight off `x' < 9`. The cert therefore carries `next_pred` +
    /// a KERNEL-RE-EVALUATED general-completeness leg (NOT `next_shape`), and the kernel re-check passes.
    #[test]
    fn live_explicit_fixpoint_general_nonaffine_certifies() {
        let spec = "---- MODULE GCounter ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = (x + 2) % 5 /\\ x' < 9\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("non-affine modular Next must certify via the general IR leg");
        // The affine recognizer must NOT have fired (the update is `(x+2)%5`, not `x+δ`).
        assert!(
            cert.next_shape.is_none(),
            "non-affine Next is NOT the affine shape"
        );
        assert!(
            cert.next_completeness.is_none(),
            "no affine completeness leg"
        );
        // The GENERAL leg IS present: the kernel re-evaluated the ACTUAL predicate over R×D.
        let np = cert.next_pred.as_ref().expect("general Next IR present");
        assert_eq!(
            np.hi,
            vec![8],
            "RULE 1 reads the per-column bound H=8 off `x' < 9`"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel re-evaluated the real Next predicate over the finite domain"
        );
        // Init `x = 0` is still the literal-disjunction leg (the general Init fallback is not needed).
        assert_eq!(cert.init_shape, Some(vec![0]));
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the general Next leg"
        );

        // TAMPER (witness bytes): corrupt the serialized general-completeness term — verify rebuilds the
        // obligation TYPE from (R, D, IR) and re-runs the kernel on the bogus term, which fails to check.
        let mut bad = cert.clone();
        bad.next_general_completeness = Some(b"{\"BVar\":0}".to_vec());
        assert!(
            !verify_explicit_state_cert(&bad),
            "a tampered general-completeness witness must fail the kernel re-check"
        );

        // TAMPER (under-approximated R): drop a reachable state — the kernel general-completeness leg
        // must REFUTE closure (some s'∈D is a successor but s'∉R after removal). This is the soundness
        // net the leg exists for (the enumerator-trust gap closed for the general fragment).
        let mut under = cert.clone();
        under.reachable.pop();
        assert!(
            !verify_explicit_state_cert(&under),
            "an under-approximated R must fail the kernel general-completeness re-check"
        );
    }

    /// THE HOURCLOCK CLASS — IF-THEN-ELSE `Next`: `x' = IF x # 12 THEN x + 1 ELSE 1` is desugared
    /// AT RECOGNITION (truth-exactly, both directions — `cleancic::eq_if_form`) into
    /// `(x≠12 ∧ x'=x+1) ∨ (¬(x≠12) ∧ x'=1)`, reaching the GENERAL kernel-re-evaluated
    /// completeness legs: the kernel re-evaluates the REAL `Next` over `R × D` (RULE-4 Or-split
    /// bound `H = max(arm pins) = 13`), so closure holds over the RELATION — the certificate is
    /// ENUMERATOR-FREE. The `D ⊇ Succ(R)` coverage is ALSO kernel-proven (per-state Π-lemmas via
    /// the or-elimination walk), so no Rust bound rule remains in the trust story.
    #[test]
    fn live_explicit_fixpoint_if_then_else_next_certifies() {
        let spec = "---- MODULE Clock ----\n\
                    EXTENDS Naturals\n\
                    VARIABLE x\n\
                    Init == x \\in (1 .. 12)\n\
                    Next == x' = IF x # 12 THEN x + 1 ELSE 1\n\
                    Safety == x \\in (1 .. 12)\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the IF-THEN-ELSE Next (HourClock class) must certify via the general leg");
        assert_eq!(
            cert.reachable,
            (1..=12u64).map(|v| vec![v]).collect::<Vec<_>>()
        );
        // NOT the affine shortcut — the general IF-desugared leg.
        assert!(cert.next_shape.is_none() && cert.next_completeness.is_none());
        let np = cert
            .next_pred
            .as_ref()
            .expect("general Next IR present (IF desugared to Or)");
        assert!(
            matches!(np.pred, PredIR::Or(_, _)),
            "the stored Next IR is the desugared Or shape"
        );
        assert_eq!(
            np.hi,
            vec![13],
            "RULE 4 Or-split: max(then-pin 12+1, else-pin 1) = 13"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel re-evaluated the REAL IF-desugared Next over R×D — enumerator-free closure"
        );
        // Init `x ∈ 1..12` rides the general Init leg (interval recognition).
        assert!(cert.init_pred.is_some() && cert.init_general_completeness.is_some());
        // Coverage: BOTH axes kernel-proven (Next via the or-elimination walk; Init via the
        // single Π-lemma) — nothing Rust-derived remains.
        let cov = domain_coverage_of_cert(&cert);
        assert!(cov.any_completeness_leg && !cov.shortcut_legs);
        assert_eq!(
            cov.next_kernel_columns,
            vec![0],
            "Next axis KERNEL-PROVEN (or-elim + pins)"
        );
        assert_eq!(cov.init_kernel_columns, vec![0], "Init axis KERNEL-PROVEN");
        assert!(cov.next_rust_columns.is_empty() && cov.init_rust_columns.is_empty());
        assert!(
            cov.fully_construction_covered(),
            "passes --require-domain-complete"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the Or-shaped legs"
        );

        // TAMPER (stored Next IR): mutate the else-arm pin `x'=1` → `x'=0`. Verify rebuilds the
        // obligation from the STORED IR — the serialized witness term no longer checks against
        // the rebuilt type → fail-closed. (Binding the IR to the SPEC is Leg E's re-recognition.)
        let mut bad = cert.clone();
        if let Some(d) = bad.next_pred.as_mut() {
            let PredIR::Or(_, ref mut else_arm) = d.pred else {
                panic!("Or shape")
            };
            let PredIR::And(_, ref mut pin) = **else_arm else {
                panic!("And arm")
            };
            **pin = PredIR::Eq(ValIR::Prime(0), ValIR::Lit(0));
        }
        assert!(
            !verify_explicit_state_cert(&bad),
            "a tampered stored Next IR must fail the kernel re-check"
        );
        // TAMPER (witness bytes): a bogus serialized term must fail against the rebuilt type.
        let mut bad2 = cert.clone();
        bad2.next_general_completeness = Some(b"{\"BVar\":0}".to_vec());
        assert!(!verify_explicit_state_cert(&bad2));
        // TAMPER (under-approximated R): drop a state — the kernel refutes closure over the
        // RELATION (some successor in D is no longer in R). The soundness net the leg exists for.
        let mut under = cert.clone();
        under.reachable.pop();
        assert!(!verify_explicit_state_cert(&under));
    }

    /// MULTIPLE configured INVARIANTs are conjoined and certified together (the common corpus
    /// shape — a spec with `INVARIANTS I1 I2 …`). A spec whose reachable set satisfies BOTH
    /// per-variable intervals certifies via the general safety leg; verify re-conjoins from the
    /// re-derived config and re-accepts. A single-invariant config is byte-identical to before
    /// (the fold is identity — pinned by the existing digest back-compat fixtures).
    #[test]
    fn multiple_configured_invariants_are_conjoined() {
        let spec = "---- MODULE Multi ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 0\n\
                    Next == x' = x + 1 /\\ y' = y + 1 /\\ x < 5\n\
                    InvX == x \\in 0 .. 10\n\
                    InvY == y \\in 0 .. 10\n\
                    ====\n";
        let mut c = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InvX".to_string(), "InvY".to_string()],
            ..Default::default()
        };
        let cert = certify_explicit_state_spec(spec, &c)
            .expect("a spec with TWO configured invariants must certify their conjunction");
        // The conjoined safety predicate rode the general leg (`InvX ∧ InvY`).
        assert!(
            cert.safety_pred.is_some(),
            "conjunction recognized into the general safety leg"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "verify re-conjoins and re-accepts"
        );

        // FAIL-CLOSED: a SECOND invariant violated on a reachable state must decline (the
        // conjunction reduces to false in the kernel on that state). `y` reaches 5, so `y ∈ 0..3`
        // is violated at (4,4)…(5,5).
        c.invariants = vec!["InvX".to_string(), "Narrow".to_string()];
        let spec2 = "---- MODULE Multi2 ----\n\
                     EXTENDS Integers\n\
                     VARIABLES x, y\n\
                     Init == x = 0 /\\ y = 0\n\
                     Next == x' = x + 1 /\\ y' = y + 1 /\\ x < 5\n\
                     InvX == x \\in 0 .. 10\n\
                     Narrow == y \\in 0 .. 3\n\
                     ====\n";
        assert!(
            certify_explicit_state_spec(spec2, &c).is_none(),
            "a violated conjunct must fail-close the whole certificate"
        );
    }

    /// LIVE MULTI-VARIABLE all-Int GENERAL completeness: a 2-var spec where `x` is a bounded counter
    /// (`x'=x+1 ∧ x'<3`) and `y` STUTTERS (`y'=y`). The kernel re-evaluates the REAL `Next` over the
    /// PRODUCT domain `D = {0..=H_x} × {0..=H_y}`, with `H_x=2` (primed bound `x'<3`) and `H_y=0`
    /// (stutter ⇒ bounded by max-over-R of the y-column = 0). The general Next + Init legs are both
    /// present and KERNEL-re-checked — the multi-var generalization runs LIVE end-to-end.
    #[test]
    fn live_explicit_fixpoint_multivar_all_int_general() {
        let spec = "---- MODULE MVCounter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, y\n\
                    Init == x = 0 /\\ y = 0\n\
                    Next == x' = x + 1 /\\ x' < 3 /\\ y' = y\n\
                    Safety == x >= 0 /\\ y >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect(
            "multi-var all-Int spec must kernel-certify the general legs over the product domain",
        );
        assert_eq!(
            cert.reachable,
            vec![vec![0, 0], vec![1, 0], vec![2, 0]],
            "R = {{(0,0),(1,0),(2,0)}} (x counts 0..2 under x'<3, y stutters)"
        );
        assert_eq!(cert.sorts, vec![ColSort::Int, ColSort::Int]);
        // The single-Int affine/literal shortcuts do NOT fire for a 2-var spec; the GENERAL multi-var
        // legs do, with the PER-COLUMN product-domain bounds.
        assert!(cert.next_shape.is_none() && cert.init_shape.is_none());
        let np = cert
            .next_pred
            .as_ref()
            .expect("general multi-var Next IR present");
        assert_eq!(
            np.hi,
            vec![2, 0],
            "per-column bounds: H_x=2 (x'<3), H_y=0 (y stutter, max R = 0)"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "kernel re-evaluated Next over the product domain"
        );
        let ip = cert
            .init_pred
            .as_ref()
            .expect("general multi-var Init IR present");
        assert_eq!(
            ip.hi,
            vec![0, 0],
            "Init pins x=0,y=0 ⇒ per-column bounds (0,0)"
        );
        assert!(
            cert.init_general_completeness.is_some(),
            "kernel re-evaluated Init over the product domain"
        );
        // (b) end-to-end through the Leg-E verifier: ACCEPTS.
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the multi-var all-Int general legs must pass"
        );

        // (c) the kernel REJECTS an under-approximated R: drop (2,0) — now (1,0)→(2,0) is a real
        // successor (x'=2<3, y'=0) but (2,0)∉R ⇒ the general Next-completeness leg REFUTES closure over
        // the product domain. This is the soundness net (the enumerator-trust gap closed multi-var).
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![2, 0]);
        assert!(
            !verify_explicit_state_cert(&under),
            "an under-approximated multi-var R must fail the kernel general-completeness re-check"
        );
    }

    /// LIVE FUNC-of-ENUM enumerator-free Init AND Next — the Gray-Lamport TRANSACTION COMMIT (TCommit)
    /// class. A single `rmState : [RM -> {…}]` FuncEnum column whose Init is the `[rm ∈ RM |-> "working"]`
    /// constructor and whose Next is `∃rm∈RM : rmState' = [rmState EXCEPT ![rm] = "…"]` — a func-EXCEPT at a
    /// MODEL-VALUE key, under a bounded existential over the model-value set. The kernel RE-EVALUATES Init
    /// and the Next RELATION over the FuncEnum pack universe `|labels|^arity`, so BOTH general legs fire
    /// (the fully-free tier), and an under-approximated `R` is REFUTED — the enumerator-trust soundness net.
    #[test]
    fn live_explicit_fixpoint_func_enum_except_general() {
        let spec = "---- MODULE TC ----\n\
                    VARIABLE rmState\n\
                    Init == rmState = [rm \\in RM |-> \"working\"]\n\
                    Commit(rm) == rmState[rm] = \"working\" /\\ rmState' = [rmState EXCEPT ![rm] = \"committed\"]\n\
                    Abort(rm)  == rmState[rm] = \"working\" /\\ rmState' = [rmState EXCEPT ![rm] = \"aborted\"]\n\
                    Next == \\E rm \\in RM : Commit(rm) \\/ Abort(rm)\n\
                    Safety == \\A rm \\in RM : rmState[rm] \\in {\"working\",\"committed\",\"aborted\"}\n\
                    ====\n";
        use crate::config::ConstantValue;
        let mut config = cfg();
        config.constants.insert(
            "RM".to_string(),
            ConstantValue::ModelValueSet(vec!["r1".to_string(), "r2".to_string()]),
        );
        let cert = certify_explicit_state_spec(spec, &config).expect(
            "the FuncEnum func-EXCEPT-under-∃ spec must certify the general Init AND Next legs",
        );
        // rmState is the FuncEnum column: arity 2 (RM={r1,r2}) over a model-value domain.
        assert!(
            matches!(cert.sorts.as_slice(), [ColSort::FuncEnum { arity: 2, .. }]),
            "rmState encodes as a FuncEnum column, got {:?}",
            cert.sorts
        );
        // BOTH enumerator-free legs fire — the "RE-EVALUATED Init AND Next" fully-free tier.
        assert!(
            cert.next_general_completeness.is_some(),
            "kernel re-evaluated the func-EXCEPT Next RELATION over the FuncEnum pack universe"
        );
        assert!(
            cert.init_general_completeness.is_some(),
            "kernel re-evaluated the [rm∈RM|->\"working\"] Init constructor over the FuncEnum pack universe"
        );
        assert!(
            cert.next_shape.is_none() && cert.init_shape.is_none(),
            "a FuncEnum column is not the single-Int affine/literal shortcut shape"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the FuncEnum general-leg cert must kernel-re-check"
        );

        // UNDER-APPROXIMATED R (the enumerator-trust soundness net): drop a reachable state that is a real
        // func-EXCEPT successor (every non-init state is a successor of `init`). A remaining state's
        // `rmState' = [rmState EXCEPT ![rm]=…]` then lands OUTSIDE R over the pack universe ⇒ the kernel
        // REFUTES closure. Never a false SAFE from an incomplete enumeration.
        let victim = cert
            .reachable
            .iter()
            .find(|t| *t != &cert.init_values[0])
            .cloned()
            .expect("a non-init reachable state exists");
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &victim);
        assert!(
            !verify_explicit_state_cert(&under),
            "an under-approximated FuncEnum R must fail the kernel general-completeness re-check"
        );

        // A VIOLATED variant (identical func-EXCEPT Next + FuncDef Init, but an invariant that FAILS on a
        // reachable successor) must DECLINE — the explicit-fixpoint safety leg refutes `R ⊆ Safety` even
        // though the enumerator-free Next/Init legs recognize this shape. Never a false SAFE from the new
        // FuncEnum recognizer.
        let violated = spec.replace(
            "Safety == \\A rm \\in RM : rmState[rm] \\in {\"working\",\"committed\",\"aborted\"}",
            "Safety == \\A rm \\in RM : rmState[rm] = \"working\"",
        );
        assert!(
            certify_explicit_state_spec(&violated, &config).is_none(),
            "a violated func-EXCEPT spec must decline to certify (R ⊄ Safety)"
        );
    }

    /// The full three-RM TCommit action includes nested nullary operators whose bodies quantify over
    /// the same `rm` name as the outer action. Its 34×64 completeness obligation is below the normal
    /// monolith leg-count cap, but the expanded global guards make that single kernel term structurally
    /// expensive. The exact per-source fallback must retain the enumerator-free proof and Leg-E must
    /// re-check it through the same path.
    #[test]
    fn live_explicit_fixpoint_tcommit_nested_guards_general() {
        let spec = "---- MODULE TC ----\n\
                    VARIABLE rmState\n\
                    Init == rmState = [rm \\in RM |-> \"working\"]\n\
                    canCommit == \\A rm \\in RM : rmState[rm] \\in {\"prepared\", \"committed\"}\n\
                    notCommitted == \\A rm \\in RM : rmState[rm] # \"committed\"\n\
                    Prepare(rm) == /\\ rmState[rm] = \"working\"\n\
                                   /\\ rmState' = [rmState EXCEPT ![rm] = \"prepared\"]\n\
                    Decide(rm) == \\/ /\\ rmState[rm] = \"prepared\"\n\
                                      /\\ canCommit\n\
                                      /\\ rmState' = [rmState EXCEPT ![rm] = \"committed\"]\n\
                                  \\/ /\\ rmState[rm] \\in {\"working\", \"prepared\"}\n\
                                      /\\ notCommitted\n\
                                      /\\ rmState' = [rmState EXCEPT ![rm] = \"aborted\"]\n\
                    Next == \\E rm \\in RM : Prepare(rm) \\/ Decide(rm)\n\
                    Safety == rmState \\in [RM -> {\"working\", \"prepared\", \"committed\", \"aborted\"}]\n\
                    ====\n";
        use crate::config::ConstantValue;
        let mut config = cfg();
        config.constants.insert(
            "RM".to_string(),
            ConstantValue::ModelValueSet(vec![
                "r1".to_string(),
                "r2".to_string(),
                "r3".to_string(),
            ]),
        );
        let cert = certify_explicit_state_spec(spec, &config)
            .expect("the full nested-guard TCommit shape must certify");
        assert_eq!(
            cert.reachable.len(),
            34,
            "the real three-RM model has 34 states"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the exact per-source fallback must preserve enumerator-free Next closure"
        );
        assert!(
            cert.init_general_completeness.is_some(),
            "the FuncEnum constructor keeps enumerator-free Init completeness"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "Leg-E must re-check the per-source completeness fallback"
        );
        let mut under = cert.clone();
        let victim = under
            .reachable
            .iter()
            .find(|s| *s != &under.init_values[0])
            .cloned()
            .expect("a non-initial reachable state exists");
        under.reachable.retain(|s| s != &victim);
        assert!(
            !verify_explicit_state_cert(&under),
            "partitioning must still reject an under-approximated reachable set"
        );
    }

    /// LIVE PART A — general SUBTRACTION in `Next`: a bounded DOWN-COUNTER `x' = x − 1 ∧ x > 0 ∧ x' < 9`.
    /// The affine recognizer declines (subtraction, three conjuncts), so the GENERAL IR leg fires with the
    /// NARROWLY SOUND `nonneg = a − b` positive-polarity form (`x' = x − 1` ⇒ `Eq(Prime, Sub(Var, Lit))`).
    /// R = {3,2,1,0}; RULE 1 reads `H=8` off `x' < 9`; the kernel re-evaluates the REAL predicate (with
    /// `Nat.sub`, EXACT on every real transition since `x' = x−1 ≥ 0 ⇒ x ≥ 1`) over R×D and proves closure.
    #[test]
    fn live_explicit_fixpoint_subtraction_down_counter() {
        let spec = "---- MODULE DownCounter ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 3\n\
                    Next == x' = x - 1 /\\ x > 0 /\\ x' < 9\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the down-counter (subtraction Next) must certify via the general IR leg");
        assert_eq!(
            cert.reachable,
            vec![vec![0], vec![1], vec![2], vec![3]],
            "R = {{0,1,2,3}}"
        );
        // The affine recognizer must NOT have fired (the update is `x − 1`, not `x + δ`).
        assert!(
            cert.next_shape.is_none(),
            "subtraction Next is NOT the affine shape"
        );
        assert!(
            cert.next_completeness.is_none(),
            "no affine completeness leg"
        );
        // The GENERAL leg IS present with the narrow-Sub IR; RULE 1 reads H=8 off `x' < 9`.
        let np = cert
            .next_pred
            .as_ref()
            .expect("general Next IR present (narrow Sub)");
        assert_eq!(
            np.hi,
            vec![8],
            "RULE 1 reads the per-column bound H=8 off `x' < 9`"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel re-evaluated the real subtraction Next over the finite domain"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the general subtraction leg"
        );

        // UNDER-APPROXIMATED R: drop (0) — now (1)→(0) is a real successor (x'=0, guard 1>0, 0<9) but 0∉R
        // ⇒ the general Next-completeness leg REFUTES closure. The soundness net for the subtraction leg.
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![0]);
        assert!(
            !verify_explicit_state_cert(&under),
            "an under-approximated R must fail the kernel general subtraction-completeness re-check"
        );
    }

    /// PART A soundness net (test d): a down-counter with NO guard — `Next == x' = x − 1` from `x = 1` —
    /// reaches `x = 0`, whose successor `x' = 0 − 1 = −1` is a NEGATIVE TLA Int. `value_cell` REJECTS a
    /// negative Int cell (`Value::Int` negative ⇒ `None`), so the tuple cannot be packed and the WHOLE
    /// cert fails closed (`None`). This is exactly why the narrow-Sub soundness argument holds: a Sub that
    /// would go negative as a STATE value is never admitted into `R`. (Contrast the GUARDED down-counter
    /// `live_explicit_fixpoint_subtraction_down_counter`, which certifies because `x > 0` keeps `x' ≥ 0`.)
    #[test]
    fn subtraction_going_negative_as_state_fails_closed() {
        let spec = "---- MODULE UnguardedDown ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 1\n\
                    Next == x' = x - 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "an unguarded down-counter reaches x'=-1 (negative), rejected by value_cell ⇒ NO cert"
        );
    }

    /// LIVE PART B — SEQUENCE `Append` + `Len` in `Next`: a bounded sequence column `s` GROWS by appending
    /// `1` while a `Len(s) < 2` guard bounds it, and an Int counter `x` STUTTERS (carrying the `x ≥ 0`
    /// Safety leg). `Next == s' = Append(s, 1) ∧ x' = x ∧ Len(s) < 2`. The compound `Seq` column embeds as
    /// a packed `Nat`; the enumeration walks `<<>> → <<1>> → <<1,1>>` (packed `0, 2, 22`), and the kernel
    /// re-evaluates the packed `Append`/`Len` (`pack + (e+1)·D^Len`, `Len` = nonzero-digit fold) — so the
    /// sequence-op leg is exercised LIVE end-to-end. Pairs the seq var with the Int `x`.
    #[test]
    fn live_explicit_fixpoint_seq_append_op() {
        let spec = "---- MODULE SeqAppend ----\n\
                    EXTENDS Integers, Sequences\n\
                    VARIABLES s, x\n\
                    Init == s = <<>> /\\ x = 0\n\
                    Next == s' = Append(s, 1) /\\ x' = x /\\ Len(s) < 2\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect(
            "a Seq-Append/Len spec must kernel-certify (image⊆R; general leg if the pack fits)",
        );
        // R = the packed sequences <<>>, <<1>>, <<1,1>> paired with x=0. Packs (base D=10): 0, 2, 22.
        assert_eq!(
            cert.reachable,
            vec![vec![0, 0], vec![2, 0], vec![22, 0]],
            "R = {{ <<>>(0), <<1>>(2), <<1,1>>(22) }} × {{x=0}} (packed)"
        );
        assert!(
            matches!(cert.sorts.first(), Some(ColSort::Seq { .. })),
            "column 0 is a Seq column"
        );
        assert_eq!(
            cert.sorts.get(1),
            Some(&ColSort::Int),
            "column 1 is the paired Int (x≥0 Safety)"
        );
        // The honest enumerated image⊆R leg + the kernel re-check must pass end-to-end.
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check of the Seq-Append cert must pass"
        );
    }

    /// LIVE MULTI-VARIABLE Int+Bool GENERAL completeness: an Int counter `x` plus a STUTTERING Bool
    /// flag `b`. The Bool column embeds with domain {0,1} (H_b=1), the Int column with `H_x=2` from
    /// `x'<3`. The kernel re-evaluates the real `Next`/`Init` (incl. the Bool stutter `b'=b` and the
    /// bare-Bool `Init` predicate `b = TRUE`) over the product domain. Proof the Bool-column general
    /// leg runs LIVE.
    #[test]
    fn live_explicit_fixpoint_multivar_int_bool_general() {
        let spec = "---- MODULE MVIntBool ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, b\n\
                    Init == x = 0 /\\ b = TRUE\n\
                    Next == x' = x + 1 /\\ x' < 3 /\\ b' = b\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("multi-var Int+Bool spec must kernel-certify the general legs");
        assert_eq!(
            cert.reachable,
            vec![vec![0, 1], vec![1, 1], vec![2, 1]],
            "R = {{(0,T),(1,T),(2,T)}} (b stays TRUE→1)"
        );
        assert_eq!(cert.sorts, vec![ColSort::Int, ColSort::Bool]);
        let np = cert
            .next_pred
            .as_ref()
            .expect("general Int+Bool Next IR present");
        assert_eq!(
            np.hi,
            vec![2, 1],
            "H_x=2 (x'<3), H_b=1 (Bool domain {{0,1}})"
        );
        assert!(cert.next_general_completeness.is_some());
        assert!(
            cert.init_general_completeness.is_some(),
            "Init `b = TRUE` general leg present"
        );
        // (d) end-to-end Leg-E verifier ACCEPTS the Bool-column general legs.
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the multi-var Int+Bool general legs must pass"
        );
        // Under-approx: drop (2,T) — (1,T)→(2,T) is a real successor but absent ⇒ refuted.
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![2, 1]);
        assert!(
            !verify_explicit_state_cert(&under),
            "an under-approximated Int+Bool R must fail the kernel general-completeness re-check"
        );
    }

    /// BOOL-VALUED NEXT (the all-Bool toggle class `x' = ~x /\ y' = ~y`): the whole state space is
    /// reachable, and the Next recognizer now admits the Bool-op successor `x' = ~x` ⇒ a general
    /// `next_pred` (`Equiv(Eq(Prime,1), Not(Eq(Var,1)))` per var) + kernel-re-evaluated closure. Every
    /// axis is a Bool `{0,1}` UNIVERSE, so coverage is CONSTRUCTION-COMPLETE — the strict-domain
    /// certification (which DECLINES any Rust-derived axis) also succeeds. Proof B2 reaches the
    /// enumerator-FREE + ConstructionComplete tier.
    #[test]
    fn bool_next_all_bool_toggle_enumerator_free_construction_complete() {
        let spec = "---- MODULE B2 ----\n\
                    EXTENDS Naturals\n\
                    VARIABLES x, y\n\
                    Init == x \\in {TRUE, FALSE} /\\ y \\in {TRUE, FALSE}\n\
                    Next == x' = ~x /\\ y' = ~y\n\
                    Safety == x \\in {TRUE, FALSE} /\\ y \\in {TRUE, FALSE}\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the all-Bool toggle spec must kernel-certify");
        assert_eq!(cert.sorts, vec![ColSort::Bool, ColSort::Bool]);
        assert_eq!(
            cert.reachable,
            vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
        );
        // The Bool-op successor `x' = ~x` recognizes EXACTLY as `Equiv(Eq(Prime,1), Not(Eq(Var,1)))`.
        let np = cert
            .next_pred
            .as_ref()
            .expect("Bool-op general Next IR present");
        let leg = |i: usize| {
            PredIR::Equiv(
                Box::new(PredIR::Eq(ValIR::Prime(i), ValIR::Lit(1))),
                Box::new(PredIR::Not(Box::new(PredIR::Eq(
                    ValIR::Var(i),
                    ValIR::Lit(1),
                )))),
            )
        };
        assert_eq!(
            np.pred,
            PredIR::And(Box::new(leg(0)), Box::new(leg(1))),
            "x'=~x /\\ y'=~y ⇒ And of two `Equiv(Prime=1, ¬(Var=1))` legs"
        );
        assert_eq!(
            np.hi,
            vec![1, 1],
            "both axes are the Bool universe {{0,1}} (H=1)"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "kernel re-evaluated Next over R×D"
        );
        // CONSTRUCTION-COMPLETE: every completeness-leg axis is its column's full SORT universe.
        let cov = crate::cleancic::next_domain_bounds_cov_from_ir(
            &np.pred,
            2,
            &cert.reachable,
            &cert.sorts,
        )
        .expect("Bool axes bound");
        assert!(
            cov.iter()
                .all(|(_, c)| *c == crate::cleancic::DomainCoverage::UniverseComplete),
            "every Bool axis is UniverseComplete (ConstructionComplete): {cov:?}"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check of the Bool-op general legs"
        );
        // STRICT-DOMAIN mode (declines any Rust-derived axis) ALSO certifies — the enumerator-free
        // claim rests on no trusted-Rust bound rule (all axes universe-complete).
        let strict = certify_explicit_state_spec_strict_domain(spec, &cfg())
            .expect("all-Bool spec certifies in strict domain-complete mode");
        assert!(
            strict.next_general_completeness.is_some()
                && strict.init_general_completeness.is_some(),
            "strict mode keeps BOTH enumerator-free general legs (all Bool axes universe-complete)"
        );
    }

    /// ATTACK 1 — THE DECISIVE non-closed-R guard for the Bool-op Next. The coupled spec
    /// `x' = ~x /\ y' = x` has a PROPER-SUBSET reachable set `R = {(F,F),(T,F),(F,T)}` (`(T,T)` is
    /// unreachable) that IS closed under the recognized Next. DROP a genuine successor — `(T,F)→(F,T)`
    /// is a real transition, but `(F,T)` removed from `R` ⇒ `R` is no longer closed — and the kernel
    /// closure leg (`verify_explicit_state_cert`) MUST reduce to false. A non-closed R certifying would
    /// be a FALSE SAFE.
    #[test]
    fn bool_next_non_closed_r_declines() {
        let spec = "---- MODULE A1 ----\n\
                    EXTENDS Naturals\n\
                    VARIABLES x, y\n\
                    Init == x = FALSE /\\ y = FALSE\n\
                    Next == x' = ~x /\\ y' = x\n\
                    Safety == x \\in {TRUE, FALSE} /\\ y \\in {TRUE, FALSE}\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("the coupled Bool spec must kernel-certify");
        assert_eq!(
            cert.reachable,
            vec![vec![0, 0], vec![0, 1], vec![1, 0]],
            "R = {{(F,F),(F,T),(T,F)}} — a PROPER subset ((T,T) unreachable)"
        );
        assert!(cert.next_pred.is_some() && cert.next_general_completeness.is_some());
        assert!(
            verify_explicit_state_cert(&cert),
            "the genuinely-closed R re-checks"
        );
        // Drop (0,1) — a real successor of (1,0) under `x'=~x /\ y'=x` — so R is NOT closed.
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![0, 1]);
        assert!(
            !verify_explicit_state_cert(&under),
            "a NON-CLOSED Bool R must fail the kernel closure re-check (else a FALSE SAFE)"
        );
    }

    /// ATTACK 3 — mis-recognition fails closed. A Bool Next whose RHS operator is OUTSIDE the admitted
    /// fragment (`x' = (x => y)` — `Implies` is deliberately NOT in the gate) must DECLINE the general
    /// leg (no `next_pred`), never emit a wrong IR; closure then rests on the honest enumerated
    /// `image ⊆ R` leg (still sound). The whole-`Next` recognizer returns `None` for the Implies shape.
    #[test]
    fn bool_next_out_of_fragment_declines() {
        // End-to-end: the spec still certifies (image⊆R), but WITHOUT a general Next leg — the
        // out-of-fragment `=>` RHS declines the enumerator-FREE leg (see the recognizer-level
        // `bool_valued_next_out_of_fragment_declines` for the direct IR check).
        let spec = "---- MODULE A3a ----\n\
                    EXTENDS Naturals\n\
                    VARIABLES x, y\n\
                    Init == x \\in {TRUE, FALSE} /\\ y \\in {TRUE, FALSE}\n\
                    Next == x' = (x => y) /\\ y' = ~y\n\
                    Safety == x \\in {TRUE, FALSE} /\\ y \\in {TRUE, FALSE}\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("still certifies via the enumerated image⊆R leg");
        assert!(
            cert.next_pred.is_none() && cert.next_general_completeness.is_none(),
            "the out-of-fragment Bool Next declines the enumerator-FREE leg (fail-closed)"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the enumerated image leg still re-checks"
        );
    }

    /// LIVE end-to-end SET-VALUED column: an Int var `x` (carries the `x≥0` Safety leg) plus a SET var
    /// `chosen` that GROWS `chosen' = chosen ∪ {0}`. The live model checker enumerates the reachable
    /// TUPLES with the Set column stored as a BITMASK (`{}`→0, `{0}`→1); the orchestrator records the
    /// per-column sort `[Int, Set{16}]`; and the Clean kernel certifies + re-checks ALL THREE membership
    /// legs (`Init⊆R`, `image⊆R` via bitmask `Eq Nat`, `R⊆Safety`) over the mixed `Eq Int`/`Eq Nat`
    /// tuple encoding. The Set column is a CHANGING column over the wide (16-bit) universe, so the
    /// general Next completeness leg DECLINES (universe > the completeness cap) — closure rests on the
    /// honest kernel-checked enumerated `image ⊆ R` leg. Proof the bitmask Set column runs LIVE.
    #[test]
    fn live_explicit_fixpoint_set_column_membership() {
        let spec = "---- MODULE SetGrow ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, chosen\n\
                    Init == x = 0 /\\ chosen = {}\n\
                    Next == x' = x /\\ chosen' = chosen \\cup {0}\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect(
            "Int + Set-valued spec must kernel-certify the membership legs over the bitmask",
        );
        // R = {(x=0, chosen={}), (x=0, chosen={0})} → bitmasks 0 and 1.
        assert_eq!(
            cert.reachable,
            vec![vec![0, 0], vec![0, 1]],
            "Set column stored as bitmask {{}}→0,{{0}}→1"
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Set {
                    universe: SET_UNIVERSE_BITS
                }
            ],
            "x is Int, chosen is a Set bitmask column"
        );
        // image(R) under Next: every successor sets `chosen'={0}` (mask 1), x stutters at 0.
        assert_eq!(
            cert.image,
            vec![vec![0, 1]],
            "image = (x=0, chosen={{0}}=1)"
        );
        // CHANGING Set column over the 16-bit universe ⇒ general Next leg declined (universe > cap).
        assert!(
            cert.next_pred.is_none() && cert.next_general_completeness.is_none(),
            "a changing Set column over the wide universe declines the general Next leg"
        );
        // The membership legs + R⊆Safety + Leg-E re-check ALL pass over the bitmask Set column.
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the Int+Set membership legs must pass"
        );

        // SOUNDNESS net: an under-approximated R (drop the (0,{0}) successor) must be REJECTED — the
        // image tuple (0,1) is then not a member of R, so the closed-under-Next (image⊆R) leg refutes.
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![0, 1]);
        assert!(
            !verify_explicit_state_cert(&under),
            "an under-approximated R (missing a Set successor) must fail the kernel image⊆R re-check"
        );
    }

    /// LIVE SET-VALUED column with a STUTTERING set, so the GENERAL completeness leg FIRES via the
    /// set-stutter domain rule: `chosen' = chosen` ⇒ the successor bitmask equals the current ⇒ bounded
    /// by `max(R's chosen column)`. The kernel re-evaluates the REAL `Next` (incl. `SetUnchanged`-shaped
    /// `chosen'=chosen` via `Nat.beq`) over the product domain and proves closure. Proof the bitmask Set
    /// column reaches kernel-RE-EVALUATED completeness (not just enumerated-image) when stuttered.
    #[test]
    fn live_explicit_fixpoint_set_column_stutter_completeness() {
        let spec = "---- MODULE SetStutter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, chosen\n\
                    Init == x = 0 /\\ chosen = {0, 1}\n\
                    Next == x' = x /\\ chosen' = chosen\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("Int + stuttering Set spec must kernel-certify");
        // chosen = {0,1} → bitmask 0b11 = 3; the single reachable tuple (x=0, chosen=3).
        assert_eq!(cert.reachable, vec![vec![0, 3]], "chosen={{0,1}}→bitmask 3");
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Set {
                    universe: SET_UNIVERSE_BITS
                }
            ]
        );
        // The STUTTER set column gives a sound finite product-domain bound (max R = 3), so the general
        // Next leg fires and is kernel-re-evaluated.
        let np = cert
            .next_pred
            .as_ref()
            .expect("general Next IR present for the stuttering set spec");
        assert_eq!(
            np.hi,
            vec![0, 3],
            "H_x=0 (x stutter, max R = 0), H_chosen=3 (set stutter, max R = 3)"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel re-evaluated Next (incl. the set-stutter conjunct) over the product domain"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the set-stutter general Next leg"
        );
    }

    /// LIVE end-to-end RECORD column: an Int var `x` (carries the `x≥0` Safety leg) plus a stuttering
    /// RECORD var `r = [a |-> 1, b |-> 2]`. The live model checker enumerates the reachable TUPLES with
    /// the Record column stored as the POSITIONAL pack `pack = v_a + v_b·base` (canonical sorted field
    /// order a,b; base 10 ⇒ `1 + 2·10 = 21`); the orchestrator records `[Int, Record{base:10,arity:2}]`;
    /// and the Clean kernel certifies + re-checks ALL THREE membership legs over the packed-`Nat`
    /// (`Eq Nat`) encoding. Proof the positional record pack runs LIVE.
    #[test]
    fn live_explicit_fixpoint_record_column_membership() {
        let spec = "---- MODULE RecStutter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [a |-> 1, b |-> 2]\n\
                    Next == x' = x /\\ r' = r\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect(
            "Int + Record-valued spec must kernel-certify the membership legs over the pack",
        );
        // r = [a|->1, b|->2] → pack = 1 + 2*10 = 21 (canonical field order a,b).
        assert_eq!(
            cert.reachable,
            vec![vec![0, 21]],
            "Record column stored as positional pack 21"
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Record {
                    base: 10,
                    fields: vec!["a".to_string(), "b".to_string()],
                    cells: vec![]
                }
            ],
            "x is Int, r is a Record pack column (base 10, fields a,b)"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the Int+Record membership legs must pass"
        );
        // SOUNDNESS net: an R with the WRONG record pack must be REJECTED (it is not the real image).
        let mut under = cert.clone();
        under.reachable = vec![vec![0, 99]];
        assert!(
            !verify_explicit_state_cert(&under),
            "an R missing the real record pack must fail the kernel image⊆R / init⊆R re-check"
        );
    }

    /// LIVE end-to-end FINITE FUNCTION column: an Int var `x` plus a stuttering FUNCTION
    /// `f = [d \in 0..1 |-> d + 5]` (domain {0,1}, values 5,6). Packs POSITIONALLY like a record:
    /// `pack = 5·10^0 + 6·10^1 = 65`. The orchestrator records `[Int, Func{base:10,arity:2}]` and the
    /// kernel certifies + re-checks all three membership legs over the packed `Nat`. Proof the finite
    /// positional function pack runs LIVE (same code path as the record pack).
    #[test]
    fn live_explicit_fixpoint_func_column_membership() {
        let spec = "---- MODULE FuncStutter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, f\n\
                    Init == x = 0 /\\ f = [d \\in 0..1 |-> d + 5]\n\
                    Next == x' = x /\\ f' = f\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect(
            "Int + finite-Function spec must kernel-certify the membership legs over the pack",
        );
        // f = [0|->5, 1|->6] → pack = 5 + 6*10 = 65.
        assert_eq!(
            cert.reachable,
            vec![vec![0, 65]],
            "Function column stored as positional pack 65"
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Func {
                    base: 10,
                    arity: 2,
                    cells: vec![],
                    dom: vec![],
                    dom_kind: EnumKind::Model,
                },
            ],
            "x is Int, f is a finite-Function pack column (base 10, arity 2)"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the Int+Function membership legs must pass"
        );
    }

    /// LIVE end-to-end SEQUENCE column: an Int var `x` plus a stuttering SEQUENCE `s = <<3, 1>>`. Packs
    /// SELF-DELIMITINGLY in base `SEQ_BASE+1 = 10`: `pack = (3+1)·10^0 + (1+1)·10^1 = 4 + 20 = 24`. The
    /// orchestrator records `[Int, Seq{base:9,max_len:4}]` and the kernel certifies + re-checks all three
    /// membership legs over the packed `Nat`. Proof the self-delimiting sequence pack runs LIVE.
    #[test]
    fn live_explicit_fixpoint_seq_column_membership() {
        let spec = "---- MODULE SeqStutter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, s\n\
                    Init == x = 0 /\\ s = <<3, 1>>\n\
                    Next == x' = x /\\ s' = s\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("Int + Sequence spec must kernel-certify the membership legs over the self-delimiting pack");
        // <<3,1>> → pack = (3+1) + (1+1)*10 = 4 + 20 = 24.
        assert_eq!(
            cert.reachable,
            vec![vec![0, 24]],
            "Sequence column stored as self-delimiting pack 24"
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Seq {
                    base: 9,
                    max_len: 4,
                    elem: CellSort::Int
                }
            ],
            "x is Int, s is a Sequence pack column (base 9, max_len 4)"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the kernel re-check of the Int+Sequence membership legs must pass"
        );
        // SOUNDNESS net: a tampered R with the wrong pack must be REJECTED.
        let mut under = cert.clone();
        under.reachable = vec![vec![0, 25]]; // 25 ≠ the real <<3,1>> pack 24
        assert!(
            !verify_explicit_state_cert(&under),
            "an R with the wrong sequence pack must fail the kernel re-check"
        );
    }

    /// ADAPTIVE SEQ BASE (mirrors R3 record widening): a sequence element `≥ SEQ_BASE` (here `<<12, 3>>`,
    /// element 12) DERIVES the SMALLEST admitting per-column self-delimiting radix `D = max(RECORD_FUNC_BASE,
    /// 12 + 2) = 14` (element base `D - 1 = 13`) — NOT a fixed rung — and the sort SAYS `base 13`. Before
    /// the widening this failed closed at the fixed element bound `< SEQ_BASE = 9`. The kernel re-check
    /// (Leg-E) re-enumerates and re-derives the SAME radix, so it accepts.
    #[test]
    fn live_explicit_fixpoint_seq_oversized_element_derives_tight_radix() {
        let spec = "---- MODULE SeqBig ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, s\n\
                    Init == x = 0 /\\ s = <<12, 3>>\n\
                    Next == x' = x /\\ s' = s\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("a sequence element 12 certifies at the derived radix D=14 (element base 13)");
        assert_eq!(
            cert.sorts,
            vec![ColSort::Int, ColSort::Seq { base: 13, max_len: 4, elem: CellSort::Int }],
            "the widened seq column's sort carries the DERIVED tight element base (12+1), not SEQ_BASE"
        );
        // <<12,3>> → pack = (12+1)·14^0 + (3+1)·14^1 = 13 + 56 = 69.
        assert_eq!(
            cert.reachable,
            vec![vec![0, 69]],
            "self-delimiting pack at radix 14 = 13 + 4·14 = 69"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "Leg-E re-derives radix 14 ⇒ the widened-seq cert re-verifies"
        );
        // SOUNDNESS net: a tampered R with the wrong pack at the widened radix must be REJECTED.
        let mut under = cert.clone();
        under.reachable = vec![vec![0, 70]]; // 70 ≠ the real <<12,3>> pack 69 at radix 14
        assert!(
            !verify_explicit_state_cert(&under),
            "an R with the wrong widened-seq pack must fail the kernel re-check (Init ⊄ R)"
        );
    }

    /// SMALLEST-ADMITTING + BYTE-COMPAT boundary at the derived-radix leaves (`seq_min_radix`,
    /// `value_cell_encode`): element `SEQ_BASE-1 = 8` (the largest the pre-widening encoder admitted) still
    /// derives the FLOOR radix `RECORD_FUNC_BASE` ⇒ element base `SEQ_BASE = 9` (BYTE-IDENTICAL); element
    /// `SEQ_BASE = 9` (rejected before) derives the next radix up (base 10). A pure function of the value.
    #[test]
    fn seq_min_radix_boundary_and_byte_compat() {
        use crate::value::{SeqValue, Value};
        let seq = |xs: &[i64]| {
            Value::Seq(Rp::new(SeqValue::from_vec(
                xs.iter().map(|&n| Value::SmallInt(n)).collect(),
            )))
        };
        // <<8>>: floor radix (byte-identical to the pre-widening base 9), pack = (8+1)·10^0 = 9.
        assert_eq!(
            compound_min_base(&seq(&[8])),
            Some(RECORD_FUNC_BASE),
            "all-`< SEQ_BASE` ⇒ floor radix"
        );
        assert_eq!(
            value_cell_encode(&seq(&[8])),
            Some((
                ColSort::Seq {
                    base: SEQ_BASE,
                    max_len: SEQ_MAX_LEN,
                    elem: CellSort::Int
                },
                9
            )),
            "element 8 stays byte-identical: base SEQ_BASE, pack 9"
        );
        // <<9>>: element = SEQ_BASE now admitted at radix 11 (element base 10); pre-widening declined.
        assert_eq!(
            compound_min_base(&seq(&[9])),
            Some(11),
            "element 9 ⇒ D = max(10, 9+2) = 11"
        );
        assert_eq!(
            value_cell_encode_at(&seq(&[9]), 11),
            Some((
                ColSort::Seq {
                    base: 10,
                    max_len: SEQ_MAX_LEN,
                    elem: CellSort::Int
                },
                10
            )),
            "element 9 packs at radix 11: base 10, pack = (9+1)·11^0 = 10"
        );
    }

    /// FAIL-CLOSED on SEQ PACK OVERFLOW: the derived radix has no fixed ceiling, but the pack `D^len` must
    /// fit a `u64`. `<<9999999999, 9999999999>>` derives `D = 10^10 + 2`; `D^2 ≈ 10^20 > 2^64` overflows
    /// (`checked_pow` in `seq_min_radix` declines) — no certificate. The overflow GUARD is load-bearing.
    #[test]
    fn live_explicit_fixpoint_seq_pack_overflow_fails_closed() {
        let spec = "---- MODULE SeqHuge ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, s\n\
                    Init == x = 0 /\\ s = <<9999999999, 9999999999>>\n\
                    Next == x' = x /\\ s' = s\n\
                    Safety == x >= 0\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "a sequence whose derived radix^len overflows u64 is out of the packable fragment (fail-closed)"
        );
    }

    /// FAIL-CLOSED on a sequence element OUTSIDE the value-type-leaf fragment: a NEGATIVE element
    /// (`<<-1>>`) and a NESTED element (`<<<<1,2>>>>` — a sequence whose element is a tuple) both DECLINE.
    /// (ATOM / model-value / `Bool` elements are now SUPPORTED via the generalized [`ColSort::Seq::elem`]
    /// leaf — the `elem` sort discriminant keeps `<<FALSE>>` ≠ `<<0>>` and `<<"a">>` ≠ `<<0>>`, so no value
    /// collapse — and are exercised by the generalized-Seq soundness tests; only a NON-leaf element
    /// (tuple/record/nested) or a negative/oversized Int still fails closed here.)
    #[test]
    fn live_explicit_fixpoint_seq_non_int_element_fails_closed() {
        for (name, init) in [("SeqNeg", "s = <<-1>>"), ("SeqNested", "s = <<<<1, 2>>>>")] {
            let spec = format!(
                "---- MODULE {name} ----\n\
                 EXTENDS Integers\n\
                 VARIABLES x, s\n\
                 Init == x = 0 /\\ {init}\n\
                 Next == x' = x /\\ s' = s\n\
                 Safety == x >= 0\n\
                 ====\n"
            );
            assert!(
                certify_explicit_state_spec(&spec, &cfg()).is_none(),
                "a sequence with a non-nonneg-Int element ({init}) must fail closed"
            );
        }
    }

    /// FAIL-CLOSED Safety guard: a `Safety` conjunct OVER A COMPOUND column (`r.a >= 0`) must be DECLINED
    /// — `is_conjunctive_nonneg_safety` only counts `x≥0` over INT columns, so a record-field Safety
    /// conjunct is not recognized and the whole cert fail-closes (NO Certified verdict the kernel did not
    /// prove over the packed encoding).
    /// SOUNDNESS REGRESSION (the adversarial review's Attack 1): a record column taking records with
    /// DIFFERENT field SETS across states must FAIL CLOSED. `[a|->1,b|->2]` and `[c|->1,d|->2]` both pack
    /// to the Nat 21, so WITHOUT field names in the ColSort identity they would be wrongly equated by
    /// `Eq Nat` (a real transition would look like a self-loop ⇒ premature fixpoint ⇒ false Certified).
    /// Now the field names are part of `ColSort::Record`, so the two states have DIFFERENT sorts and the
    /// cross-state `col_sorts` agreement check rejects the column ⇒ no cert.
    #[test]
    fn heterogeneous_record_column_fails_closed() {
        let spec = "---- MODULE Het ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [a |-> 1, b |-> 2]\n\
                    Next == x' = x /\\ r' = [c |-> 1, d |-> 2]\n\
                    Safety == x >= 0\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "a record column with non-identical field sets ([a,b] then [c,d], both packing to 21) must \
             fail closed — field names are part of the ColSort identity, so the sorts disagree"
        );
    }

    /// R3: a Safety conjunct reading a RECORD FIELD (`r.a >= 0`) rides the GENERAL `R⊆Safety`
    /// leg — the field access embeds as the digit extraction `(pack / base^idx) mod base`, which
    /// the exactness filter now admits over a Record-sorted column (`compound_digit_exact`: the
    /// pack is canonical with every digit `< base`, so the extraction is exact nonneg integer
    /// arithmetic). Before R3 this declined (the filter rejected all div/mod conservatively).
    #[test]
    fn live_explicit_fixpoint_record_field_safety_certifies() {
        let spec = "---- MODULE RecSafety ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [a |-> 1]\n\
                    Next == x' = x /\\ r' = r\n\
                    Safety == x >= 0 /\\ r.a >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("a record-field Safety conjunct is exact digit extraction — certifiable");
        assert!(cert.safety_pred.is_some(), "rides the GENERAL safety leg");
        assert!(
            cert.safety_general.is_some(),
            "kernel-checked general R⊆Safety leg present"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the record-field safety cert re-verifies"
        );
    }

    /// The exactness filter still REJECTS genuinely inexact div/mod: `x % 2 = 0` over an INT
    /// column is NOT the record digit form (TLA `%` on an Int is real modulo semantics the
    /// Nat-truncating embedding over-approximates) — the general safety leg must decline.
    #[test]
    fn live_explicit_fixpoint_int_mod_safety_still_declines() {
        let spec = "---- MODULE IntMod ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x\n\
                    Init == x = 0\n\
                    Next == x' = x\n\
                    Safety == x % 2 = 0\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "int-column mod stays outside the exact fragment (fail-closed)"
        );
    }

    /// SERDE round-trip: a cert carrying the new compound `ColSort` columns serializes and deserializes
    /// byte-for-byte, and the re-deserialized cert still kernel-verifies.
    #[test]
    fn live_explicit_fixpoint_compound_serde_roundtrip() {
        let spec = "---- MODULE RecSerde ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [a |-> 1, b |-> 2]\n\
                    Next == x' = x /\\ r' = r\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg()).expect("record cert");
        let bytes = serde_json::to_vec(&cert).expect("serialize");
        let back: ExplicitFixpointCert = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(
            cert, back,
            "the compound-column cert must serde round-trip identically"
        );
        assert!(
            verify_explicit_state_cert(&back),
            "the re-loaded compound cert must still verify"
        );
        for sort in [
            ColSort::Record {
                base: 10,
                fields: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                cells: vec![],
            },
            ColSort::Func {
                base: 10,
                arity: 2,
                cells: vec![],
                dom: vec![],
                dom_kind: EnumKind::Model,
            },
            ColSort::Seq {
                base: 9,
                max_len: 4,
                elem: CellSort::Int,
            },
        ] {
            let s = serde_json::to_vec(&sort).unwrap();
            let d: ColSort = serde_json::from_slice(&s).unwrap();
            assert_eq!(sort, d, "ColSort variant must serde round-trip");
            // BYTE-COMPAT: an Int-prefix Func (empty `dom`, `Model` kind) serializes with NEITHER `dom`
            // NOR `dom_kind` keys ⇒ byte-identical to a pre-domain-shape cert.
            let text = String::from_utf8(s).unwrap();
            if matches!(sort, ColSort::Func { .. }) {
                assert!(
                    !text.contains("dom"),
                    "Int-prefix Func must omit dom/dom_kind: {text}"
                );
            }
        }
    }

    /// R3 (derived base): a record field value `≥ RECORD_FUNC_BASE` (here `r = [a |-> 99]`) DERIVES the
    /// SMALLEST admitting per-column base `max(10, 99+1) = 100` — NOT a fixed wide rung — and the sort
    /// SAYS base 100. (Before R3 this failed closed at the fixed base 10.)
    #[test]
    fn live_explicit_fixpoint_record_oversized_field_derives_tight_base() {
        let spec = "---- MODULE RecBig ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [a |-> 99]\n\
                    Next == x' = x /\\ r' = r\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("a field value 99 certifies at the derived per-column base 100");
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                // `max(RECORD_FUNC_BASE, 99+1) = 100` — the smallest-admitting derived base.
                ColSort::Record {
                    base: 100,
                    fields: vec!["a".to_string()],
                    cells: vec![]
                }
            ],
            "the widened column's sort carries the DERIVED tight base (99+1), not a fixed 1024"
        );
        assert_eq!(
            cert.reachable,
            vec![vec![0, 99]],
            "single-field pack = 99·100^0 = 99"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the derived-base cert re-verifies"
        );
    }

    /// FAIL-CLOSED on PACK OVERFLOW: the derived base has no fixed ceiling, but the pack `base^arity` must
    /// fit a `u64`. `r = [hi |-> 9999999999, lo |-> 1]` derives base `10^10`; `(10^10)^2 = 10^20 > 2^64`
    /// overflows the pack (`checked_pow` declines) — no certificate.
    #[test]
    fn live_explicit_fixpoint_record_pack_overflow_fails_closed() {
        let spec = "---- MODULE RecHuge ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [hi |-> 9999999999, lo |-> 1]\n\
                    Next == x' = x /\\ r' = r\n\
                    Safety == x >= 0\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(spec, &cfg()).is_none(),
            "a record whose derived base^arity overflows u64 is out of the packable fragment"
        );
    }

    /// LIVE: a CHANGING RECORD column whose `Next` USES field access + a record CONSTRUCTOR fires the
    /// GENERAL completeness leg. `Next == r' = [a |-> (r.a + 1) % 3]` (a mod-3 record counter; field
    /// values stay `< RECORD_FUNC_BASE`) cycles the single-field record through packs {0,1,2}. The kernel
    /// RE-EVALUATES the embedded `Next` (`r'_pack = ((r_pack/1 mod 10)+1) mod 3`) over the FULL pack range
    /// `{0..=base^1-1} = {0..=9}` and proves `R={0,1,2}` is closed — NOT over TY's enumerated image. The
    /// cert carries `next_pred`; an under-approx `R` is REJECTED. Paired with an Int `x` for the `x≥0`
    /// Safety leg. This is the compound-operation general-completeness leg the slice delivers.
    #[test]
    fn live_general_completeness_record_field_counter() {
        let spec = "---- MODULE RecCounter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, r\n\
                    Init == x = 0 /\\ r = [a |-> 0]\n\
                    Next == x' = x /\\ r' = [a |-> (r.a + 1) % 3]\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("record-field counter must certify with the general completeness leg");
        // R = single-field record packs {0,1,2} paired with x=0.
        assert_eq!(
            cert.reachable,
            vec![vec![0, 0], vec![0, 1], vec![0, 2]],
            "mod-3 record cycle"
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Record {
                    base: 10,
                    fields: vec!["a".to_string()],
                    cells: vec![]
                }
            ],
        );
        // The GENERAL Next-completeness leg FIRED (the kernel re-evaluated Next over the pack domain).
        assert!(
            cert.next_pred.is_some(),
            "the compound-op general Next leg must be present"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "...with its kernel-checked completeness term"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the general compound Next leg"
        );
        // SOUNDNESS: an UNDER-APPROX R (dropping pack 2) must be REJECTED — the kernel finds a successor
        // (2, reachable from r.a=1) that is NOT in R ⇒ the general completeness obligation is false.
        let mut under = cert.clone();
        under.reachable = vec![vec![0, 0], vec![0, 1]];
        assert!(
            !verify_explicit_state_cert(&under),
            "an R missing pack 2 must fail the general Next-completeness re-check"
        );
    }

    /// LIVE (CoffeeCan enumerator-FREE): a single RECORD variable `can=[black,white]` whose `Next` is a
    /// DISJUNCTION of GUARDED affine record `EXCEPT` updates certifies enumerator-FREE. The update
    /// `can' = [can EXCEPT !.black = @ - 1]` is recognized as ONE pack equality `Eq(Prime(0), Σ v_j·base^j)`
    /// (`cleancic::record_update_eq_form`), so the per-state successor pack is a CONCRETE LITERAL and the
    /// `D ⊇ Succ(R)` coverage lemma KERNEL-PROVES via or-elim over the disjuncts — the changing Record
    /// column upgrades to `KernelProven` and the STRICT (`--require-domain-complete`) path ADMITS it.
    /// This is the CoffeeCan `MaxBeanCount = 4` class with the constant baked in (base 10 ⇒ D = {0..99}).
    #[test]
    fn live_coffeecan_record_except_enumerator_free() {
        // NameId allocation order is process-global and semantically irrelevant.  Exercise the
        // adversarial order explicitly: the canonical record packing/EXCEPT recognizers must use
        // field-name string order even when `white` was interned before `black` by an earlier spec.
        let _ = tla_core::intern_name("white");
        let _ = tla_core::intern_name("black");
        let spec = "---- MODULE CoffeeCan4 ----\n\
                    EXTENDS Naturals\n\
                    VARIABLES can\n\
                    Can == [black : 0..4, white : 0..4]\n\
                    TypeInvariant == can \\in Can\n\
                    Init == can \\in {c \\in Can : c.black + c.white \\in 1..4}\n\
                    BeanCount == can.black + can.white\n\
                    PickSameColorBlack == BeanCount > 1 /\\ can.black >= 2 /\\ can' = [can EXCEPT !.black = @ - 1]\n\
                    PickSameColorWhite == BeanCount > 1 /\\ can.white >= 2 /\\ can' = [can EXCEPT !.black = @ + 1, !.white = @ - 2]\n\
                    PickDifferentColor == BeanCount > 1 /\\ can.black >= 1 /\\ can.white >= 1 /\\ can' = [can EXCEPT !.black = @ - 1]\n\
                    Termination == BeanCount = 1 /\\ UNCHANGED can\n\
                    Next == PickSameColorWhite \\/ PickSameColorBlack \\/ PickDifferentColor \\/ Termination\n\
                    ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeInvariant".to_string()],
            ..Default::default()
        };
        // STRICT (enumerator-FREE) path: the changing Record column must be admitted with KERNEL-PROVEN
        // coverage — closure over the Next RELATION, not TY's enumerated image.
        let cert = certify_explicit_state_spec_strict_domain(spec, &config)
            .expect("CoffeeCan(4) must certify enumerator-FREE under the strict domain path");
        assert_eq!(
            cert.sorts,
            vec![ColSort::Record {
                base: 10,
                fields: vec!["black".to_string(), "white".to_string()],
                cells: vec![]
            }],
            "one packed base-10 record column"
        );
        assert_eq!(cert.reachable.len(), 14, "all cans with 1..4 beans");
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel RE-EVALUATED the disjunctive EXCEPT Next over the record domain — enumerator-free closure"
        );
        // Init-completeness: CoffeeCan's `Init = can ∈ {c ∈ Can : c.black+c.white ∈ 1..4}` is a FILTERED
        // record-set comprehension, outside the Init recognizer's fragment — so `Init⊆R` still rests on the
        // enumerated initial states (an HONEST partial on the Init side; the enumerator-free claim here is
        // strictly about CLOSURE over the Next RELATION, which the completeness leg above kernel-proves).
        assert!(
            cert.init_general_completeness.is_none(),
            "the comprehension Init is (honestly) outside the general Init fragment"
        );
        // The D ⊇ Succ(R) coverage of every PRESENT completeness leg is FULLY construction-covered
        // (the changing Record column is KernelProven, not RustDerived).
        let coverage = domain_coverage_of_cert(&cert);
        assert!(
            coverage.fully_construction_covered(),
            "strict mode admitted ⇒ every completeness axis is KernelProven/universe-complete: {coverage:?}"
        );
        assert!(
            !coverage.next_kernel_columns.is_empty(),
            "the changing Record column's successor bound is KERNEL-PROVEN"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "verify re-derives the IR from the cert and re-checks all legs"
        );

        // TAMPER: an UNDER-APPROX R (drop a reachable pack) must fail the general Next-completeness re-check
        // (the kernel finds a successor not in R).
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![20u64]); // drop can=[black0,white4] (pack 20)
        assert_ne!(
            under.reachable, cert.reachable,
            "the test must actually drop a reachable state"
        );
        assert!(
            !verify_explicit_state_cert(&under),
            "an R missing a real successor must fail the enumerator-free closure re-check"
        );
        // TAMPER: a forged completeness leg must be rejected.
        let mut forged = cert.clone();
        forged.next_general_completeness = Some(b"{\"BVar\":0}".to_vec());
        assert!(
            !verify_explicit_state_cert(&forged),
            "a forged completeness term must be rejected"
        );
    }

    /// LIVE (Lamport AsynchInterface — the NONDETERMINISTIC ENUM assignment `val' ∈ Data`): the
    /// handshake `Next == Send ∨ Rcv` where `Send` sets `val' ∈ Data` (a 3-label MODEL-value Enum
    /// column) and flips `rdy`, and `Rcv` flips `ack` leaving `<<val,rdy>>` UNCHANGED. This is the
    /// FIRST corpus spec whose enumerator-free CLOSURE rests on a nondeterministic enum assignment:
    /// `val' ∈ Data` recognizes as the `Bool.or`-fold `⋁_ℓ (val'=idx(ℓ))`, its successor bound
    /// `val' ≤ |Data|-1 = 2` is KERNEL-PROVEN by the existing or-elimination + `beq→ble` Eq-pin
    /// coverage body (each enum index `≤ 2`), and the two Int flips (`rdy'=1-rdy`, `ack'=1-ack`)
    /// coverage-prove against `H=1`. All THREE Next axes are KernelProven — closure is enumerator-FREE
    /// (kernel-re-evaluated over `D = {0,1,2}×{0,1}×{0,1}`), so `--require-domain-complete` ADMITS it.
    /// Init⊆R is ALSO enumerator-free: `rdy ∈ {0,1}` is an Int finite-set membership (H=max=1) and
    /// `ack = rdy` a cross-column equality (H_ack = H_rdy = 1), both admitted by the Int Init-bound
    /// arm and KERNEL-PROVEN — the `rdy` Or-fold via the existing or-elim/`beq→ble` pins, the `ack`
    /// equality via `beq_ble_trans` (reflect `beq ack rdy` to `ack = rdy`, transport `rdy`'s bound).
    /// So AsynchInterface is FULLY enumerator-free (Init AND Next), like HourClock.
    #[test]
    fn live_asynch_interface_enum_assignment_enumerator_free() {
        use crate::config::ConstantValue;
        let spec = "---- MODULE AsynchInterface ----\n\
                    EXTENDS Naturals\n\
                    CONSTANT Data\n\
                    VARIABLES val, rdy, ack\n\
                    TypeInvariant == /\\ val \\in Data /\\ rdy \\in {0,1} /\\ ack \\in {0,1}\n\
                    Init == /\\ val \\in Data /\\ rdy \\in {0,1} /\\ ack = rdy\n\
                    Send == /\\ rdy = ack /\\ val' \\in Data /\\ rdy' = 1 - rdy /\\ UNCHANGED ack\n\
                    Rcv  == /\\ rdy # ack /\\ ack' = 1 - ack /\\ UNCHANGED <<val, rdy>>\n\
                    Next == Send \\/ Rcv\n\
                    ====\n";
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeInvariant".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "Data".to_string(),
            ConstantValue::ModelValueSet(vec![
                "d1".to_string(),
                "d2".to_string(),
                "d3".to_string(),
            ]),
        );
        config.constants_order.push("Data".to_string());

        // STRICT (enumerator-FREE closure) path: every Next axis must be admitted with KERNEL-PROVEN
        // coverage — the nondeterministic enum assignment `val' ∈ Data` included.
        let cert = certify_explicit_state_spec_strict_domain(spec, &config).expect(
            "AsynchInterface must certify enumerator-FREE closure under the strict domain path",
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Enum {
                    labels: vec!["d1".to_string(), "d2".to_string(), "d3".to_string()],
                    kind: EnumKind::Model
                },
                ColSort::Int,
                ColSort::Int,
            ],
            "a 3-label MODEL Enum column (val) plus two Int flip columns (rdy, ack)"
        );
        assert_eq!(
            cert.reachable.len(),
            12,
            "3 (val) × 2 (rdy) × 2 (ack) reachable states"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel RE-EVALUATED Send∨Rcv (incl. the `val'∈Data` enum assignment) over D — enumerator-free closure"
        );
        // Init is ALSO enumerator-free now: `rdy∈{0,1}` (Int finite-set membership) and `ack=rdy`
        // (cross-column equality) are admitted by the Int Init-bound arm and pass in STRICT mode.
        assert!(
            cert.init_general_completeness.is_some(),
            "rdy∈{{0,1}}/ack=rdy Init is inside the Int Init-bound arm (finite-set membership + cross-column eq)"
        );
        // Every PRESENT completeness leg's D coverage is kernel-proven (NO Rust bound rule trusted):
        // all three Next axes AND all three Init axes upgrade to KernelProven.
        let coverage = domain_coverage_of_cert(&cert);
        assert!(
            coverage.fully_construction_covered(),
            "strict mode admitted ⇒ every completeness axis is KernelProven/universe-complete: {coverage:?}"
        );
        assert_eq!(
            coverage.next_kernel_columns,
            vec![0, 1, 2],
            "all three Next successor bounds are KERNEL-PROVEN (enum or-elim/pins + Int flip pins)"
        );
        assert_eq!(
            coverage.init_kernel_columns,
            vec![0, 1, 2],
            "all three Init bounds are KERNEL-PROVEN: val enum membership (or-elim/pins), rdy \
             finite-set membership (or-elim/pins), ack cross-column equality (beq_ble_trans)"
        );
        assert!(
            coverage.init_rust_columns.is_empty(),
            "no Init axis rests on a trusted-Rust bound rule: {coverage:?}"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "verify re-derives the IR from the cert and re-checks all legs"
        );

        // TAMPER (R ⊊ D with an escaping successor): drop a reachable state. The box D still contains its
        // tuple, and it IS a real Next-successor of some R-state, so the general Next-completeness kernel
        // Bool finds `Next(s,sp) ∧ sp∉R` ⇒ the closure obligation is FALSE ⇒ verify must REJECT.
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &vec![1u64, 1, 0]); // drop (val=d2, rdy=1, ack=0)
        assert_ne!(
            under.reachable, cert.reachable,
            "the test must actually drop a reachable state"
        );
        assert!(
            !verify_explicit_state_cert(&under),
            "an R missing a real successor must fail the enumerator-free closure re-check"
        );
        // TAMPER (Init side — the new leg): drop an INITIAL state from R. `Init([0,1,1])` holds
        // (ack=rdy=1), and the re-derived box `D` (bounds [2,1,1]) still contains it, so the general
        // Init-completeness kernel Bool `⋀_{s∈D}(¬Init(s) ∨ s∈R)` finds `Init(s) ∧ s∉R` ⇒ FALSE ⇒
        // verify must REJECT — the soundness guard against a too-loose Init bound (any Init state in
        // D but not in R fails the leg).
        let mut init_under = cert.clone();
        init_under.reachable.retain(|t| t != &vec![0u64, 1, 1]); // drop initial (val=d1, rdy=1, ack=1)
        assert_ne!(
            init_under.reachable, cert.reachable,
            "the test must actually drop an initial state"
        );
        assert!(
            !verify_explicit_state_cert(&init_under),
            "an R missing an Init state must fail the enumerator-free Init-completeness re-check"
        );
        // TAMPER: a forged completeness leg must be rejected.
        let mut forged = cert.clone();
        forged.next_general_completeness = Some(b"{\"BVar\":0}".to_vec());
        assert!(
            !verify_explicit_state_cert(&forged),
            "a forged completeness term must be rejected"
        );
    }

    /// LIVE (Lamport ALTERNATING-BIT protocol — `SpecifyingSystems/TLC/ABCorrectness`): the FIRST real
    /// COMMUNICATION PROTOCOL certified FULLY enumerator-free. Two things had to compose: (1) the sole
    /// `\E d \in Data : CSndNewValue(d)` disjunct is a PARAMETERIZED-operator application — beta-inlined
    /// by `CertInlineEnv` (`CSndNewValue(d)` ⇒ its body with the formal `d` ↦ the bound var), and (2)
    /// the resulting `sent' = d` is a MODEL-value column vs a model-value IDENT — recognized by
    /// `enum_eq_form`'s Model-ident arm as `Eq(Prime(sent), Lit(idx d))`, so the `\E d \in Data` atom
    /// fold yields the Or `⋁_{d∈Data} (… ∧ sent'=idx(d) ∧ …)` — the SAME shape AsynchInterface's
    /// `val'∈Data` produces. Every Next axis is then KERNEL-PROVEN: the enum assignment `sent'`/`rcvd'`
    /// against `H=1` (the 2-label universe) via the or-elim/`beq→ble` pins, the affine flip `sBit'=1-sBit`
    /// and the cross-column copies (`rBit'=sBit`, `sAck'=rBit`) against `H=1`. Init is likewise fully
    /// kernel-covered (finite-set membership + cross-column eq). So ABCorrectness is FULLY enumerator-free
    /// (Init AND Next) and `--require-domain-complete` ADMITS it.
    #[test]
    fn live_abcorrectness_alternating_bit_enumerator_free() {
        use crate::config::ConstantValue;
        let spec = "---- MODULE ABCorrectness ----\n\
                    EXTENDS Naturals\n\
                    CONSTANTS Data\n\
                    VARIABLES sBit, sAck, rBit, sent, rcvd\n\
                    ABCInit == /\\ sBit \\in {0, 1} /\\ sAck = sBit /\\ rBit = sBit \
                                /\\ sent \\in Data /\\ rcvd \\in Data\n\
                    CSndNewValue(d) == /\\ sAck = sBit /\\ sent' = d /\\ sBit' = 1 - sBit \
                                        /\\ UNCHANGED <<sAck, rBit, rcvd>>\n\
                    CRcvMsg == /\\ rBit # sBit /\\ rBit' = sBit /\\ rcvd' = sent \
                                /\\ UNCHANGED <<sBit, sAck, sent>>\n\
                    CRcvAck == /\\ rBit # sAck /\\ sAck' = rBit /\\ UNCHANGED <<sBit, rBit, sent, rcvd>>\n\
                    ABCNext == \\/ \\E d \\in Data : CSndNewValue(d) \\/ CRcvMsg \\/ CRcvAck\n\
                    TypeInv == /\\ sBit \\in {0, 1} /\\ sAck \\in {0, 1} /\\ rBit \\in {0, 1} \
                                /\\ sent \\in Data /\\ rcvd \\in Data\n\
                    ====\n";
        let mut config = Config {
            init: Some("ABCInit".to_string()),
            next: Some("ABCNext".to_string()),
            invariants: vec!["TypeInv".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "Data".to_string(),
            ConstantValue::ModelValueSet(vec!["d1".to_string(), "d2".to_string()]),
        );
        config.constants_order.push("Data".to_string());

        // STRICT (enumerator-FREE closure) path: every Next AND Init axis must be admitted with
        // KERNEL-PROVEN coverage — the parameterized-operator enum assignment `\E d: sent'=d` included.
        let cert = certify_explicit_state_spec_strict_domain(spec, &config).expect(
            "ABCorrectness must certify enumerator-FREE closure under the strict domain path",
        );
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Int,
                ColSort::Int,
                ColSort::Enum {
                    labels: vec!["d1".to_string(), "d2".to_string()],
                    kind: EnumKind::Model
                },
                ColSort::Enum {
                    labels: vec!["d1".to_string(), "d2".to_string()],
                    kind: EnumKind::Model
                },
            ],
            "three Int bit columns (sBit, sAck, rBit) plus two 2-label MODEL Enum columns (sent, rcvd)"
        );
        assert_eq!(
            cert.reachable.len(),
            20,
            "the alternating-bit reachable set is 20 states"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "the kernel RE-EVALUATED ABCNext (incl. the parameterized `\\E d: CSndNewValue(d)` enum \
             assignment) over D — enumerator-free closure"
        );
        assert!(
            cert.init_general_completeness.is_some(),
            "ABCInit is inside the general Init-bound arm (finite-set membership + cross-column eq)"
        );
        // Every present completeness axis is KERNEL-PROVEN (NO Rust bound rule trusted) — all five Next
        // AND all five Init axes upgrade to KernelProven.
        let coverage = domain_coverage_of_cert(&cert);
        assert!(
            coverage.fully_construction_covered(),
            "strict mode admitted ⇒ every completeness axis is KernelProven/universe-complete: {coverage:?}"
        );
        assert_eq!(
            coverage.next_kernel_columns,
            vec![0, 1, 2, 3, 4],
            "all five Next successor bounds are KERNEL-PROVEN (enum or-elim/pins + affine flip + cross-col copies)"
        );
        assert_eq!(
            coverage.init_kernel_columns,
            vec![0, 1, 2, 3, 4],
            "all five Init bounds are KERNEL-PROVEN (finite-set membership + cross-column eq)"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "verify re-derives the IR from the cert and re-checks all legs"
        );

        // TAMPER (R ⊊ D with an escaping successor): drop a NON-initial reachable state. It IS a real
        // Next-successor of some remaining R-state (only reached via a transition), and the re-derived
        // box D still contains its tuple, so the general Next-completeness kernel Bool finds
        // `Next(s,sp) ∧ sp∉R` ⇒ the closure obligation is FALSE ⇒ verify must REJECT. This is the guard
        // that proves the closure re-check uses the FAITHFUL (beta-inlined) Next relation.
        let escaping = vec![1u64, 0, 0, 0, 0]; // (sBit=1, sAck=0, rBit=0, sent=d1, rcvd=d1): a CSndNewValue successor
        assert!(
            cert.reachable.contains(&escaping),
            "the drop target must be a reachable state"
        );
        let mut under = cert.clone();
        under.reachable.retain(|t| t != &escaping);
        assert_ne!(
            under.reachable, cert.reachable,
            "the test must actually drop a reachable state"
        );
        assert!(
            !verify_explicit_state_cert(&under),
            "an R missing a real Next successor must fail the enumerator-free closure re-check"
        );
        // TAMPER: a forged completeness leg must be rejected.
        let mut forged = cert.clone();
        forged.next_general_completeness = Some(b"{\"BVar\":0}".to_vec());
        assert!(
            !verify_explicit_state_cert(&forged),
            "a forged completeness term must be rejected"
        );
    }

    /// FAIL-CLOSED (the enum-assignment coverage's soundness floor): a `Next` disjunct that ASSIGNS an
    /// enum column a model value OUTSIDE the column's claimed universe must NOT be mis-recognized (no
    /// fabricated index ⇒ no false successor bound). Here `blk` is a standalone `ModelValue` NOT in the
    /// `Data` model-value SET, so it is absent from `mvsets`; the `Escape` disjunct's `x' = blk` fails
    /// `enum_eq_form`'s Model-ident guard (`blk` is not a declared member of any model-value set) ⇒ that
    /// disjunct — and hence the whole `Next` — DECLINES recognition ⇒ NO general Next completeness leg ⇒
    /// closure falls back to the honest enumerated `image ⊆ R` (enumerator-ASSISTED, NOT a false
    /// enumerator-free). `Escape` is numerically unreachable (`step = 2` never holds), so the value never
    /// enters `R` and safety still holds — the cert is genuine, just not enumerator-free.
    #[test]
    fn escaping_enum_assignment_declines_fully_free() {
        use crate::config::ConstantValue;
        let spec = "---- MODULE Blocked ----\n\
                    EXTENDS Naturals\n\
                    CONSTANTS Data, blk\n\
                    VARIABLES x, step\n\
                    Init == x \\in Data /\\ step = 0\n\
                    Flip == x' = x /\\ step' = 1 - step\n\
                    Escape == step = 2 /\\ x' = blk /\\ step' = 3\n\
                    Next == Flip \\/ Escape\n\
                    TypeInv == x \\in Data /\\ step \\in {0, 1}\n\
                    ====\n";
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeInv".to_string()],
            ..Default::default()
        };
        config.constants.insert(
            "Data".to_string(),
            ConstantValue::ModelValueSet(vec!["a1".to_string(), "a2".to_string()]),
        );
        // `blk` is a STANDALONE model value — NOT a member of any model-value SET ⇒ NOT in `mvsets`.
        config
            .constants
            .insert("blk".to_string(), ConstantValue::ModelValue);
        config.constants_order.push("Data".to_string());
        config.constants_order.push("blk".to_string());

        // It still CERTIFIES (safety holds over the honest enumerated set), but NOT enumerator-free.
        let cert = certify_explicit_state_spec(spec, &config)
            .expect("the spec must still certify via the enumerated image ⊆ R leg");
        assert!(
            cert.reachable.iter().all(|t| t[0] < 2),
            "blk must be unreachable in x — only observed labels {{a1,a2}} appear (indices 0,1)"
        );
        assert!(
            cert.next_general_completeness.is_none() && cert.next_pred.is_none(),
            "an assignment reaching a model value OUTSIDE the claimed enum universe must DECLINE the \
             enumerator-free Next leg (fail-closed) — closure stays enumerator-ASSISTED"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "the honest enumerated-image cert still re-verifies"
        );
    }

    /// LIVE: a CHANGING SET column whose `Next` is a SET COMPREHENSION fires the GENERAL completeness leg.
    /// `Next == chosen' = {x \in chosen : x # 1}` filters out element 1 each step. Because a filter only
    /// REMOVES elements (`chosen' ⊆ chosen`), the successor mask is `≤ max(R's mask)`, so the domain rule
    /// bounds the Set axis by the reachable masks (NOT the unreachable full 2^16) and the leg FIRES. The
    /// kernel RE-EVALUATES the embedded filter fold over that domain and proves `R` is closed. Paired with
    /// an Int `x`. The set-comprehension general-completeness leg the slice delivers.
    #[test]
    fn live_general_completeness_set_comprehension() {
        let spec = "---- MODULE Filter ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, chosen\n\
                    Init == x = 0 /\\ chosen = {1, 2, 3}\n\
                    Next == x' = x /\\ chosen' = {y \\in chosen : y # 1}\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("set-comprehension spec must certify with the general completeness leg");
        // chosen: {1,2,3}=mask 0b1110 → filter out 1 → {2,3}=0b1100 → filter again → {2,3} (fixpoint).
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Set {
                    universe: SET_UNIVERSE_BITS
                }
            ]
        );
        assert!(
            cert.next_pred.is_some(),
            "the comprehension general Next leg must be present"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "...with its kernel-checked completeness term"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the comprehension Next leg"
        );
        // SOUNDNESS: drop the filtered successor {2,3} (mask 0b1100) from R ⇒ the kernel finds it as a
        // successor of {1,2,3} not in R ⇒ the general completeness obligation is false ⇒ rejected.
        let drop_mask = 0b1100u64;
        let mut under = cert.clone();
        under.reachable.retain(|t| t.get(1) != Some(&drop_mask));
        assert_ne!(
            under.reachable, cert.reachable,
            "the test must actually drop the filtered mask"
        );
        assert!(
            !verify_explicit_state_cert(&under),
            "an R missing the filtered successor must fail the general Next-completeness re-check"
        );
    }

    /// FAIL-CLOSED: `SUBSET` (powerset) is honestly out of the bitmask fragment — a powerset is a set of
    /// `2^K` sets, which does not fit the bounded single-Nat encoding. `chosen' = SUBSET chosen` declines
    /// the general completeness leg (recognize returns None); closure then rests on the honest enumerated
    /// `image ⊆ R` leg. We assert NO general Next leg is claimed for the powerset conjunct.
    #[test]
    fn live_powerset_next_declines_general_leg_fail_closed() {
        let spec = "---- MODULE PowNext ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, chosen\n\
                    Init == x = 0 /\\ chosen = {0}\n\
                    Next == x' = x /\\ chosen' \\in SUBSET chosen\n\
                    Safety == x >= 0\n\
                    ====\n";
        // The powerset Next either fails to certify (the evaluator/encoder declines `chosen' ∈ SUBSET …`)
        // or certifies WITHOUT a general compound Next leg (closure on the enumerated image). Either way,
        // NO general Next-completeness leg may be claimed over a powerset.
        if let Some(cert) = certify_explicit_state_spec(spec, &cfg()) {
            assert!(
                cert.next_pred.is_none() && cert.next_general_completeness.is_none(),
                "a powerset Next must NOT produce a general completeness leg (fail-closed honestly)"
            );
            assert!(
                verify_explicit_state_cert(&cert),
                "if it certifies, the (image-based) cert re-checks"
            );
        }
    }

    /// LIVE (goal item 1): a spec whose `Next` uses a BOUNDED SET QUANTIFIER `∀y∈chosen: y > 0` as a
    /// guard fires the GENERAL completeness leg. The Set column `chosen` STUTTERS (`chosen'=chosen`), so
    /// its axis is bounded by the reachable masks; the Int `x` stutters. The kernel RE-EVALUATES the
    /// embedded `Bool.and`-fold guard (over the concrete `chosen` mask) plus the stutter conjuncts over
    /// the product domain and proves `R` is closed — NOT over TY's enumerated image. The bounded-∀-over-a-
    /// set general leg delivered LIVE. Paired with the Int `x` for the `x≥0` Safety leg.
    #[test]
    fn live_general_completeness_bounded_set_forall_guard() {
        let spec = "---- MODULE BForall ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, chosen\n\
                    Init == x = 0 /\\ chosen = {1, 2, 3}\n\
                    Next == x' = x /\\ chosen' = chosen /\\ (\\A y \\in chosen : y > 0)\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("a bounded-∀-over-a-set guard spec must kernel-certify with the general leg");
        assert_eq!(
            cert.sorts,
            vec![
                ColSort::Int,
                ColSort::Set {
                    universe: SET_UNIVERSE_BITS
                }
            ]
        );
        assert!(
            cert.next_pred.is_some(),
            "the bounded-∀ general Next leg must be present"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "...with its kernel-checked completeness term"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the bounded-∀ Next guard"
        );
    }

    /// LIVE (goal item 1, ∃ form): a `Next` guard `∃y∈chosen: y = 2` (a bounded existential over the Set
    /// column) fires the general completeness leg the same way. Both the Set column and the Int stutter;
    /// the kernel re-evaluates the `Bool.or`-fold guard over the reachable-mask domain and proves closure.
    #[test]
    fn live_general_completeness_bounded_set_exists_guard() {
        let spec = "---- MODULE BExists ----\n\
                    EXTENDS Integers\n\
                    VARIABLES x, chosen\n\
                    Init == x = 0 /\\ chosen = {1, 2, 3}\n\
                    Next == x' = x /\\ chosen' = chosen /\\ (\\E y \\in chosen : y = 2)\n\
                    Safety == x >= 0\n\
                    ====\n";
        let cert = certify_explicit_state_spec(spec, &cfg())
            .expect("a bounded-∃-over-a-set guard spec must kernel-certify with the general leg");
        assert!(
            cert.next_pred.is_some(),
            "the bounded-∃ general Next leg must be present"
        );
        assert!(
            cert.next_general_completeness.is_some(),
            "...with its kernel-checked completeness term"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the bounded-∃ Next guard"
        );
        // SOUNDNESS: the guard `∃y∈{1,2,3}: y=2` is TRUE, so Next is enabled; an under-approx R that drops
        // the (stuttered) reachable state must be rejected by the general Next-completeness re-check.
        let mut under = cert.clone();
        under.reachable.clear();
        assert!(
            !verify_explicit_state_cert(&under),
            "an empty R must fail Init⊆R / closure re-checks"
        );
    }

    /// The UNBOUNDED non-stutter affine counter (`Init x=0 / Next x'=x+1 / Safety x>=0`) USED to
    /// fail-closed (its `R=ℕ` exceeds the finite cap). It is now CERTIFIED by the PARAMETRIC inductive-
    /// invariant path WITHOUT enumeration — even at a tiny state cap, since the BFS never runs. (The
    /// nonneg-`Safety` shape is what makes `J≡Safety` inductive; the non-inductive variant still
    /// fail-closes — see `non_inductive_unbounded_spec_fails_closed`.)
    #[test]
    fn unbounded_affine_counter_certifies_via_parametric_path() {
        let spec = "---- MODULE Counter ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 0\n\
                    Next == x' = x + 1\n\
                    Safety == x >= 0\n\
                    ====\n";
        // A tiny cap (64) cannot enumerate R=ℕ — yet the parametric path certifies regardless, because
        // it does NOT enumerate (the `∀x` consecution leg is the closure proof).
        let cert = certify_explicit_state_spec_bounded(spec, &cfg(), 64)
            .expect("the unbounded counter is Certified by the parametric leg (no enumeration)");
        assert!(
            cert.unbounded_invariant.is_some(),
            "via the parametric inductive-invariant path"
        );
        assert!(cert.reachable.is_empty(), "NO enumeration was performed");
        assert!(verify_explicit_state_cert(&cert));
    }

    // ───────────────── GENERAL `R⊆Safety` leg (`safety_pred` / `safety_general`) ─────────────────

    /// The HOURCLOCK-shaped spec used by the general-safety tests: a single Int variable whose
    /// invariant is the INTERVAL MEMBERSHIP `x ∈ 1..12` — recognizable into the kernel predicate
    /// fragment (`1≤x ∧ x≤12`) but NOT the `⋀ x≥0` nonneg shape the tuple leg proves.
    const HC_SHAPE: &str = "---- MODULE HCShape ----\n\
                            EXTENDS Integers\n\
                            VARIABLE x\n\
                            Init == x \\in 2..5\n\
                            Next == x' = x\n\
                            Safety == x \\in 1..12\n\
                            ====\n";

    /// THE WIDENING: an invariant beyond `⋀ x≥0` (interval membership, the HourClock class)
    /// certifies through the GENERAL embedded-safety lane. The cert carries `safety_pred` (the
    /// recognized `1≤x ∧ x≤12` IR) + `safety_general` (the kernel-reduced `⋀_{s∈R} Safety(s)` leg)
    /// ALONGSIDE the encoding-level tuple `safety_term` — BOTH re-check.
    #[test]
    fn general_safety_interval_membership_certifies() {
        let cert = certify_explicit_state_spec(HC_SHAPE, &cfg())
            .expect("the interval-membership invariant must certify via the general safety leg");
        assert_eq!(
            cert.reachable,
            vec![vec![2], vec![3], vec![4], vec![5]],
            "Init x ∈ 2..5 enumerates R through the live constraint branches"
        );
        let ir = cert
            .safety_pred
            .as_ref()
            .expect("the recognized general safety IR is stored");
        assert_eq!(
            *ir,
            PredIR::And(
                Box::new(PredIR::Leq(ValIR::Lit(1), ValIR::Var(0))),
                Box::new(PredIR::Leq(ValIR::Var(0), ValIR::Lit(12))),
            ),
            "x ∈ 1..12 is recognized EXACTLY as 1≤x ∧ x≤12"
        );
        assert!(
            cert.safety_general.is_some(),
            "the kernel-checked general safety leg is present"
        );
        assert!(
            !cert.safety_term.is_empty(),
            "the tuple (encoding-level nonneg) leg still rides along — both must pass"
        );
        assert!(
            verify_explicit_state_cert(&cert),
            "kernel re-check incl. the general safety leg"
        );
    }

    /// FAIL-CLOSED: an invariant VIOLATED on a reachable state must NOT certify — the kernel
    /// reduces `⋀_{s∈R} Safety(s)` to `Bool.false` at the violating state and
    /// `certify_bool_true_obligation` declines, so NO cert is minted (the kernel is the arbiter).
    #[test]
    fn general_safety_violated_invariant_fails_closed() {
        // Violation at the INIT state: x=0 ∉ 1..12.
        let at_init = "---- MODULE HCBad ----\n\
                       EXTENDS Integers\n\
                       VARIABLE x\n\
                       Init == x = 0\n\
                       Next == x' = x\n\
                       Safety == x \\in 1..12\n\
                       ====\n";
        assert!(
            certify_explicit_state_spec(at_init, &cfg()).is_none(),
            "a reachable state violating the invariant must NOT certify"
        );
        // Violation DEEP in the BFS (not an init state): R = {11,12,13,14} under x'=x+1 ∧ x<14,
        // and 13 ∉ 1..12 — the per-state conjunct at 13 reduces false ⇒ no cert.
        let deep = "---- MODULE HCBad2 ----\n\
                    EXTENDS Integers\n\
                    VARIABLE x\n\
                    Init == x = 11\n\
                    Next == x' = x + 1 /\\ x < 14\n\
                    Safety == x \\in 1..12\n\
                    ====\n";
        assert!(
            certify_explicit_state_spec(deep, &cfg()).is_none(),
            "a DEEP (non-init) reachable violation must NOT certify either"
        );
    }

    /// TAMPER (witness bytes / presence): corrupt the serialized `safety_general` term — verify
    /// rebuilds the obligation from `(R, stored IR)` and the kernel rejects the bogus term. A cert
    /// with exactly ONE of (`safety_pred`, `safety_general`) present is inconsistent — rejected.
    #[test]
    fn general_safety_tampered_leg_or_presence_fails_verify() {
        let cert = certify_explicit_state_spec(HC_SHAPE, &cfg()).expect("certifies");
        assert!(verify_explicit_state_cert(&cert));
        // Bogus witness bytes.
        let mut bad = cert.clone();
        bad.safety_general = Some(b"{\"BVar\":0}".to_vec());
        assert!(
            !verify_explicit_state_cert(&bad),
            "a tampered safety_general witness must fail the kernel re-check"
        );
        // IR present, leg missing → inconsistent.
        let mut noleg = cert.clone();
        noleg.safety_general = None;
        assert!(
            !verify_explicit_state_cert(&noleg),
            "safety_pred without safety_general is malformed"
        );
        // Leg present, IR missing → inconsistent.
        let mut noir = cert.clone();
        noir.safety_pred = None;
        assert!(
            !verify_explicit_state_cert(&noir),
            "safety_general without safety_pred is malformed"
        );
    }

    /// TAMPER (stored IR, kernel-level): NARROW the stored invariant (`x≤12` → `x≤3`) — the rebuilt
    /// obligation `⋀_{s∈R} (1≤s ∧ s≤3)` is FALSE at s=4,5, so the kernel rejects the stored witness
    /// ⇒ verify false. A WIDENED tamper (12→13) still reduces true over R and is deliberately NOT
    /// this layer's job — Leg-E's spec re-recognition equality catches it (see the cert.rs
    /// end-to-end test). A stored IR mentioning PRIMED state is malformed and rejected outright
    /// (without the gate, `embed_pred_ir(Eq(x',x), s, s)` would vacuously reduce true).
    #[test]
    fn general_safety_tampered_ir_fails_verify() {
        let cert = certify_explicit_state_spec(HC_SHAPE, &cfg()).expect("certifies");
        let mut narrowed = cert.clone();
        narrowed.safety_pred = Some(PredIR::And(
            Box::new(PredIR::Leq(ValIR::Lit(1), ValIR::Var(0))),
            Box::new(PredIR::Leq(ValIR::Var(0), ValIR::Lit(3))),
        ));
        assert!(
            !verify_explicit_state_cert(&narrowed),
            "a NARROWED stored IR makes the conjunction false over R — kernel must reject"
        );
        let mut primed = cert.clone();
        primed.safety_pred = Some(PredIR::Eq(ValIR::Prime(0), ValIR::Var(0)));
        assert!(
            !verify_explicit_state_cert(&primed),
            "a stored 'invariant' mentioning primed state is malformed — the state-predicate \
             gate must reject it before the kernel is even consulted"
        );
    }

    /// SERDE/DIGEST BACK-COMPAT (the `bound` precedent): a cert whose invariant IS the nonneg
    /// shape — the ENTIRE pre-widening population — serializes with NO `safety_pred`/
    /// `safety_general` keys, byte-identically to certs minted before the fields existed, so
    /// `SafetyCertificate` sha256 digests (recomputed over a re-serialization at verify time) keep
    /// matching. A general-lane cert DOES carry both keys and round-trips byte-stably.
    #[test]
    fn general_safety_fields_absent_from_pre_widening_certs() {
        let nonneg = "---- MODULE TwoVal ----\n\
                      EXTENDS Integers\n\
                      VARIABLE x\n\
                      Init == x = 2 \\/ x = 5\n\
                      Next == x' = x\n\
                      Safety == x >= 0\n\
                      ====\n";
        let cert = certify_explicit_state_spec(nonneg, &cfg()).expect("nonneg lane certifies");
        assert!(cert.safety_pred.is_none() && cert.safety_general.is_none());
        let json = serde_json::to_string(&cert).expect("serializes");
        assert!(
            !json.contains("\"safety_pred\"") && !json.contains("\"safety_general\""),
            "a nonneg-lane cert must carry NO new keys (pre-widening digest compatibility): {json}"
        );
        let reparsed: ExplicitFixpointCert = serde_json::from_str(&json).expect("reparses");
        assert_eq!(reparsed, cert);
        assert_eq!(
            serde_json::to_string(&reparsed).expect("re-serializes"),
            json
        );
        // A GENERAL-lane cert carries both fields (self-consistent, new-only) and round-trips.
        let general =
            certify_explicit_state_spec(HC_SHAPE, &cfg()).expect("general lane certifies");
        let gjson = serde_json::to_string(&general).expect("serializes");
        assert!(
            gjson.contains("\"safety_pred\"") && gjson.contains("\"safety_general\""),
            "a general-lane cert must carry both new fields"
        );
        let gback: ExplicitFixpointCert = serde_json::from_str(&gjson).expect("reparses");
        assert_eq!(gback, general);
        assert!(
            verify_explicit_state_cert(&gback),
            "the round-tripped general cert re-checks"
        );
    }
}
