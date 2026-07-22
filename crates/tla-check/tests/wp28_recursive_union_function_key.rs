// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-28 end-to-end regression: a RECURSIVE operator's result used as a
//! FUNCTION KEY must read the row the interpreter reads.
//!
//! The miscompile this pins (btree `GetValue`, caught fail-closed by the
//! hybrid shadow differential and diagnosed in WP-25/WP-28):
//!
//! * `ChildNodeFor(node, key)` returns `lastOf[node]` on one arm and
//!   `childOf[node, closestKey]` on the other. Both ranges are the SAME
//!   `Nodes \cup {NIL}` union, but each is cited by a different layout proof,
//!   so their `proof_source` fields differ.
//! * The shape merge at that IF join used to drop the `TaggedScalarUnion`
//!   shape whenever the citations differed, leaving `ChildNodeFor`'s inferred
//!   return shape `None`.
//! * A `TaggedScalarUnion` register physically holds the union-slot INDEX. The
//!   raw-member decode at every consumer is SHAPE-DRIVEN
//!   (`decode_scalar_key_reg_raw_value`), so dropping the shape does not lose
//!   precision conservatively — it silently REINTERPRETS the index as a member
//!   value. `FindLeafNode` re-entered on node `n-1` (the index of node `n`),
//!   and `GetValue`'s `key \in keysOf[node]` / `valOf[node, key]` read the
//!   wrong row: `ret'` came out `MISSING` where the interpreter returned the
//!   stored value.
//!
//! The model is btree itself at the smallest constants that still build a
//! two-level tree (`MaxNode = 5`, `MaxKey = 3`, `Vals = {x, y}` — 5,655
//! states, about a second), because the defect needs a real inner node whose
//! child pointer comes out of the union-ranged `childOf`/`lastOf` pair.
//!
//! Pre-fix this test FAILS on the differential (`mismatch_fallback = 112`,
//! `native_residue = 56`), which is exactly the property the in-crate
//! `tla-ir` unit tests could not pin: `wp20_tagged_extern_return_enabled()`
//! defaults to `cfg!(test)`, so in-crate tests silently ran with the merge
//! enabled while the shipped binary ran with it disabled. This test links
//! `tla-ir` as an ordinary dependency and therefore sees the PRODUCTION
//! default.

mod common;

use tla_check::ModelChecker;
use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

const WP28_TLA: &str = r#"
----------------------------- MODULE wp28btree -----------------------------
EXTENDS TLC,
        Naturals,
        FiniteSets,
        Sequences

CONSTANTS Vals,
          MaxKey,
          MaxNode,
          MaxOccupancy,

          \* states
          READY,
          GET_VALUE,
          FIND_LEAF_TO_ADD,
          WHICH_TO_SPLIT,
          ADD_TO_LEAF,
          SPLIT_LEAF,
          SPLIT_INNER,
          SPLIT_ROOT_LEAF,
          SPLIT_ROOT_INNER,
          UPDATE_LEAF

Keys == 1..MaxKey
Nodes == 1..MaxNode

NIL == CHOOSE x : x \notin Nodes
MISSING == CHOOSE v : v \notin Vals


VARIABLES root,
          isLeaf, keysOf, childOf, lastOf, valOf,
          focus,
          toSplit,
          op, args, ret,
          state

TypeOk == /\ root \in Nodes
          /\ isLeaf \in [Nodes -> BOOLEAN]
          /\ keysOf \in [Nodes -> SUBSET Keys]
          /\ childOf \in [Nodes \X Keys -> Nodes \union {NIL}]
          /\ lastOf \in [Nodes -> Nodes \union {NIL}]
          /\ valOf \in [Nodes \X Keys -> Vals \union {NIL}]
          /\ focus \in Nodes \union {NIL}
          /\ toSplit \in Seq(Nodes)
          /\ op \in {"get", "insert", "update", NIL}
          /\ ret \in Vals \union {"ok", "error", MISSING, NIL}
          /\ state \in {READY, GET_VALUE, FIND_LEAF_TO_ADD, WHICH_TO_SPLIT, ADD_TO_LEAF, SPLIT_LEAF, SPLIT_INNER, SPLIT_ROOT_LEAF, SPLIT_ROOT_INNER, UPDATE_LEAF}

\* Max element in a set
Max(xs) == CHOOSE x \in xs : (\A y \in xs \ {x} : x > y)

\* Find the appropriate child node associated with the key
ChildNodeFor(node, key) ==
    LET keys == keysOf[node]
        maxKey == Max(keys)
        closestKey ==  CHOOSE k \in keys : /\ k>key
                                           /\ ~(\E j \in keys \ {k} : j>key /\ j<k)
    IN IF keys = {} \/ key >= maxKey
       THEN lastOf[node]
       \* smallest k that's bigger than key
       ELSE
       childOf[node, closestKey]


\* Identify the leaf node based on key
\* Find the leaf node associated with a key
RECURSIVE FindLeafNode(_, _)
FindLeafNode(node, key) ==
    IF isLeaf[node] THEN node ELSE FindLeafNode(ChildNodeFor(node, key), key)

AtMaxOccupancy(node) == Cardinality(keysOf[node]) = MaxOccupancy


\* We model a "free" (not yet part of the tree) node as one as a leaf with no keys
IsFree(node) == isLeaf[node] /\ keysOf[node] = {}

ChooseFreeNode == CHOOSE n \in Nodes : IsFree(n)


Init == /\ isLeaf = [n \in Nodes |-> TRUE]
        /\ keysOf = [n \in Nodes |-> {}]
        /\ childOf = [n \in Nodes, k \in Keys |-> NIL]
        /\ lastOf = [n \in Nodes |-> NIL]
        /\ valOf = [n \in Nodes, k \in Keys |-> NIL]
        /\ root = ChooseFreeNode
        /\ focus = NIL
        /\ toSplit = <<>>
        /\ op = NIL
        /\ args = NIL
        /\ ret = NIL
        /\ state = READY

GetReq(key) == 
    /\ state = READY
    /\ op' = "get"
    /\ args' = <<key>>
    /\ ret' = NIL
    /\ state' = GET_VALUE
    /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, valOf, focus, toSplit>>

GetValue ==
    LET key == args[1] 
        node == FindLeafNode(root, key) IN
    /\ state = GET_VALUE
    /\ state' = READY
    /\ ret' = IF key \in keysOf[node] THEN valOf[node, key] ELSE MISSING
    /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, valOf, focus, toSplit, args, op>>
    

InsertReq(key, val) ==
    /\ state = READY
    /\ op' = "insert"
    /\ args' = <<key, val>>
    /\ ret' = NIL
    /\ state' = FIND_LEAF_TO_ADD
    /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, valOf, focus, toSplit>>

UpdateReq(key, val) ==
    LET leaf == FindLeafNode(root, key)
    IN /\ state = READY
       /\ op' = "update"
       /\ args' = <<key, val>>
       /\ ret' = NIL
       /\ focus' = leaf
       /\ state' = UPDATE_LEAF
       /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, valOf, toSplit>>

UpdateLeaf ==
    LET key == args[1]
        val == args[2]
    IN /\ state = UPDATE_LEAF
       /\ valOf' = IF key \in keysOf[focus] THEN [valOf EXCEPT ![focus, key]=val] ELSE valOf
       /\ ret' = IF key \in keysOf[focus] THEN "ok" ELSE "error"
       /\ state' = READY
       /\ focus' = NIL
       /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, toSplit, args, op>>

FindLeafToAdd ==
    LET key == args[1]
        leaf == FindLeafNode(root, key)
    IN /\ state = FIND_LEAF_TO_ADD
       /\ focus' = leaf
       /\ toSplit' = IF AtMaxOccupancy(leaf) THEN <<leaf>> ELSE <<>>
       /\ state' = IF AtMaxOccupancy(leaf) THEN WHICH_TO_SPLIT ELSE ADD_TO_LEAF
       /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, valOf, args, op, ret>>


ParentOf(n) == CHOOSE p \in Nodes: \/ \E k \in Keys: n = childOf[p, k]
                                   \/ lastOf[p]=n

WhichToSplit ==
    LET  node == Head(toSplit)
         parent == ParentOf(node)
         splitParent == AtMaxOccupancy(parent)
         noMoreSplits == ~splitParent  \* if the parent doesn't need splitting, we don't need to consider more nodes for splitting
    IN /\ state = WHICH_TO_SPLIT
       /\ toSplit' =
           CASE node = root   -> toSplit
             [] splitParent   -> <<parent>> \o toSplit
             [] OTHER         -> toSplit
       /\ state' =
            CASE node # root /\ noMoreSplits /\ isLeaf[node]  -> SPLIT_LEAF
              [] node # root /\ noMoreSplits /\ ~isLeaf[node] -> SPLIT_INNER
              [] node = root /\ isLeaf[node]                  -> SPLIT_ROOT_LEAF
              [] node = root /\ ~isLeaf[node]                 -> SPLIT_ROOT_INNER
              [] OTHER                                        -> WHICH_TO_SPLIT
       /\ UNCHANGED <<root, isLeaf, keysOf, childOf, lastOf, valOf, op, args, ret, focus>>

\* Adding the <<key, val>> pair in args to the node indicated by focus
\* If the key is already present, this is an error
AddToLeaf ==
    LET key == args[1]
        val == args[2] IN
       /\ state = ADD_TO_LEAF
       /\ ret' = IF key \notin keysOf[focus] THEN "ok" ELSE "error"
       /\ keysOf' = IF key \notin keysOf[focus] THEN [keysOf EXCEPT ![focus]=@ \union {key}] ELSE keysOf
       /\ valOf' = IF key \notin keysOf[focus] THEN [valOf EXCEPT ![focus,key]=val] ELSE valOf
       /\ state' = READY
       /\ UNCHANGED <<root, isLeaf, childOf, lastOf, op, args, focus, toSplit>>

\* Return the pivot (midpoint) of a set of keys. If there are an even number of keys, bias towards the smaller one
PivotOf(keys) == CHOOSE k \in keys :
    LET smaller == {x \in keys : x < k}
        larger == {x \in keys: x > k} IN
     \/ Cardinality(smaller) = Cardinality(larger)
     \/ Cardinality(smaller) = Cardinality(larger)+1

SplitRootLeaf ==
    LET n1 == Head(toSplit)
        n2 == ChooseFreeNode
        newRoot == CHOOSE n \in Nodes : IsFree(n) /\ (n # n2)
        keys == keysOf[n1]
        pivot == PivotOf(keys)
        n1Keys == {x \in keys: x<pivot}
        n2Keys == {x \in keys: x>=pivot} 
        keyToInsert == args[1] IN
    /\ state = SPLIT_ROOT_LEAF
    /\ root' = newRoot
    /\ isLeaf' = [isLeaf EXCEPT ![newRoot]=FALSE, ![n2]=TRUE]
    /\ keysOf' = [keysOf EXCEPT ![newRoot]={pivot}, ![n1]=n1Keys, ![n2]=n2Keys]
    /\ childOf' = [childOf EXCEPT ![newRoot, pivot]=n1]
    /\ lastOf' = [lastOf EXCEPT ![newRoot]=n2]
    /\ valOf' = [n \in Nodes, k \in Keys |->
        CASE n=n1 /\ k \in n2Keys -> NIL
          [] n=n2 /\ k \in n2Keys -> valOf[n1, k]
          [] OTHER                -> valOf[n, k]]
    \* No more splits necessary, add the focus to the leaf
    \* Note that the focus may have changed due to the split
    /\ state' = ADD_TO_LEAF
    /\ focus' = IF keyToInsert < pivot THEN n1 ELSE n2
    /\ UNCHANGED <<op, args, ret, toSplit>>

ParentKeyOf(node) ==
    LET p == ParentOf(node) IN
    CHOOSE k \in keysOf[p]: childOf[p, k] = node

IsLastOfParent(node) == lastOf[ParentOf(node)] = node

SplitRootInner ==
    LET n1 == Head(toSplit)
        n2 == ChooseFreeNode
        newRoot == CHOOSE n \in Nodes : IsFree(n) /\ (n # n2)
        keys == keysOf[n1]
        pivot == PivotOf(keys)
        (* when splitting an inner node, pivot does not appear in either node, only in parent *)
        n1Keys == {x \in keys: x<pivot}
        n2Keys == {x \in keys: x>pivot} IN
    /\ state = SPLIT_ROOT_INNER
    /\ root' = newRoot
    /\ isLeaf' = [isLeaf EXCEPT ![newRoot]=FALSE, ![n2]=FALSE]
    /\ keysOf' = [keysOf EXCEPT ![newRoot]={pivot}, ![n1]=n1Keys, ![n2]=n2Keys]
    /\ childOf' = [n \in Nodes, k \in Keys |->
        CASE n=newRoot /\ k=pivot -> n1
          [] n=n1 /\ k \in n2Keys -> NIL
          [] n=n1 /\ k \in n1Keys -> childOf[n1, k]
          [] n=n2 /\ k \in n2Keys -> childOf[n1, k]
          [] OTHER                -> childOf[n, k]]
    /\ lastOf' = [lastOf EXCEPT ![newRoot]=n2, ![n1]=childOf[n1, pivot], ![n2]=lastOf[n1]]
    /\ toSplit' = <<>>
    /\ state' = ADD_TO_LEAF
    /\ UNCHANGED <<op, args, ret, focus, valOf>>

SplitLeaf ==
    LET n1 == Head(toSplit)
        n2 == ChooseFreeNode
        keys == keysOf[n1]
        pivot == PivotOf(keys)
        parent == ParentOf(n1)
        n1Keys == {x \in keys: x<pivot}
        n2Keys == {x \in keys: x>=pivot}
        keyToInsert == args[1]
    IN
    /\ state = SPLIT_LEAF
    /\ isLeaf' = [isLeaf EXCEPT ![n2]=TRUE]
    /\ keysOf' = [keysOf EXCEPT ![parent]=@ \union {pivot}, ![n1]=n1Keys, ![n2]=n2Keys]
    \* In the parent, point the pivot key to n1, and point the parent key to n2.
    \* TODO: handle the edge case where n1 was the last element
    /\ childOf' = IF IsLastOfParent(n1)
                  THEN [childOf EXCEPT ![parent, pivot]=n1]
                  ELSE [childOf EXCEPT ![parent, pivot]=n1, ![parent, ParentKeyOf(n1)]=n2]
    /\ lastOf' = IF IsLastOfParent(n1) THEN [lastOf EXCEPT ![parent]=n2] ELSE lastOf
    /\ valOf' = [n \in Nodes, k \in Keys |->
        CASE n=n1 /\ k \in n2Keys -> NIL
          [] n=n2 /\ k \in n2Keys -> valOf[n1, k]
          [] OTHER                -> valOf[n, k]]
    /\ state' = ADD_TO_LEAF
    /\ focus' = IF keyToInsert < pivot THEN n1 ELSE n2
    /\ UNCHANGED <<root, toSplit, op, args, ret>>


Next == \/ \E key \in Keys, val \in Vals : 
            \/ InsertReq(key, val)
            \/ UpdateReq(key, val)
        \/ \E key \in Keys: GetReq(key)
        \/ GetValue
        \/ FindLeafToAdd
        \/ WhichToSplit
        \/ AddToLeaf
        \/ SplitLeaf
        \/ SplitRootLeaf
        \/ SplitRootInner
        \/ UpdateLeaf

vars == <<root, isLeaf, keysOf, childOf, lastOf, valOf, focus, toSplit, op, args, ret, state>>

Spec == Init /\ [][Next]_vars /\ WF_op(\E key \in Keys: GetReq(key))

\*
\* Refinement mapping
\*

Leaves == {n \in Nodes : isLeaf[n]}

\*
\* Invariants
\*
Inners == {n \in Nodes: ~isLeaf[n]}

InnersMustHaveLast == \A n \in Inners : lastOf[n] # NIL
KeyOrderPreserved == \A n \in Inners : (\A k \in keysOf[n] : (\A kc \in keysOf[childOf[n, k]]: kc < k))
LeavesCantHaveLast == \A n \in Leaves : lastOf[n] = NIL
KeysInLeavesAreUnique ==
    \A n1, n2 \in Leaves : ((keysOf[n1] \intersect keysOf[n2]) # {}) => n1=n2
FreeNodesRemain == \E n \in Nodes : IsFree(n)

===="#;

const WP28_CFG: &str = r#"
INIT Init
NEXT Next

CONSTANTS
    READY = ready
    GET_VALUE = get_value
    FIND_LEAF_TO_ADD = find_leaf_to_add
    WHICH_TO_SPLIT = which_to_split
    ADD_TO_LEAF = add_to_leaf
    SPLIT_ROOT_LEAF = split_root_leaf
    SPLIT_ROOT_INNER = split_root_inner
    SPLIT_INNER = split_inner
    SPLIT_LEAF = split_leaf
    UPDATE_LEAF = update_leaf

    NIL = nil
    MISSING = missing

    Vals = {x,y}

    MaxOccupancy = 2
    MaxNode = 5
    MaxKey = 3

INVARIANT
TypeOk
InnersMustHaveLast
LeavesCantHaveLast
KeyOrderPreserved
KeysInLeavesAreUnique
FreeNodesRemain
"#;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

/// Interpreter ground truth: every gate that could route work to native code
/// removed.
fn interpreter_state_count() -> usize {
    let _guards = vec![
        common::EnvVarGuard::remove("TY_TRUST_CG"),
        common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_COMPOUND_READ"),
        common::EnvVarGuard::remove("TY_HYBRID_ENGINE_GAP"),
        common::EnvVarGuard::remove("TY_JIT"),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
    ];
    clear_for_test_reset();

    let module = common::parse_module(WP28_TLA);
    let config = Config::parse(WP28_CFG).expect("valid cfg");
    states_found("interpreter baseline", &check_module(&module, &config))
}

/// Calibrated on this exact model and gate stack:
///
/// * with the WP-28 shape merge in place, `native_dispatched = 12_928`;
/// * with it reverted, the fail-closed backstop declines the recursive-operator
///   action and the count drops to `11_324`.
///
/// The floor sits between the two, so "declined instead of compiled" fails the
/// test just as loudly as "compiled wrong".
const NATIVE_DISPATCH_FLOOR: u64 = 12_000;

struct NativeOutcome {
    states: usize,
    routed: u64,
    mismatch: u64,
    native_dispatched: u64,
    native_errors: u64,
}

/// One run with the full hybrid gate stack. `por_operator_bodies` toggles
/// `TY_POR_RESOLVE_OPERATOR_BODIES`, the gate that admits the
/// recursive-operator action (`GetValue`) to native dispatch at all.
fn native_run(por_operator_bodies: bool) -> NativeOutcome {
    let mut guards = vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_COMPOUND_READ", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_ENGINE_GAP", Some("1")),
        common::EnvVarGuard::set("TY_TAGGED_SCALAR_UNION", Some("1")),
        common::EnvVarGuard::set("TY_SCALAR_TUPLE_UNION", Some("1")),
        common::EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_SAMPLE"),
        common::EnvVarGuard::remove("TY_HYBRID_BURN_IN"),
        common::EnvVarGuard::remove("TY_JIT"),
    ];
    guards.push(if por_operator_bodies {
        common::EnvVarGuard::set("TY_POR_RESOLVE_OPERATOR_BODIES", Some("1"))
    } else {
        common::EnvVarGuard::remove("TY_POR_RESOLVE_OPERATOR_BODIES")
    });
    let _guards = guards;
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(WP28_TLA);
    let config = Config::parse(WP28_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let states = states_found("native run", &result);
    let (routed, mismatch, _projected, native_dispatched, _matched, _declined, native_errors) =
        checker.hybrid_dispatch_stats_for_testing();
    NativeOutcome {
        states,
        routed,
        mismatch,
        native_dispatched,
        native_errors,
    }
}

/// THE pin, in two halves.
///
/// 1. **Soundness.** The shadow differential must find ZERO divergences and the
///    reachable-state set must be exactly the interpreter's. A dropped union
///    shape consumed as a raw member value shows up here as
///    `mismatch_fallback > 0`.
/// 2. **The shape is really exercised.** The WP-28 fail-closed backstop turns
///    an UNPROVEN call-result convention into a decline, so soundness alone can
///    also be satisfied by simply not compiling the action. The dispatch-count
///    floor below is what distinguishes "compiled and correct" from "declined":
///    it is met only when the recursive-operator action itself compiles.
///
/// The native run goes FIRST on purpose: several gates latch into a `OnceLock`
/// on first read, so `TY_POR_RESOLVE_OPERATOR_BODIES=1` — the gate that admits
/// the recursive-operator action — has to be in the environment before any
/// check runs in this process.
#[cfg_attr(test, ntest::timeout(300000))]
#[test]
fn recursive_operator_result_as_function_key_matches_the_interpreter() {
    let out = native_run(true);
    let baseline = interpreter_state_count();

    eprintln!(
        "[wp28] states={} routed={} native_dispatched={} mismatch={} native_errors={}",
        out.states, out.routed, out.native_dispatched, out.mismatch, out.native_errors
    );

    assert_eq!(
        out.mismatch, 0,
        "the native/interpreter differential must find ZERO divergences; a \
         non-zero count is the WP-28 miscompile — a tagged-scalar-union INDEX \
         consumed as a raw member value, which made btree's FindLeafNode \
         re-enter on node n-1"
    );
    assert_eq!(out.native_errors, 0, "no runtime-error fallbacks expected");
    assert_eq!(
        out.states, baseline,
        "the reachable-state set must be exactly the interpreter's"
    );
    assert!(
        out.routed > 0,
        "the hybrid path must actually run (routed={})",
        out.routed
    );
    assert!(
        out.native_dispatched >= NATIVE_DISPATCH_FLOOR,
        "native_dispatched={} is below the floor {NATIVE_DISPATCH_FLOOR}: the \
         recursive-operator action declined instead of compiling, so this test \
         would pin nothing about the union-index consumption it exists for",
        out.native_dispatched,
    );
}
