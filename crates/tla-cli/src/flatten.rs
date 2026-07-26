// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! EXTENDS-closure FLATTENING (roadmap R6-lite, `docs/north-star-roadmap.md`): turn a
//! module that `EXTENDS` sibling user modules into ONE self-contained module source, so
//! the certificate lanes — whose certs embed a single `spec_src` and whose re-checkers
//! re-derive everything from it alone — can run on corpus-shaped specs (`MC.tla`
//! wrappers extending the real spec).
//!
//! TLA+ `EXTENDS` is namespace inclusion: the extender sees every top-level definition
//! of the extended module, un-renamed. Concatenating the extended modules' bodies
//! (dependencies first) above the extender's body is therefore semantics-preserving
//! PROVIDED nothing relies on module boundaries. Everything that does rely on them is
//! REJECTED fail-closed:
//!
//!   * `INSTANCE` anywhere in the closure (renaming/substitution semantics — R6 full);
//!   * `LOCAL` definitions (invisible to the extender; inlining would expose them);
//!   * duplicate top-level operator names across modules (shadowing);
//!   * a non-standard extended module with no sibling `.tla` file.
//!
//! Standard-library modules (`Naturals`, `TLC`, …) are kept on the flattened module's
//! own `EXTENDS` line — the evaluator provides them natively.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use tla_core::ast::{Module, Unit};

/// Is `name` a module the evaluator provides natively (kept on the flattened `EXTENDS`
/// line rather than inlined)? This is EXACTLY the evaluator's own stdlib set
/// (`tla_core::is_stdlib_module` — Naturals/Integers/…, the TLAPS proof modules,
/// and the bundled CommunityModules like FiniteSetsExt/SequencesExt/Functions), so the
/// flattened module resolves those names identically to `ty check`.
///
/// A sibling `.tla` file takes PRECEDENCE over the stub — mirroring the model checker's
/// resolution (`is_stdlib_module(name) && !loaded_names.contains(name)` in
/// `resolve/multi_module.rs`): if the corpus ships a user module of that name, it is the
/// real definition and must be inlined, not shadowed by the native stub.
fn keep_as_extends(dir: &Path, name: &str) -> bool {
    !dir.join(format!("{name}.tla")).exists() && tla_core::is_stdlib_module(name)
}

/// The result of flattening: the self-contained source and what was inlined.
pub(crate) struct Flattened {
    /// A single-module source, parse-compatible with every single-source lane.
    pub source: String,
    /// User modules inlined (dependency order) — empty means the input was already
    /// self-contained and `source` is the original text unchanged.
    pub inlined: Vec<String>,
}

/// Flatten the `EXTENDS` closure of `file` into one self-contained module source.
/// Returns the original source untouched when there is nothing to inline. Errors are
/// fail-closed decline reasons (INSTANCE / LOCAL / clash / unresolvable module).
pub(crate) fn flatten_extends_closure(file: &Path) -> Result<Flattened> {
    let dir = file.parent().unwrap_or_else(|| Path::new("."));
    let main_src =
        std::fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;
    let (main_name, main_extends) = module_header(&main_src)
        .with_context(|| format!("{}: not a parseable TLA+ module", file.display()))?;

    // Module-binding parity with `ty check` (SOUNDNESS — closes a confirmed false safe).
    // `ty check` binds the module matching the FILE STEM (`lower_main_module` with the stem
    // hint); the certificate lanes bind the FIRST module of the flattened source
    // (`tla_core::lower`). In a single-module or EXTENDS spec the file holds one top-level
    // module, so these agree and the primary path below runs unchanged. But a file with SEVERAL
    // sibling top-level modules whose stem-matching (main) module is NOT first — a safe
    // dependency module before the real main, say — would let certify bind a DIFFERENT module
    // than check verifies (the confirmed false safe). Rather than DECLINE such a file, FLATTEN the
    // stem/main module's own EXTENDS closure into a self-contained source whose FIRST (only)
    // module IS the stem/main module: `tla_core::lower` then binds exactly what `ty check` binds,
    // at both mint and every re-check (which re-lowers this same embedded `spec_src`). Mint-parity
    // with check is the soundness property; `flatten_reordered_target` enforces it with a
    // defensive re-lower assertion and fails closed on INSTANCE/LOCAL/name-clash, never guessing.
    //
    // `intra` maps every top-level module of this file to its own `----MODULE…====` block, so both
    // the reorder path AND the primary path can inline an intra-file EXTENDS dependency (a sibling
    // module living in the SAME file), matching `ty check`'s intra-file EXTENDS resolution.
    let intra: std::collections::BTreeMap<String, String> = {
        let tree = tla_core::parse_to_syntax_tree(&main_src);
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty());
        let check_binds = tla_core::lower_main_module(tla_core::FileId(0), &tree, stem).module;
        let certify_binds = tla_core::lower(tla_core::FileId(0), &tree).module;
        let blocks: std::collections::BTreeMap<String, String> =
            top_level_module_blocks(&tree).into_iter().collect();
        // Reorder/inline ONLY on a genuine multi-module disagreement. When check and certify
        // already bind the same module (single module, or the main module IS first) the primary
        // path runs unchanged — no behavior/digest change for existing specs.
        if let (Some(cm), Some(fm)) = (check_binds, certify_binds) {
            if cm.name.node != fm.name.node {
                return flatten_reordered_target(dir, &blocks, &cm.name.node);
            }
        }
        blocks
    };

    // Full-module identity-`INSTANCE` wrapper (e.g. Apalache's AP* annotation modules over a
    // base spec): resolve it to the instanced module's self-contained source. This is the same
    // identity substitution the model checker already applies for an unnamed `INSTANCE M`
    // (importing M's operators, evaluated against the enclosing same-named parameters); it is
    // FRONT-END module resolution only and fails closed on any non-identity form.
    if let Some(flat) = resolve_identity_instance_wrapper(dir, &main_src)? {
        return Ok(flat);
    }

    // Fast path: nothing user-defined to inline.
    if main_extends.iter().all(|m| keep_as_extends(dir, m)) {
        return Ok(Flattened {
            source: main_src,
            inlined: Vec::new(),
        });
    }

    // Depth-first inline of the user-module closure, dependencies before dependents.
    let mut std_extends: BTreeSet<String> = BTreeSet::new();
    let mut inlined: Vec<String> = Vec::new();
    let mut bodies: Vec<String> = Vec::new();
    let mut in_progress: Vec<String> = vec![main_name.clone()];

    for dep in &main_extends {
        if intra.contains_key(dep) {
            // An intra-file sibling module takes precedence — inline it from this file.
            visit_extends(
                dir,
                &intra,
                dep,
                &mut std_extends,
                &mut inlined,
                &mut bodies,
                &mut in_progress,
            )?;
        } else if keep_as_extends(dir, dep) {
            std_extends.insert(dep.clone());
        } else {
            visit_extends(
                dir,
                &intra,
                dep,
                &mut std_extends,
                &mut inlined,
                &mut bodies,
                &mut in_progress,
            )?;
        }
    }
    bodies.push(module_body(&main_src, &main_name)?);

    // Fail-closed semantic guards over the WHOLE flattened body.
    guard_flattened_bodies(&bodies)?;

    let ext_line = if std_extends.is_empty() {
        String::new()
    } else {
        format!(
            "EXTENDS {}\n",
            std_extends.iter().cloned().collect::<Vec<_>>().join(", ")
        )
    };
    let source = format!(
        "---- MODULE {main_name} ----\n{ext_line}{}\n====\n",
        bodies.join("\n")
    );
    Ok(Flattened { source, inlined })
}

/// Extract each TOP-LEVEL `MODULE` block from an already-parsed tree as `(name, source_text)`.
///
/// The source text is the module node's own span — the `----MODULE Name----` header through its
/// matching `====` terminator, inclusive, with any nested submodules kept verbatim inside it. Used
/// to build the intra-file module map so a stem/main module that is NOT the first top-level module
/// (or that EXTENDS an intra-file sibling) can be flattened to a self-contained source.
fn top_level_module_blocks(tree: &tla_core::SyntaxNode) -> Vec<(String, String)> {
    use tla_core::SyntaxKind;
    let mut out = Vec::new();
    for node in tree.children().filter(|n| n.kind() == SyntaxKind::Module) {
        // Name = the first Ident token after the MODULE keyword (mirrors the lowering pass).
        let mut saw_kw = false;
        let mut name = None;
        for el in node.children_with_tokens() {
            if let Some(tok) = el.as_token() {
                if tok.kind() == SyntaxKind::ModuleKw {
                    saw_kw = true;
                } else if saw_kw && tok.kind() == SyntaxKind::Ident {
                    name = Some(tok.text().to_string());
                    break;
                }
            }
        }
        if let Some(name) = name {
            out.push((name, node.text().to_string()));
        }
    }
    out
}

/// Depth-first inline of module `name` and its EXTENDS closure, dependencies before dependents.
///
/// Resolves each module source by PRECEDENCE: an intra-file sibling (`intra`, a module in the same
/// file) first, else a sibling `.tla` file on disk — mirroring `keep_as_extends`, where a real
/// definition takes precedence over the native stub. `bodies`/`inlined` accumulate in dependency
/// order; `std_extends` collects the stdlib names kept on the flattened `EXTENDS` line.
#[allow(clippy::too_many_arguments)]
fn visit_extends(
    dir: &Path,
    intra: &std::collections::BTreeMap<String, String>,
    name: &str,
    std_extends: &mut BTreeSet<String>,
    inlined: &mut Vec<String>,
    bodies: &mut Vec<String>,
    in_progress: &mut Vec<String>,
) -> Result<()> {
    if inlined.iter().any(|m| m == name) {
        return Ok(()); // already inlined via another path
    }
    if in_progress.iter().any(|m| m == name) {
        bail!("EXTENDS cycle through module `{name}`");
    }
    let src = if let Some(block) = intra.get(name) {
        block.clone()
    } else {
        let path = dir.join(format!("{name}.tla"));
        std::fs::read_to_string(&path).with_context(|| {
            format!(
                "extended module `{name}` has no sibling file {}",
                path.display()
            )
        })?
    };
    let (parsed_name, extends) =
        module_header(&src).with_context(|| format!("module `{name}`: unparseable header"))?;
    if parsed_name != name {
        bail!("module `{name}`: header declares MODULE `{parsed_name}`, expected `{name}`");
    }
    in_progress.push(name.to_string());
    for dep in &extends {
        if intra.contains_key(dep) {
            visit_extends(dir, intra, dep, std_extends, inlined, bodies, in_progress)?;
        } else if keep_as_extends(dir, dep) {
            std_extends.insert(dep.clone());
        } else {
            visit_extends(dir, intra, dep, std_extends, inlined, bodies, in_progress)?;
        }
    }
    in_progress.pop();
    bodies.push(module_body(&src, name)?);
    inlined.push(name.to_string());
    Ok(())
}

/// Fail-closed semantic guards over a flattened body set: reject module-boundary semantics
/// (`INSTANCE`/`LOCAL`) and duplicate top-level definition names across modules (shadowing). A
/// clash is a decline, never a silent guess.
fn guard_flattened_bodies(bodies: &[String]) -> Result<()> {
    let joined = bodies.join("\n");
    for line in joined.lines() {
        let t = line.trim_start();
        if t.starts_with("INSTANCE") || t.contains(" INSTANCE ") || t.starts_with("LOCAL ") {
            bail!(
                "module boundary semantics (INSTANCE/LOCAL) in the EXTENDS closure — \
                 flattening would change meaning (fail-closed; roadmap R6)"
            );
        }
    }
    let mut seen = BTreeSet::new();
    for line in joined.lines() {
        if let Some(head) = def_head(line) {
            if !seen.insert(head.to_string()) {
                bail!("definition `{head}` appears in more than one module of the EXTENDS closure");
            }
        }
    }
    Ok(())
}

/// Flatten a NON-first stem/main module (the one `ty check` binds via `lower_main_module`) and its
/// intra-file/cross-file EXTENDS closure into a self-contained single-module source whose FIRST
/// (only) module IS that stem module — so `certify`'s first-module `tla_core::lower` binds exactly
/// what `ty check` verifies, at mint and every re-check.
///
/// SOUNDNESS: the emitted source is re-lowered and asserted to bind `target_name`; a mismatch (or
/// an INSTANCE/LOCAL/name-clash anywhere in the closure) is a fail-closed decline — never certify a
/// different module than check verifies, in EITHER direction.
fn flatten_reordered_target(
    dir: &Path,
    intra: &std::collections::BTreeMap<String, String>,
    target_name: &str,
) -> Result<Flattened> {
    let target_block = intra.get(target_name).ok_or_else(|| {
        anyhow::anyhow!(
            "NOT CERTIFIED: internal — stem module `{target_name}` is not among the file's \
             top-level modules"
        )
    })?;
    let (parsed, target_extends) = module_header(target_block)
        .with_context(|| format!("stem module `{target_name}`: unparseable header"))?;
    if parsed != target_name {
        bail!("stem module `{target_name}`: header declares MODULE `{parsed}`");
    }

    let mut std_extends: BTreeSet<String> = BTreeSet::new();
    let mut inlined: Vec<String> = Vec::new();
    let mut bodies: Vec<String> = Vec::new();
    let mut in_progress: Vec<String> = vec![target_name.to_string()];
    for dep in &target_extends {
        if intra.contains_key(dep) {
            visit_extends(
                dir,
                intra,
                dep,
                &mut std_extends,
                &mut inlined,
                &mut bodies,
                &mut in_progress,
            )?;
        } else if keep_as_extends(dir, dep) {
            std_extends.insert(dep.clone());
        } else {
            visit_extends(
                dir,
                intra,
                dep,
                &mut std_extends,
                &mut inlined,
                &mut bodies,
                &mut in_progress,
            )?;
        }
    }
    bodies.push(module_body(target_block, target_name)?);

    guard_flattened_bodies(&bodies)?;

    let ext_line = if std_extends.is_empty() {
        String::new()
    } else {
        format!(
            "EXTENDS {}\n",
            std_extends.iter().cloned().collect::<Vec<_>>().join(", ")
        )
    };
    let source = format!(
        "---- MODULE {target_name} ----\n{ext_line}{}\n====\n",
        bodies.join("\n")
    );

    // SOUNDNESS assertion (mint-parity with `ty check`): the FIRST module of the emitted source —
    // the module every certificate lane binds via `tla_core::lower`, at mint and every re-check —
    // MUST be the stem/main module `ty check` verifies. If the reorder did not achieve that, refuse.
    let tree = tla_core::parse_to_syntax_tree(&source);
    match tla_core::lower(tla_core::FileId(0), &tree).module {
        Some(m) if m.name.node == target_name => {}
        other => bail!(
            "NOT CERTIFIED: reordered flatten binds module `{}`, not the stem module `{target_name}` \
             that `ty check` verifies — refusing to certify a different module (fail-closed)",
            other.map(|m| m.name.node).unwrap_or_else(|| "<none>".to_string())
        ),
    }

    Ok(Flattened { source, inlined })
}

/// Collect the top-level CONSTANT + VARIABLE parameter names declared by a module.
fn declared_param_names(module: &Module) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for unit in &module.units {
        match &unit.node {
            Unit::Constant(cs) => {
                for c in cs {
                    names.insert(c.name.node.clone());
                }
            }
            Unit::Variable(vs) => {
                for v in vs {
                    names.insert(v.node.clone());
                }
            }
            _ => {}
        }
    }
    names
}

/// Resolve a PURE full-module identity-`INSTANCE` wrapper to the instanced module's
/// self-contained source.
///
/// The supported shape (e.g. Apalache's `APEWD840` over `EWD840`) is a module whose ONLY
/// content is CONSTANT/VARIABLE declarations (plus optional ASSUMEs) and a single standalone,
/// non-`LOCAL`, `WITH`-free `INSTANCE M`, where the wrapper declares EXACTLY M's CONSTANT/VARIABLE
/// parameters by the same names. For that shape the unnamed `INSTANCE M` is the IDENTITY
/// substitution — it imports M's operators into the unqualified scope evaluated against the
/// same-named parameters — so M's definitions are semantically unchanged when used verbatim, and
/// certifying the wrapper is identical to certifying M under the same config. This is exactly the
/// operator set `ty check` imports for the same wrapper.
///
/// Returns:
/// - `Ok(None)` when the main module is NOT such a wrapper (no standalone INSTANCE, or the module
///   defines its own operators/theorems) — the caller proceeds with normal EXTENDS flattening or
///   the raw source, so no self-sufficient spec is disturbed.
/// - `Ok(Some(..))` when the wrapper is resolved to `M`'s self-contained source.
/// - `Err(..)` (fail-closed) when the module IS a pure declarations+INSTANCE wrapper but the
///   instance is not the supported identity form (a `WITH` clause, a `LOCAL`/multiple instance, a
///   non-standard EXTENDS alongside it, an unresolvable/mismatched-parameter instanced module).
fn resolve_identity_instance_wrapper(dir: &Path, main_src: &str) -> Result<Option<Flattened>> {
    let tree = tla_core::parse_to_syntax_tree(main_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let Some(module) = lowered.module else {
        return Ok(None); // unparseable here — leave it to the existing (textual) path
    };

    // Standalone (unnamed) INSTANCE units. A NAMED instance `X == INSTANCE M` lowers to an
    // Operator with an InstanceExpr body, not a `Unit::Instance`, so it is not collected here.
    let instances: Vec<&tla_core::ast::InstanceDecl> = module
        .units
        .iter()
        .filter_map(|u| match &u.node {
            Unit::Instance(inst) => Some(inst),
            _ => None,
        })
        .collect();
    if instances.is_empty() {
        return Ok(None); // not an instance wrapper — normal flattening handles it
    }

    // The wrapper may define its OWN nullary value-operators (e.g. Apalache's `RMVal == {...}`,
    // referenced by a cfg `CONSTANT RM <- RMVal` operator-substitution). Those are CARRIED verbatim
    // into the resolved source — appended AFTER M's body, where they see M's constants/variables/
    // operators — so the substitution can resolve against them. This is sound because the wrapper's
    // scope (M's imported operators + the same-named parameters) equals M's own scope under the
    // identity instance, so a wrapper value-operator means the same thing appended to M's body.
    //
    // A THEOREM, a RECURSIVE forward-declaration, a LOCAL definition, or a PARAMETERIZED operator is
    // OUT of the supported value-operator shape: the module might be self-sufficient (its own
    // Init/Next) or rely on module boundaries, so leave it untouched (`Ok(None)`) — resolving it to
    // the instanced module could drop or misplace those definitions.
    // Each wrapper own value-operator as (name, source span, canonical body text). The canonical
    // body is a span-insensitive pretty-print of the body AST — the gate below for dropping a
    // benign Class-A identity restatement (comments / whitespace / NameId / spans normalize away).
    let mut own_ops: Vec<(String, tla_core::Span, String)> = Vec::new();
    for unit in &module.units {
        match &unit.node {
            Unit::Theorem(_) | Unit::Recursive(_) => return Ok(None),
            Unit::Operator(op) => {
                if op.local || !op.params.is_empty() {
                    return Ok(None);
                }
                own_ops.push((
                    op.name.node.clone(),
                    op.name.span.merge(op.body.span),
                    tla_core::pretty_expr(&op.body.node),
                ));
            }
            _ => {}
        }
    }

    // From here the module is declarations + optional own value-operators + standalone INSTANCE(s)
    // with no control logic of its own, so a non-identity instance is a fail-closed decline (there
    // is no other certify path for it).
    if instances.len() != 1 {
        bail!(
            "the module has {} standalone INSTANCEs — only a single full-module identity-instance \
             wrapper is resolved here (fail-closed)",
            instances.len()
        );
    }
    let inst = instances[0];
    if inst.local {
        bail!("standalone `LOCAL INSTANCE` is not the supported identity-instance wrapper (fail-closed)");
    }
    if !inst.substitutions.is_empty() {
        bail!(
            "parameterized `INSTANCE {} WITH …` is out of scope — the certify front-end resolves \
             only the identity substitution, not renaming/`WITH` (fail-closed)",
            inst.module.node
        );
    }
    for ext in &module.extends {
        if !keep_as_extends(dir, ext.node.as_str()) {
            bail!(
                "the identity-instance wrapper also EXTENDS user module `{}` — combined \
                 EXTENDS+INSTANCE is out of scope (fail-closed)",
                ext.node
            );
        }
    }

    // Resolve the instanced module to a self-contained source (reusing EXTENDS flattening for it).
    let m_name = inst.module.node.clone();
    let m_path = dir.join(format!("{m_name}.tla"));
    let m_flat = flatten_extends_closure(&m_path)
        .with_context(|| format!("resolving full-module `INSTANCE {m_name}`"))?;

    // Identity guard: the wrapper must declare EXACTLY M's CONSTANT/VARIABLE parameters, by the
    // same names. An unnamed `INSTANCE M` maps each of M's parameters to the same-named symbol in
    // scope; set-equality makes that substitution the identity and total, so M's definitions are
    // unchanged when used verbatim.
    let m_tree = tla_core::parse_to_syntax_tree(&m_flat.source);
    let m_lowered = tla_core::lower(tla_core::FileId(0), &m_tree);
    let Some(m_module) = m_lowered.module else {
        bail!("instanced module `{m_name}` does not lower to a module (fail-closed)");
    };
    let wrapper_params = declared_param_names(&module);
    let m_params = declared_param_names(&m_module);
    if wrapper_params != m_params {
        bail!(
            "the wrapper's declared parameters {wrapper_params:?} do not exactly match those of \
             `INSTANCE {m_name}` {m_params:?} — the substitution is not the identity (fail-closed)"
        );
    }

    // Build the self-contained source: M's body, under the wrapper's module name, keeping M's
    // (standard-only, post-flatten) EXTENDS line.
    let (parsed_m_name, m_extends) = module_header(&m_flat.source)
        .with_context(|| format!("instanced module `{m_name}`: unparseable header"))?;
    for ext in &m_extends {
        if !keep_as_extends(dir, ext.as_str()) {
            bail!("instanced module `{m_name}` still EXTENDS user module `{ext}` after flattening (fail-closed)");
        }
    }
    let m_body = module_body(&m_flat.source, &parsed_m_name)?;
    let ext_line = if m_extends.is_empty() {
        String::new()
    } else {
        format!("EXTENDS {}\n", m_extends.join(", "))
    };

    // Carry the wrapper's own value-operators (verbatim source text) AFTER M's body. Map each
    // instanced-module operator to (arity, canonical body text). A wrapper own-op that shares a
    // name with an M operator is normally an ambiguous redefinition — a fail-closed decline, never
    // a silent shadow. The SOLE exception (Class-A identity restatement, e.g. Apalache's
    // `vars == <<…>>` restated only to hang an `@type:` annotation): M's same-named operator is
    // NULLARY and its body is AST-identical to the wrapper's. This branch is reached only after the
    // :330 identity guard passed, so the wrapper and M share the same scope (M's imported operators
    // + the same-named parameters); an AST-equal nullary body therefore provably denotes the SAME
    // value, so DROPPING the wrapper's duplicate keeps M's single identical definition and is a
    // no-op — behavior-preserving by construction, not a heuristic. Any other clash (different
    // body, or a parameterized M op of that name) still declines. Spans index into `main_src`.
    let m_ops: std::collections::BTreeMap<String, (usize, String)> = m_module
        .units
        .iter()
        .filter_map(|u| match &u.node {
            Unit::Operator(op) => Some((
                op.name.node.clone(),
                (op.params.len(), tla_core::pretty_expr(&op.body.node)),
            )),
            _ => None,
        })
        .collect();
    let mut own_defs = String::new();
    for (name, span, body) in &own_ops {
        match m_ops.get(name) {
            // No M operator of this name — carry the wrapper's own value-operator verbatim.
            None => {
                own_defs.push_str(&main_src[span.start as usize..span.end as usize]);
                own_defs.push('\n');
            }
            // Identity restatement of M's same-named NULLARY operator with an AST-identical body:
            // an exact duplicate in the identical scope. Drop it (keep M's) — provably a no-op.
            Some((0, m_body)) if m_body == body => {}
            // Any other name clash (differing body, or a parameterized M op) is ambiguous.
            Some(_) => {
                bail!(
                    "the identity-instance wrapper redefines `{name}` from `INSTANCE {m_name}` \
                     with a different body or arity — ambiguous redefinition (fail-closed)"
                );
            }
        }
    }

    let source = format!(
        "---- MODULE {} ----\n{ext_line}{m_body}\n{own_defs}====\n",
        module.name.node
    );

    let mut inlined = m_flat.inlined;
    inlined.push(m_name);
    Ok(Some(Flattened { source, inlined }))
}

/// Remove TLA+ comments from one line while carrying nested block-comment depth across lines.
/// The returned bytes are valid UTF-8 because non-comment bytes are copied verbatim. This is used
/// only to classify module-header lines; the body itself retains its original text.
fn header_code(line: &str, block_depth: &mut usize) -> String {
    let bytes = line.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if *block_depth > 0 {
            if bytes[i..].starts_with(b"(*") {
                *block_depth += 1;
                i += 2;
            } else if bytes[i..].starts_with(b"*)") {
                *block_depth -= 1;
                i += 2;
            } else {
                i += 1;
            }
        } else if bytes[i..].starts_with(b"\\*") {
            break;
        } else if bytes[i..].starts_with(b"(*") {
            *block_depth += 1;
            i += 2;
        } else if bytes[i] == b'"' {
            // Comment delimiters inside a TLA+ string are ordinary bytes. Strings do not span
            // lines, but honor backslash escapes so an escaped quote does not end the scan early.
            out.push(bytes[i]);
            i += 1;
            while i < bytes.len() {
                out.push(bytes[i]);
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 1;
                    out.push(bytes[i]);
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).expect("comment removal preserves UTF-8")
}

/// Parse a module's name and EXTENDS list from its header lines (textual — the header
/// grammar is line-oriented and this must work on files the full parser may reject
/// later; the flattened result still goes through the real parser). Comment-only lines
/// are legal between `MODULE` and `EXTENDS` and must not terminate the header.
fn module_header(src: &str) -> Option<(String, Vec<String>)> {
    let mut name = None;
    let mut extends = Vec::new();
    let mut block_depth = 0usize;
    for line in src.lines() {
        let code = header_code(line, &mut block_depth);
        let t = code.trim();
        if name.is_none() {
            if let Some(rest) = t
                .strip_prefix("----")
                .map(|r| r.trim_start_matches('-').trim())
            {
                if let Some(m) = rest.strip_prefix("MODULE") {
                    name = Some(m.trim_end_matches('-').trim().to_string());
                }
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix("EXTENDS") {
            extends.extend(
                rest.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            );
        } else if !t.is_empty() {
            break; // first non-EXTENDS content line ends the header
        }
    }
    name.map(|n| (n, extends))
}

/// The body of a module: everything after the header (module line + EXTENDS lines) and
/// before the terminating `====` line.
fn module_body(src: &str, name: &str) -> Result<String> {
    let mut out = Vec::new();
    let mut in_body = false;
    let mut seen_header = false;
    let mut block_depth = 0usize;
    for line in src.lines() {
        let code = header_code(line, &mut block_depth);
        let t = code.trim();
        if !seen_header {
            if t.starts_with("----") && t.contains("MODULE") {
                seen_header = true;
            }
            continue;
        }
        if !in_body {
            // Header comments are semantically inert and deliberately omitted. Keeping their raw
            // fragments while removing a later `EXTENDS` on the same block-comment line could
            // produce an unterminated comment in the flattened source.
            if t.starts_with("EXTENDS") || t.is_empty() {
                continue;
            }
            in_body = true;
        }
        if t.chars().take_while(|&c| c == '=').count() >= 4 {
            return Ok(out.join("\n"));
        }
        out.push(line.to_string());
    }
    bail!("module `{name}`: no terminating ==== line")
}

/// A top-level definition head (`Name ==` or `Name(args) ==`) at column 0.
fn def_head(line: &str) -> Option<&str> {
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let (lhs, _) = line.split_once("==")?;
    let lhs = lhs.trim_end();
    let head = lhs.split('(').next()?.trim();
    (!head.is_empty()
        && head
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '!')
        && head.chars().next().is_some_and(char::is_alphabetic))
    .then_some(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn flattens_mc_wrapper_over_base_module() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Base.tla",
            "---- MODULE Base ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\nNext == x' = x + 1 /\\ x < 3\n====\n",
        );
        write(
            dir.path(),
            "MC.tla",
            "---- MODULE MC ----\nEXTENDS Base, TLC\nn == 3\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("MC.tla")).unwrap();
        assert_eq!(f.inlined, vec!["Base".to_string()]);
        assert!(f.source.contains("MODULE MC"));
        assert!(f.source.contains("EXTENDS Naturals, TLC"));
        assert!(f.source.contains("Init == x = 0"));
        assert!(f.source.contains("n == 3"));
        // The flattened source must be a parseable single module.
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        let lowered = tla_core::lower(tla_core::FileId(0), &tree);
        assert!(lowered.module.is_some(), "flattened source must lower");
    }

    #[test]
    fn self_contained_passes_through_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let src = "---- MODULE Solo ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 0\n====\n";
        write(dir.path(), "Solo.tla", src);
        let f = flatten_extends_closure(&dir.path().join("Solo.tla")).unwrap();
        assert!(f.inlined.is_empty());
        assert_eq!(f.source, src);
    }

    #[test]
    fn instance_local_and_clashes_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Inst.tla",
            "---- MODULE Inst ----\nEXTENDS Naturals\nFoo == INSTANCE Other\n====\n",
        );
        write(
            dir.path(),
            "MCi.tla",
            "---- MODULE MCi ----\nEXTENDS Inst\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("MCi.tla")).is_err(),
            "INSTANCE declines"
        );

        write(
            dir.path(),
            "Loc.tla",
            "---- MODULE Loc ----\nLOCAL Hidden == 1\nInit == Hidden = 1\n====\n",
        );
        write(
            dir.path(),
            "MCl.tla",
            "---- MODULE MCl ----\nEXTENDS Loc\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("MCl.tla")).is_err(),
            "LOCAL declines"
        );

        write(dir.path(), "A.tla", "---- MODULE A ----\nDup == 1\n====\n");
        write(
            dir.path(),
            "MCc.tla",
            "---- MODULE MCc ----\nEXTENDS A\nDup == 2\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("MCc.tla")).is_err(),
            "name clash declines"
        );

        write(
            dir.path(),
            "MCm.tla",
            "---- MODULE MCm ----\nEXTENDS NoSuchSibling\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("MCm.tla")).is_err(),
            "missing sibling declines"
        );
    }

    /// A pure declarations + bare `INSTANCE M` wrapper (identity substitution, the AP* shape) is
    /// resolved to M's self-contained source, and the result lowers with M's definitions present.
    #[test]
    fn identity_instance_wrapper_resolves_to_base() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Base.tla",
            "---- MODULE Base ----\nEXTENDS Naturals\nCONSTANT N\nVARIABLE x\n\
             Init == x = 0\nNext == x < N /\\ x' = x + 1\nSpec == Init /\\ [][Next]_x\n\
             Inv == x >= 0\n====\n",
        );
        write(
            dir.path(),
            "APBase.tla",
            "---- MODULE APBase ----\nEXTENDS Naturals\nCONSTANT N\nVARIABLE x\nINSTANCE Base\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("APBase.tla")).unwrap();
        assert_eq!(f.inlined, vec!["Base".to_string()]);
        assert!(
            f.source.contains("MODULE APBase"),
            "keeps the wrapper's module name"
        );
        // The instanced module's operators are now inline in the self-contained source.
        assert!(f.source.contains("Init == x = 0"));
        assert!(f.source.contains("Spec == Init"));
        assert!(f.source.contains("Inv == x >= 0"));
        // And it must lower as one module.
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        let lowered = tla_core::lower(tla_core::FileId(0), &tree);
        assert!(
            lowered.module.is_some(),
            "resolved instance source must lower"
        );
    }

    /// An identity-`INSTANCE` wrapper that ALSO defines its OWN nullary value-operators (the shape a
    /// cfg `CONSTANT X <- XVal` substitution needs — e.g. Apalache's `RMVal == {…}`) resolves to M's
    /// body WITH those value-operators CARRIED verbatim after it, so the `<-` lookup can find `XVal`.
    #[test]
    fn identity_instance_wrapper_carries_own_value_operators() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Ctr.tla",
            "---- MODULE Ctr ----\nEXTENDS Naturals\nCONSTANT N\nVARIABLE x\n\
             Init == x = 0\nNext == x < N /\\ x' = x + 1\nSpec == Init /\\ [][Next]_x\n\
             Inv == x <= 3\n====\n",
        );
        write(
            dir.path(),
            "APCtr.tla",
            "---- MODULE APCtr ----\nEXTENDS Naturals\nCONSTANT N\nVARIABLE x\nINSTANCE Ctr\n\
             NVal == 3\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("APCtr.tla")).unwrap();
        assert_eq!(f.inlined, vec!["Ctr".to_string()]);
        // M's operators (from the INSTANCE) AND the wrapper's own value-operator are both present.
        assert!(f.source.contains("Spec == Init"), "M's Spec is inlined");
        assert!(
            f.source.contains("NVal == 3"),
            "the wrapper's own value-operator is carried"
        );
        // The carried source must lower as one standalone module.
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        let lowered = tla_core::lower(tla_core::FileId(0), &tree);
        assert!(
            lowered.module.is_some(),
            "carried-operator source must lower"
        );
    }

    /// Carrying is fail-closed against ambiguity: a wrapper value-operator that CLASHES with an
    /// operator of the instanced module is a decline (never a silent redefinition). A THEOREM, a
    /// RECURSIVE forward-decl, or a PARAMETERIZED/LOCAL own-operator is out of the value-operator
    /// shape and falls through unchanged (`Ok(None)` → possibly self-sufficient), not resolved.
    #[test]
    fn own_operator_carry_guards_are_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Base.tla",
            "---- MODULE Base ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\nDup == 1\n====\n",
        );
        // Own operator `Dup` clashes with Base's `Dup` — ambiguous redefinition ⇒ decline (Err).
        write(
            dir.path(),
            "APClash.tla",
            "---- MODULE APClash ----\nVARIABLE x\nINSTANCE Base\nDup == 2\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("APClash.tla")).is_err(),
            "clashing own operator declines"
        );
        // A PARAMETERIZED own operator is not a value-operator ⇒ fall through unchanged.
        write(
            dir.path(),
            "APParam.tla",
            "---- MODULE APParam ----\nVARIABLE x\nINSTANCE Base\nF(a) == a\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("APParam.tla")).unwrap();
        assert!(
            f.inlined.is_empty(),
            "parameterized own-operator wrapper is not instance-resolved"
        );
    }

    /// Class-A (the Apalache `vars == <<…>>` annotation shape): a wrapper that restates one of M's
    /// NULLARY operators with an AST-identical body is resolved by DROPPING the duplicate — the
    /// flattened source keeps M's single definition, not two. Whitespace/comment differences in the
    /// restatement normalize away (the gate is the pretty-printed body, not raw text). A restatement
    /// whose body DIFFERS still declines (ambiguous redefinition).
    #[test]
    fn identity_instance_wrapper_drops_ast_identical_restatement() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Base.tla",
            "---- MODULE Base ----\nVARIABLE x\nvars == << x >>\nInit == x = 0\nNext == x' = x\n====\n",
        );
        // Identical restatement of `vars` (only whitespace + a comment differ) ⇒ dropped, resolves.
        write(
            dir.path(),
            "APId.tla",
            "---- MODULE APId ----\nVARIABLE x\n\\* @type: <<Int>>;\nvars == <<x>>\nINSTANCE Base\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("APId.tla")).unwrap();
        assert_eq!(f.inlined, vec!["Base".to_string()]);
        assert!(
            f.source.contains("MODULE APId"),
            "keeps the wrapper's module name"
        );
        // The duplicate is dropped: M's single `vars ==` survives, not two (no redeclaration clash).
        assert_eq!(
            f.source.matches("vars ==").count(),
            1,
            "AST-identical restatement is dropped, not carried"
        );
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        assert!(
            tla_core::lower(tla_core::FileId(0), &tree).module.is_some(),
            "resolved source must lower as one module"
        );

        // A restatement whose body DIFFERS is still an ambiguous redefinition ⇒ decline (Err).
        write(
            dir.path(),
            "APDiff.tla",
            "---- MODULE APDiff ----\nVARIABLE x\nvars == << x, x >>\nINSTANCE Base\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("APDiff.tla")).is_err(),
            "restatement with a differing body still declines"
        );
    }

    /// A parameterized `INSTANCE M WITH …` and a non-identity parameter set both fail closed —
    /// the certify front-end resolves only the identity substitution.
    #[test]
    fn non_identity_instance_forms_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Base.tla",
            "---- MODULE Base ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\n====\n",
        );
        // WITH clause (renaming) — out of scope.
        write(
            dir.path(),
            "APWith.tla",
            "---- MODULE APWith ----\nVARIABLE y\nINSTANCE Base WITH x <- y\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("APWith.tla")).is_err(),
            "WITH-substitution instance declines"
        );
        // Non-identity parameter set (wrapper declares an extra variable) — decline.
        write(
            dir.path(),
            "APExtra.tla",
            "---- MODULE APExtra ----\nVARIABLE x, z\nINSTANCE Base\n====\n",
        );
        assert!(
            flatten_extends_closure(&dir.path().join("APExtra.tla")).is_err(),
            "non-identity parameter set declines"
        );
    }

    /// A module that defines its own operators is NOT treated as an instance wrapper even when it
    /// also has a standalone INSTANCE: it may be self-sufficient, so it falls through unchanged
    /// (no regression for such specs). A NAMED instance (`X == INSTANCE M`) is likewise not a
    /// standalone-instance wrapper.
    #[test]
    fn self_sufficient_module_with_instance_falls_through() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Base.tla",
            "---- MODULE Base ----\nVARIABLE x\nBOp == x\n====\n",
        );
        // Own operators + a NAMED instance: not a pure wrapper — returned unchanged.
        let named = "---- MODULE Named ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\n\
                     I == INSTANCE Base\n====\n";
        write(dir.path(), "Named.tla", named);
        let f = flatten_extends_closure(&dir.path().join("Named.tla")).unwrap();
        assert!(
            f.inlined.is_empty(),
            "self-sufficient module is not instance-resolved"
        );
        assert_eq!(
            f.source, named,
            "self-sufficient module passes through unchanged"
        );
    }

    /// SOUNDNESS (the module-binding fix): a multi-module file whose stem/main module is NOT first
    /// flattens to a self-contained source whose FIRST (only) module IS the stem module — so
    /// certify's first-module `lower` binds exactly what `ty check` (stem-hinted) binds. The
    /// unrelated earlier sibling is dropped (the stem module does not depend on it).
    #[test]
    fn nonfirst_stem_module_is_reordered_to_front() {
        let dir = tempfile::tempdir().unwrap();
        // `Main.tla`: `Helper` first, `Main` (= file stem) second. Independent modules.
        write(
            dir.path(),
            "Main.tla",
            "---- MODULE Helper ----\nEXTENDS Naturals\nVARIABLE y\nInitH == y = 0\n\
             NextH == y' = y\nInvH == y >= 0\n====\n\
             ---- MODULE Main ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\n\
             Next == x' = x\nInv == x >= 0\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("Main.tla")).unwrap();
        // First-module lowering (what every certificate lane binds) MUST be `Main`.
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        let bound = tla_core::lower(tla_core::FileId(0), &tree).module.unwrap();
        assert_eq!(
            bound.name.node, "Main",
            "reordered source must bind the stem module Main"
        );
        // The unrelated first module is dropped; only `Main`'s own content remains.
        assert!(f.source.contains("Inv == x >= 0"));
        assert!(
            !f.source.contains("InvH"),
            "unrelated sibling `Helper` must be dropped"
        );
        assert!(
            f.inlined.is_empty(),
            "an independent sibling is not an inlined dependency"
        );
    }

    /// SOUNDNESS: when the (non-first) stem module EXTENDS an intra-file sibling, that dependency is
    /// INLINED into the self-contained source (matching `ty check`'s intra-file EXTENDS resolution),
    /// and the emitted first module is still the stem module.
    #[test]
    fn nonfirst_stem_module_inlines_intrafile_extends_dep() {
        let dir = tempfile::tempdir().unwrap();
        // `IntraMain.tla`: `Base` first, `IntraMain` (= stem) second, which EXTENDS the sibling Base.
        write(
            dir.path(),
            "IntraMain.tla",
            "---- MODULE Base ----\nEXTENDS Naturals\nVARIABLE x\nInit == x = 0\n\
             Next == x' = x\nInv == x >= 0\n====\n\
             ---- MODULE IntraMain ----\nEXTENDS Base\nInvMain == x <= 4\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("IntraMain.tla")).unwrap();
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        let bound = tla_core::lower(tla_core::FileId(0), &tree).module.unwrap();
        assert_eq!(
            bound.name.node, "IntraMain",
            "emitted source binds the stem module"
        );
        assert_eq!(
            f.inlined,
            vec!["Base".to_string()],
            "the intra-file dep is inlined"
        );
        assert!(
            f.source.contains("Init == x = 0"),
            "Base's operators are inlined"
        );
        assert!(
            f.source.contains("InvMain == x <= 4"),
            "the stem module's own body is present"
        );
        assert!(
            f.source.contains("EXTENDS Naturals"),
            "Base's stdlib EXTENDS is kept"
        );
    }

    /// A multi-module file with no stem match binds the LAST module (mirroring `lower_main_module`'s
    /// fallback), reordered to the front of the self-contained source.
    #[test]
    fn multi_module_no_stem_match_binds_last() {
        let dir = tempfile::tempdir().unwrap();
        // File `Zzz.tla` matches neither `A` nor `B`; `lower_main_module` falls back to LAST (`B`).
        write(
            dir.path(),
            "Zzz.tla",
            "---- MODULE A ----\nVARIABLE x\nInitA == x = 0\nNextA == x' = x\n====\n\
             ---- MODULE B ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\nInv == x = 0\n====\n",
        );
        let f = flatten_extends_closure(&dir.path().join("Zzz.tla")).unwrap();
        let tree = tla_core::parse_to_syntax_tree(&f.source);
        let bound = tla_core::lower(tla_core::FileId(0), &tree).module.unwrap();
        assert_eq!(
            bound.name.node, "B",
            "no stem match binds the last module, per lower_main_module"
        );
    }
}
