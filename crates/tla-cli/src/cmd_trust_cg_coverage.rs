// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty trust_cg-coverage` -- static trust-codegen action-shape inventory.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tla_check::{resolve_spec_from_config, Config};
use tla_core::ast::{BoundPattern, BoundVar, ExceptPathElement, Expr, Module, OperatorDef, Unit};
use tla_core::{
    lower_error_diagnostic, lower_main_module, lower_single_expr, parse, parse_error_diagnostic,
    FileId, SyntaxNode,
};

use crate::cli_schema::{TrustCgCoverageArgs, TrustCgCoverageOutputFormat};
use crate::helpers::read_source;

const SCHEMA: &str = "ty.trust_cg_action_coverage.v1";

pub(crate) fn cmd_trust_cg_coverage(args: TrustCgCoverageArgs) -> Result<()> {
    let report = if let Some(file) = &args.file {
        let config = args.config.as_deref();
        CoverageReport::single(analyze_spec(None, file, config)?)
    } else {
        analyze_baseline(&args)?
    };

    if let Some(path) = &args.output_file {
        write_json(path, &report)?;
        if matches!(args.output, TrustCgCoverageOutputFormat::Human) {
            println!(
                "trust-cg action coverage JSON written to {}",
                path.display()
            );
        }
    }

    if let Some(path) = &args.report {
        write_markdown(path, &report)?;
        if matches!(args.output, TrustCgCoverageOutputFormat::Human) {
            println!(
                "trust-cg action coverage report written to {}",
                path.display()
            );
        }
    }

    match args.output {
        TrustCgCoverageOutputFormat::Human => print_human(&report),
        TrustCgCoverageOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }

    Ok(())
}

fn analyze_baseline(args: &TrustCgCoverageArgs) -> Result<CoverageReport> {
    let baseline_text = std::fs::read_to_string(&args.baseline)
        .with_context(|| format!("read baseline {}", args.baseline.display()))?;
    let baseline: Baseline =
        serde_json::from_str(&baseline_text).context("parse spec baseline JSON")?;

    let examples_dir = args
        .examples_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(&baseline.inputs.examples_dir));

    let mut specs = BTreeMap::new();
    for (name, spec) in baseline.specs {
        if !args.spec.is_empty() && !args.spec.iter().any(|wanted| wanted == &name) {
            continue;
        }

        let Some(source) = spec.source else {
            specs.insert(
                name.clone(),
                SpecCoverage::skip(name, "baseline entry has no source"),
            );
            continue;
        };
        if source.mode.as_deref().is_some_and(|mode| mode != "check") {
            specs.insert(
                name.clone(),
                SpecCoverage::skip(name, "baseline source is not check mode"),
            );
            continue;
        }

        let tla_path = examples_dir.join(&source.tla_path);
        let cfg_path = examples_dir.join(&source.cfg_path);
        let cfg = cfg_path.is_file().then_some(cfg_path.as_path());
        let analysis = analyze_spec(Some(name.clone()), &tla_path, cfg).unwrap_or_else(|err| {
            SpecCoverage::error(
                name.clone(),
                tla_path.clone(),
                cfg_path.clone(),
                err.to_string(),
            )
        });
        specs.insert(name, analysis);
    }

    Ok(CoverageReport::new(specs))
}

fn analyze_spec(name: Option<String>, file: &Path, config: Option<&Path>) -> Result<SpecCoverage> {
    let source = read_source(file)?;
    let tree = parse_or_report_for_coverage(file, &source)?;
    let hint_name = file.file_stem().and_then(|s| s.to_str());
    let lower_result = lower_main_module(FileId(0), &tree, hint_name);
    if !lower_result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &lower_result.errors {
            let diagnostic = lower_error_diagnostic(&file_path, &err.message, err.span);
            diagnostic.eprint(&file_path, &source);
        }
        bail!("lower failed with {} error(s)", lower_result.errors.len());
    }
    let module = lower_result.module.context("lower produced no module")?;
    let next_target = config_next_target(config, &tree)?;
    analyze_module(name, file, config, &module, &next_target)
}

fn parse_or_report_for_coverage(file: &Path, source: &str) -> Result<SyntaxNode> {
    let result = parse(source);
    if !result.errors.is_empty() {
        let file_path = file.display().to_string();
        for err in &result.errors {
            let diagnostic = parse_error_diagnostic(&file_path, &err.message, err.start, err.end);
            diagnostic.eprint(&file_path, source);
        }
        bail!("parse failed with {} error(s)", result.errors.len());
    }
    Ok(SyntaxNode::new_root(result.green_node))
}

struct NextTarget {
    name: String,
    inline_expr: Option<Expr>,
}

fn config_next_target(config: Option<&Path>, tree: &SyntaxNode) -> Result<NextTarget> {
    let Some(config) = config else {
        return Ok(NextTarget::named("Next"));
    };
    if !config.is_file() {
        return Ok(NextTarget::named("Next"));
    }
    let text =
        std::fs::read_to_string(config).with_context(|| format!("read {}", config.display()))?;
    let parsed = Config::parse(&text).map_err(|errors| {
        for err in &errors {
            eprintln!("{}:{}: {}", config.display(), err.line(), err);
        }
        anyhow::anyhow!("config parse failed with {} error(s)", errors.len())
    })?;
    if parsed.specification_conflicts_with_init_next() {
        bail!("SPECIFICATION and INIT/NEXT are mutually exclusive");
    }

    if let Some(next) = &parsed.next {
        return Ok(NextTarget::named(next.clone()));
    }

    if parsed.specification.is_some() {
        let resolved = resolve_spec_from_config(&parsed, tree)
            .map_err(|err| anyhow::anyhow!("resolve SPECIFICATION failed: {err}"))?;
        let inline_expr = resolved
            .next_node
            .as_ref()
            .map(|node| {
                lower_single_expr(FileId(0), node)
                    .with_context(|| format!("lower inline NEXT expression `{}`", node.text()))
            })
            .transpose()?;
        return Ok(NextTarget {
            name: resolved.next,
            inline_expr,
        });
    }

    Ok(NextTarget::named("Next"))
}

impl NextTarget {
    fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inline_expr: None,
        }
    }
}

fn analyze_module(
    name: Option<String>,
    file: &Path,
    config: Option<&Path>,
    module: &Module,
    next_target: &NextTarget,
) -> Result<SpecCoverage> {
    let operators = operators_by_name(module);
    let next_expr = if let Some(inline_expr) = &next_target.inline_expr {
        inline_expr
    } else {
        let next_name = next_target.name.as_str();
        &operators
            .get(next_name)
            .with_context(|| format!("NEXT operator `{next_name}` not found"))?
            .body
            .node
    };

    let mut actions = Vec::new();
    collect_action_exprs(next_expr, &operators, &mut actions, 0);

    let mut classified = Vec::with_capacity(actions.len());
    for action in actions {
        let class = classify_expr(action.expr);
        classified.push(ActionCoverage {
            name: action.name,
            classification: class.classification,
            shape: class.shape,
            diagnostic: class.diagnostic,
        });
    }

    let counts = CoverageCounts::from_actions(&classified);
    Ok(SpecCoverage {
        status: SpecStatus::Ok,
        spec: name.unwrap_or_else(|| module.name.node.clone()),
        module: Some(module.name.node.clone()),
        file: file.display().to_string(),
        config: config.map(|p| p.display().to_string()),
        next: Some(next_target.name.clone()),
        counts,
        actions: classified,
        error: None,
    })
}

fn operators_by_name(module: &Module) -> BTreeMap<String, &OperatorDef> {
    module
        .units
        .iter()
        .filter_map(|unit| match &unit.node {
            Unit::Operator(def) => Some((def.name.node.clone(), def)),
            _ => None,
        })
        .collect()
}

#[derive(Clone)]
struct ActionExpr<'a> {
    name: String,
    expr: &'a Expr,
}

fn collect_action_exprs<'a>(
    expr: &'a Expr,
    operators: &BTreeMap<String, &'a OperatorDef>,
    out: &mut Vec<ActionExpr<'a>>,
    depth: usize,
) {
    if depth > 16 {
        out.push(ActionExpr {
            name: "<recursion-limit>".to_string(),
            expr,
        });
        return;
    }

    match expr {
        Expr::Or(left, right) => {
            collect_action_exprs(&left.node, operators, out, depth);
            collect_action_exprs(&right.node, operators, out, depth);
        }
        Expr::Ident(name, _) => collect_named_action(name, operators, out, depth, expr),
        Expr::Apply(op, _) => {
            if let Expr::Ident(name, _) = &op.node {
                collect_named_action(name, operators, out, depth, expr);
            } else {
                out.push(ActionExpr {
                    name: action_name(expr),
                    expr,
                });
            }
        }
        Expr::Label(label) => collect_action_exprs(&label.body.node, operators, out, depth),
        _ => out.push(ActionExpr {
            name: action_name(expr),
            expr,
        }),
    }
}

fn collect_named_action<'a>(
    name: &str,
    operators: &BTreeMap<String, &'a OperatorDef>,
    out: &mut Vec<ActionExpr<'a>>,
    depth: usize,
    fallback: &'a Expr,
) {
    let Some(def) = operators.get(name) else {
        out.push(ActionExpr {
            name: name.to_string(),
            expr: fallback,
        });
        return;
    };

    if matches!(def.body.node, Expr::Or(_, _)) {
        collect_action_exprs(&def.body.node, operators, out, depth + 1);
    } else {
        out.push(ActionExpr {
            name: name.to_string(),
            expr: &def.body.node,
        });
    }
}

fn action_name(expr: &Expr) -> String {
    match expr {
        Expr::Apply(op, _) => match &op.node {
            Expr::Ident(name, _) => name.clone(),
            Expr::ModuleRef(target, name, _) => format!("{target}!{name}"),
            _ => "<anonymous>".to_string(),
        },
        Expr::ModuleRef(target, name, _) => format!("{target}!{name}"),
        Expr::Ident(name, _) => name.clone(),
        _ => "<anonymous>".to_string(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClassifiedAction {
    classification: ActionClassification,
    shape: ActionShape,
    diagnostic: &'static str,
}

fn classify_expr(expr: &Expr) -> ClassifiedAction {
    let expr = strip_action_wrapper(expr);

    if let Some(summary) = exists_chain_summary(expr) {
        return classify_exists_summary(
            summary,
            ExistsDiagnostics {
                single: ActionShape::TopLevelExistsSingleBound,
                multi: ActionShape::TopLevelExistsMultiBound,
                optional_single: ActionShape::OptionalExistsSingleBoundDisjunction,
                optional_multi: ActionShape::OptionalExistsMultiBoundDisjunction,
                single_diagnostic: "top-level EXISTS chain binds one value",
                multi_diagnostic: "top-level EXISTS chain binds multiple values",
                optional_single_diagnostic:
                    "top-level EXISTS chain contains one optional disjunction EXISTS branch",
                optional_multi_diagnostic:
                    "top-level EXISTS chain contains an optional disjunction EXISTS branch and binds multiple values",
            },
        );
    }

    if let Some(summary) = guarded_exists_chain_summary(expr) {
        return classify_exists_summary(
            summary,
            ExistsDiagnostics {
                single: ActionShape::GuardedExistsSingleBoundConjunction,
                multi: ActionShape::GuardedExistsMultiBoundConjunction,
                optional_single: ActionShape::OptionalExistsSingleBoundDisjunction,
                optional_multi: ActionShape::OptionalExistsMultiBoundDisjunction,
                single_diagnostic: "conjunction contains exactly one supported EXISTS chain",
                multi_diagnostic:
                    "conjunction contains one supported EXISTS chain that binds multiple values",
                optional_single_diagnostic:
                    "conjunction contains one supported optional disjunction EXISTS branch",
                optional_multi_diagnostic:
                    "conjunction contains an optional disjunction EXISTS branch and binds multiple values",
            },
        );
    }

    if contains_exists(expr) {
        ClassifiedAction {
            classification: ActionClassification::OtherUnsupported,
            shape: ActionShape::OtherUnsupportedExists,
            diagnostic: "contains an EXISTS shape outside the supported top-level, guarded conjunction, or optional disjunction forms",
        }
    } else {
        ClassifiedAction {
            classification: ActionClassification::BindingFree,
            shape: ActionShape::BindingFree,
            diagnostic: "no EXISTS appears in the action expression",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExistsDiagnostics {
    single: ActionShape,
    multi: ActionShape,
    optional_single: ActionShape,
    optional_multi: ActionShape,
    single_diagnostic: &'static str,
    multi_diagnostic: &'static str,
    optional_single_diagnostic: &'static str,
    optional_multi_diagnostic: &'static str,
}

fn classify_exists_summary(
    summary: ExistsChainSummary,
    diagnostics: ExistsDiagnostics,
) -> ClassifiedAction {
    match (summary.binders, summary.contains_optional_disjunction) {
        (1, false) => ClassifiedAction {
            classification: ActionClassification::ExistsSingleBound,
            shape: diagnostics.single,
            diagnostic: diagnostics.single_diagnostic,
        },
        (n, false) if n > 1 => ClassifiedAction {
            classification: ActionClassification::ExistsMultiBound,
            shape: diagnostics.multi,
            diagnostic: diagnostics.multi_diagnostic,
        },
        (1, true) => ClassifiedAction {
            classification: ActionClassification::ExistsSingleBound,
            shape: diagnostics.optional_single,
            diagnostic: diagnostics.optional_single_diagnostic,
        },
        (n, true) if n > 1 => ClassifiedAction {
            classification: ActionClassification::ExistsMultiBound,
            shape: diagnostics.optional_multi,
            diagnostic: diagnostics.optional_multi_diagnostic,
        },
        _ => ClassifiedAction {
            classification: ActionClassification::OtherUnsupported,
            shape: ActionShape::OtherUnsupportedExists,
            diagnostic: "supported EXISTS shape has no binders",
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExistsChainSummary {
    binders: usize,
    contains_optional_disjunction: bool,
}

impl ExistsChainSummary {
    fn binders(binders: usize) -> Self {
        Self {
            binders,
            contains_optional_disjunction: false,
        }
    }

    fn optional_branch(summary: Self) -> Self {
        Self {
            binders: summary.binders,
            contains_optional_disjunction: true,
        }
    }

    fn with_nested(self, nested: Self) -> Self {
        Self {
            binders: self.binders + nested.binders,
            contains_optional_disjunction: self.contains_optional_disjunction
                || nested.contains_optional_disjunction,
        }
    }
}

fn supported_conjunct_summary(expr: &Expr) -> Option<ExistsChainSummary> {
    exists_chain_summary(expr).or_else(|| optional_disjunction_exists_summary(expr))
}

fn supported_disjunct_summary(expr: &Expr) -> Option<ExistsChainSummary> {
    exists_chain_summary(expr).or_else(|| guarded_exists_chain_summary(expr))
}

fn exists_chain_summary(expr: &Expr) -> Option<ExistsChainSummary> {
    match strip_action_wrapper(expr) {
        Expr::Exists(bounds, body) => {
            let current = count_binders(bounds);
            if current == 0 {
                return None;
            }
            let summary = ExistsChainSummary::binders(current);
            match guarded_exists_chain_summary(&body.node) {
                Some(nested) => Some(summary.with_nested(nested)),
                None if contains_exists(strip_action_wrapper(&body.node)) => None,
                None => Some(summary),
            }
        }
        _ => None,
    }
}

fn guarded_exists_chain_summary(expr: &Expr) -> Option<ExistsChainSummary> {
    let mut summary: Option<ExistsChainSummary> = None;
    let mut exists_sites = 0;
    let mut all_supported = true;
    visit_conjuncts(
        strip_action_wrapper(expr),
        &mut |conjunct| match supported_conjunct_summary(conjunct) {
            Some(conjunct_summary) => {
                summary = Some(match summary {
                    Some(existing) => existing.with_nested(conjunct_summary),
                    None => conjunct_summary,
                });
                exists_sites += 1;
            }
            None if contains_exists(strip_action_wrapper(conjunct)) => all_supported = false,
            None => {}
        },
    );

    if all_supported && exists_sites == 1 {
        summary
    } else {
        None
    }
}

fn optional_disjunction_exists_summary(expr: &Expr) -> Option<ExistsChainSummary> {
    let expr = strip_action_wrapper(expr);
    if !matches!(strip_label(expr), Expr::Or(_, _)) {
        return None;
    }

    let mut summary: Option<ExistsChainSummary> = None;
    let mut exists_branches = 0;
    let mut all_supported = true;
    visit_disjuncts(
        expr,
        &mut |disjunct| match supported_disjunct_summary(disjunct) {
            Some(disjunct_summary) => {
                summary = Some(match summary {
                    Some(existing) => existing.with_nested(disjunct_summary),
                    None => disjunct_summary,
                });
                exists_branches += 1;
            }
            None if contains_exists(strip_action_wrapper(disjunct)) => all_supported = false,
            None => {}
        },
    );

    if all_supported && exists_branches == 1 {
        summary.map(ExistsChainSummary::optional_branch)
    } else {
        None
    }
}

fn visit_disjuncts<'a>(expr: &'a Expr, visit: &mut impl FnMut(&'a Expr)) {
    match strip_label(expr) {
        Expr::Or(left, right) => {
            visit_disjuncts(&left.node, visit);
            visit_disjuncts(&right.node, visit);
        }
        other => visit(other),
    }
}

fn strip_label(expr: &Expr) -> &Expr {
    match expr {
        Expr::Label(label) => strip_label(&label.body.node),
        other => other,
    }
}

fn strip_action_wrapper(expr: &Expr) -> &Expr {
    match strip_label(expr) {
        Expr::Let(defs, body) if defs.iter().all(|def| !contains_exists(&def.body.node)) => {
            strip_action_wrapper(&body.node)
        }
        other => other,
    }
}

fn count_binders(bounds: &[BoundVar]) -> usize {
    bounds
        .iter()
        .map(|bound| match &bound.pattern {
            Some(BoundPattern::Tuple(items)) => items.len(),
            Some(BoundPattern::Var(_)) | None => 1,
        })
        .sum()
}

fn visit_conjuncts<'a>(expr: &'a Expr, visit: &mut impl FnMut(&'a Expr)) {
    match strip_label(expr) {
        Expr::And(left, right) => {
            visit_conjuncts(&left.node, visit);
            visit_conjuncts(&right.node, visit);
        }
        other => visit(other),
    }
}

fn contains_exists(expr: &Expr) -> bool {
    match expr {
        Expr::Exists(_, _) => true,
        Expr::Apply(op, args) => {
            contains_exists(&op.node) || args.iter().any(|arg| contains_exists(&arg.node))
        }
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::FuncSet(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b)
        | Expr::FuncApply(a, b)
        | Expr::WeakFair(a, b)
        | Expr::StrongFair(a, b)
        | Expr::LeadsTo(a, b) => contains_exists(&a.node) || contains_exists(&b.node),
        Expr::Not(e)
        | Expr::Powerset(e)
        | Expr::BigUnion(e)
        | Expr::Domain(e)
        | Expr::Prime(e)
        | Expr::Enabled(e)
        | Expr::Unchanged(e)
        | Expr::Neg(e)
        | Expr::Always(e)
        | Expr::Eventually(e)
        | Expr::RecordAccess(e, _) => contains_exists(&e.node),
        Expr::Forall(bounds, body)
        | Expr::FuncDef(bounds, body)
        | Expr::SetBuilder(body, bounds) => {
            bounds_have_exists(bounds) || contains_exists(&body.node)
        }
        Expr::Choose(bound, body) | Expr::SetFilter(bound, body) => {
            bound_has_exists(bound) || contains_exists(&body.node)
        }
        Expr::SetEnum(items) | Expr::Tuple(items) | Expr::Times(items) => {
            items.iter().any(|item| contains_exists(&item.node))
        }
        Expr::Record(fields) => fields.iter().any(|(_, value)| contains_exists(&value.node)),
        Expr::RecordSet(fields) => fields.iter().any(|(_, value)| contains_exists(&value.node)),
        Expr::Case(arms, other) => {
            arms.iter()
                .any(|arm| contains_exists(&arm.guard.node) || contains_exists(&arm.body.node))
                || other
                    .as_ref()
                    .is_some_and(|other| contains_exists(&other.node))
        }
        Expr::If(condition, then_expr, else_expr) => {
            contains_exists(&condition.node)
                || contains_exists(&then_expr.node)
                || contains_exists(&else_expr.node)
        }
        Expr::Let(defs, body) => {
            defs.iter().any(|def| contains_exists(&def.body.node)) || contains_exists(&body.node)
        }
        Expr::InstanceExpr(_, substitutions) => substitutions
            .iter()
            .any(|sub| contains_exists(&sub.to.node)),
        Expr::SubstIn(substitutions, body) => {
            substitutions
                .iter()
                .any(|sub| contains_exists(&sub.to.node))
                || contains_exists(&body.node)
        }
        Expr::ModuleRef(_, _, args) => args.iter().any(|arg| contains_exists(&arg.node)),
        Expr::Label(label) => contains_exists(&label.body.node),
        Expr::Except(base, specs) => {
            contains_exists(&base.node)
                || specs.iter().any(|spec| {
                    spec.path.iter().any(|element| match element {
                        ExceptPathElement::Index(index) => contains_exists(&index.node),
                        ExceptPathElement::Field(_) => false,
                    }) || contains_exists(&spec.value.node)
                })
        }
        Expr::Bool(_)
        | Expr::Int(_)
        | Expr::String(_)
        | Expr::Ident(_, _)
        | Expr::StateVar(_, _, _)
        | Expr::OpRef(_)
        | Expr::Lambda(_, _) => false,
    }
}

fn bounds_have_exists(bounds: &[BoundVar]) -> bool {
    bounds.iter().any(bound_has_exists)
}

fn bound_has_exists(bound: &BoundVar) -> bool {
    bound
        .domain
        .as_ref()
        .is_some_and(|domain| contains_exists(&domain.node))
}

fn write_json(path: &Path, report: &CoverageReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)? + "\n")
        .with_context(|| format!("write {}", path.display()))
}

fn write_markdown(path: &Path, report: &CoverageReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }

    let mut out = String::new();
    out.push_str("# trust-codegen Action Coverage\n\n");
    let _ = write!(
        out,
        "- Specs: {}\n- Actions: {}\n- Binding-free: {}\n- Single-EXISTS-bound: {}\n- Multi-EXISTS-bound: {}\n- Unsupported: {}\n- Specs with 0% compiled-path candidates: {}\n\n",
        report.summary.specs,
        report.summary.total_actions,
        report.summary.binding_free,
        report.summary.exists_single_bound,
        report.summary.exists_multi_bound,
        report.summary.other_unsupported,
        report.summary.zero_compiled_candidate_specs,
    );
    out.push_str("| Spec | Total | Binding-free | Single EXISTS | Unsupported | Status |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | --- |\n");
    for spec in report.specs.values() {
        let unsupported = spec.counts.exists_multi_bound + spec.counts.other_unsupported;
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} |",
            spec.spec,
            spec.counts.total_actions,
            spec.counts.binding_free,
            spec.counts.exists_single_bound,
            unsupported,
            spec.status.as_str(),
        );
    }
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))
}

fn print_human(report: &CoverageReport) {
    println!("trust-cg action coverage");
    println!("  specs: {}", report.summary.specs);
    println!("  actions: {}", report.summary.total_actions);
    println!("  binding_free: {}", report.summary.binding_free);
    println!(
        "  exists_single_bound: {}",
        report.summary.exists_single_bound
    );
    println!("  unsupported: {}", report.summary.unsupported());
}

#[derive(Debug, Deserialize)]
struct Baseline {
    inputs: BaselineInputs,
    specs: BTreeMap<String, BaselineSpec>,
}

#[derive(Debug, Deserialize)]
struct BaselineInputs {
    examples_dir: String,
}

#[derive(Debug, Deserialize)]
struct BaselineSpec {
    source: Option<BaselineSource>,
}

#[derive(Debug, Deserialize)]
struct BaselineSource {
    tla_path: String,
    cfg_path: String,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Debug, Serialize)]
struct CoverageReport {
    schema: &'static str,
    schema_version: u32,
    generated_at: String,
    summary: CoverageSummary,
    specs: BTreeMap<String, SpecCoverage>,
}

impl CoverageReport {
    fn new(specs: BTreeMap<String, SpecCoverage>) -> Self {
        let summary = CoverageSummary::from_specs(&specs);
        Self {
            schema: SCHEMA,
            schema_version: 1,
            generated_at: chrono::Utc::now().to_rfc3339(),
            summary,
            specs,
        }
    }

    fn single(spec: SpecCoverage) -> Self {
        let mut specs = BTreeMap::new();
        specs.insert(spec.spec.clone(), spec);
        Self::new(specs)
    }
}

#[derive(Debug, Serialize)]
struct CoverageSummary {
    specs: usize,
    ok_specs: usize,
    skipped_specs: usize,
    error_specs: usize,
    total_actions: usize,
    binding_free: usize,
    exists_single_bound: usize,
    exists_multi_bound: usize,
    other_unsupported: usize,
    zero_compiled_candidate_specs: usize,
}

impl CoverageSummary {
    fn from_specs(specs: &BTreeMap<String, SpecCoverage>) -> Self {
        let mut summary = Self {
            specs: specs.len(),
            ok_specs: 0,
            skipped_specs: 0,
            error_specs: 0,
            total_actions: 0,
            binding_free: 0,
            exists_single_bound: 0,
            exists_multi_bound: 0,
            other_unsupported: 0,
            zero_compiled_candidate_specs: 0,
        };

        for spec in specs.values() {
            match spec.status {
                SpecStatus::Ok => summary.ok_specs += 1,
                SpecStatus::Skipped => summary.skipped_specs += 1,
                SpecStatus::Error => summary.error_specs += 1,
            }
            summary.total_actions += spec.counts.total_actions;
            summary.binding_free += spec.counts.binding_free;
            summary.exists_single_bound += spec.counts.exists_single_bound;
            summary.exists_multi_bound += spec.counts.exists_multi_bound;
            summary.other_unsupported += spec.counts.other_unsupported;
            if spec.status == SpecStatus::Ok
                && spec.counts.binding_free + spec.counts.exists_single_bound == 0
            {
                summary.zero_compiled_candidate_specs += 1;
            }
        }
        summary
    }

    fn unsupported(&self) -> usize {
        self.exists_multi_bound + self.other_unsupported
    }
}

#[derive(Debug, Serialize)]
struct SpecCoverage {
    status: SpecStatus,
    spec: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    module: Option<String>,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next: Option<String>,
    counts: CoverageCounts,
    actions: Vec<ActionCoverage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl SpecCoverage {
    fn skip(spec: String, reason: impl Into<String>) -> Self {
        Self {
            status: SpecStatus::Skipped,
            spec,
            module: None,
            file: String::new(),
            config: None,
            next: None,
            counts: CoverageCounts::default(),
            actions: Vec::new(),
            error: Some(reason.into()),
        }
    }

    fn error(spec: String, file: PathBuf, config: PathBuf, error: String) -> Self {
        Self {
            status: SpecStatus::Error,
            spec,
            module: None,
            file: file.display().to_string(),
            config: Some(config.display().to_string()),
            next: None,
            counts: CoverageCounts::default(),
            actions: Vec::new(),
            error: Some(error),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SpecStatus {
    Ok,
    Skipped,
    Error,
}

impl SpecStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Skipped => "skipped",
            Self::Error => "error",
        }
    }
}

#[derive(Default, Debug, Serialize)]
struct CoverageCounts {
    total_actions: usize,
    binding_free: usize,
    exists_single_bound: usize,
    exists_multi_bound: usize,
    other_unsupported: usize,
}

impl CoverageCounts {
    fn from_actions(actions: &[ActionCoverage]) -> Self {
        let mut counts = Self {
            total_actions: actions.len(),
            ..Self::default()
        };
        for action in actions {
            match action.classification {
                ActionClassification::BindingFree => counts.binding_free += 1,
                ActionClassification::ExistsSingleBound => counts.exists_single_bound += 1,
                ActionClassification::ExistsMultiBound => counts.exists_multi_bound += 1,
                ActionClassification::OtherUnsupported => counts.other_unsupported += 1,
            }
        }
        counts
    }
}

#[derive(Debug, Serialize)]
struct ActionCoverage {
    name: String,
    classification: ActionClassification,
    shape: ActionShape,
    diagnostic: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ActionClassification {
    BindingFree,
    ExistsSingleBound,
    ExistsMultiBound,
    OtherUnsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ActionShape {
    BindingFree,
    TopLevelExistsSingleBound,
    TopLevelExistsMultiBound,
    GuardedExistsSingleBoundConjunction,
    GuardedExistsMultiBoundConjunction,
    OptionalExistsSingleBoundDisjunction,
    OptionalExistsMultiBoundDisjunction,
    OtherUnsupportedExists,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze_source(source: &str) -> SpecCoverage {
        let tree = parse(source);
        assert!(tree.errors.is_empty(), "{:?}", tree.errors);
        let syntax = SyntaxNode::new_root(tree.green_node);
        let lowered = lower_main_module(FileId(0), &syntax, Some("Test"));
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let module = lowered.module.unwrap();
        let next = NextTarget::named("Next");
        analyze_module(None, Path::new("Test.tla"), None, &module, &next).unwrap()
    }

    #[test]
    fn classifies_flat_next_disjunction() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
VARIABLE x
Init == x = 0
A == x' = x + 1
B == \E i \in {1, 2}: x' = i
C == \E i \in {1, 2}, j \in {3, 4}: x' = i + j
Next == A \/ B \/ C
===="#,
        );

        assert_eq!(spec.counts.total_actions, 3);
        assert_eq!(spec.counts.binding_free, 1);
        assert_eq!(spec.counts.exists_single_bound, 1);
        assert_eq!(spec.counts.exists_multi_bound, 1);
        assert_eq!(
            spec.actions
                .iter()
                .map(|a| (a.name.as_str(), a.classification, a.shape))
                .collect::<Vec<_>>(),
            vec![
                (
                    "A",
                    ActionClassification::BindingFree,
                    ActionShape::BindingFree
                ),
                (
                    "B",
                    ActionClassification::ExistsSingleBound,
                    ActionShape::TopLevelExistsSingleBound
                ),
                (
                    "C",
                    ActionClassification::ExistsMultiBound,
                    ActionShape::TopLevelExistsMultiBound
                ),
            ]
        );
    }

    #[test]
    fn let_wrapped_single_exists_is_supported() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next == LET candidates == {1, 2}
        IN \E i \in candidates : x' = i
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.exists_single_bound, 1);
        assert_eq!(spec.counts.other_unsupported, 0);
        assert_eq!(
            spec.actions[0].classification,
            ActionClassification::ExistsSingleBound
        );
        assert_eq!(
            spec.actions[0].shape,
            ActionShape::TopLevelExistsSingleBound
        );
    }

    #[test]
    fn guarded_single_exists_conjunction_is_supported() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = 1 /\ (\E i \in {1}: i = 1)
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.exists_single_bound, 1);
        assert_eq!(
            spec.actions[0].classification,
            ActionClassification::ExistsSingleBound
        );
        assert_eq!(
            spec.actions[0].shape,
            ActionShape::GuardedExistsSingleBoundConjunction
        );
    }

    #[test]
    fn optional_single_exists_disjunction_conjunct_is_supported() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next ==
  /\ x >= 0
  /\ (x' = x
      \/ /\ x < 10
         /\ \E i \in {1, 2}: x' = i)
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.exists_single_bound, 1);
        assert_eq!(spec.counts.other_unsupported, 0);
        assert_eq!(
            spec.actions[0].classification,
            ActionClassification::ExistsSingleBound
        );
        assert_eq!(
            spec.actions[0].shape,
            ActionShape::OptionalExistsSingleBoundDisjunction
        );
    }

    #[test]
    fn nested_optional_exists_disjunction_counts_all_binders() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
EXTENDS Integers
VARIABLE x
Init == x = 0
Next ==
  /\ \E level \in {0, 1}:
      /\ \E prefix \in {0, 1}:
          /\ (x' = prefix
              \/ /\ prefix < 1
                 /\ \E lit \in {1, 2}: x' = lit)
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.exists_multi_bound, 1);
        assert_eq!(spec.counts.other_unsupported, 0);
        assert_eq!(
            spec.actions[0].classification,
            ActionClassification::ExistsMultiBound
        );
        assert_eq!(
            spec.actions[0].shape,
            ActionShape::OptionalExistsMultiBoundDisjunction
        );
    }

    #[test]
    fn two_exists_disjunction_branches_are_other_unsupported() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x = 0 /\ ((\E i \in {1}: x' = i) \/ (\E j \in {2}: x' = j))
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.other_unsupported, 1);
        assert_eq!(
            spec.actions[0].classification,
            ActionClassification::OtherUnsupported
        );
        assert_eq!(spec.actions[0].shape, ActionShape::OtherUnsupportedExists);
    }

    #[test]
    fn ewd998_chan_id_actions_get_precise_exists_shape_diagnostics() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
EXTENDS Integers, Sequences
CONSTANT Node
VARIABLE active, counter, inbox
vars == <<active, counter, inbox>>
InitiateProbe(n) ==
  /\ n = "n1"
  /\ \E j \in 1..Len(inbox[n]):
      /\ inbox[n][j] = "tok"
      /\ inbox' = [inbox EXCEPT ![n] = Tail(@)]
  /\ UNCHANGED <<active, counter>>
PassToken(n) ==
  /\ n # "n1"
  /\ ~ active[n]
  /\ \E j \in 1..Len(inbox[n]):
      /\ inbox[n][j] = "tok"
      /\ inbox' = [inbox EXCEPT ![n] = Tail(@)]
  /\ UNCHANGED <<active, counter>>
SendMsg(n) ==
  /\ active[n]
  /\ counter' = [counter EXCEPT ![n] = @ + 1]
  /\ \E j \in Node \ {n}:
      /\ inbox' = [inbox EXCEPT ![j] = Append(@, n)]
  /\ UNCHANGED active
RecvMsg(n) ==
  /\ counter' = [counter EXCEPT ![n] = @ - 1]
  /\ \E j \in 1..Len(inbox[n]):
      /\ inbox[n][j] = "pl"
      /\ inbox' = [inbox EXCEPT ![n] = Tail(@)]
  /\ UNCHANGED active
Deactivate(n) ==
  /\ active[n]
  /\ active' = [active EXCEPT ![n] = FALSE]
  /\ UNCHANGED <<counter, inbox>>
System(n) == InitiateProbe(n) \/ PassToken(n)
Environment(n) == SendMsg(n) \/ RecvMsg(n) \/ Deactivate(n)
Next == System("n1") \/ Environment("n1")
===="#,
        );

        assert_eq!(spec.counts.total_actions, 5);
        assert_eq!(spec.counts.exists_single_bound, 4);
        assert_eq!(spec.counts.other_unsupported, 0);
        assert_eq!(
            spec.actions
                .iter()
                .map(|a| (a.name.as_str(), a.shape))
                .collect::<Vec<_>>(),
            vec![
                (
                    "InitiateProbe",
                    ActionShape::GuardedExistsSingleBoundConjunction
                ),
                (
                    "PassToken",
                    ActionShape::GuardedExistsSingleBoundConjunction
                ),
                ("SendMsg", ActionShape::GuardedExistsSingleBoundConjunction),
                ("RecvMsg", ActionShape::GuardedExistsSingleBoundConjunction),
                ("Deactivate", ActionShape::BindingFree),
            ]
        );
    }

    #[test]
    fn linear_nested_exists_body_is_multi_bound() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == x' = 1 /\ (\E i \in {1}: \E j \in {1}: i = j)
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.exists_multi_bound, 1);
        assert_eq!(spec.counts.other_unsupported, 0);
        assert_eq!(
            spec.actions[0].classification,
            ActionClassification::ExistsMultiBound
        );
    }

    #[test]
    fn two_guarded_single_exists_conjuncts_are_other_unsupported() {
        let spec = analyze_source(
            r#"---- MODULE Test ----
VARIABLE x
Init == x = 0
Next == (\E i \in {1}: x' = i) /\ (\E j \in {2}: x' = j)
===="#,
        );

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.other_unsupported, 1);
    }

    #[test]
    fn resolves_specification_config_to_next_operator() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tla = dir.path().join("Test.tla");
        let cfg = dir.path().join("Test.cfg");
        std::fs::write(
            &tla,
            r#"---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
A == x' = x + 1
B == \E i \in {1, 2}: x' = i
Next == A \/ B
Spec == Init /\ [][Next]_vars
===="#,
        )
        .expect("write tla");
        std::fs::write(&cfg, "SPECIFICATION Spec\n").expect("write cfg");

        let spec = analyze_spec(None, &tla, Some(&cfg)).expect("analyze spec");

        assert_eq!(spec.next.as_deref(), Some("Next"));
        assert_eq!(spec.counts.total_actions, 2);
        assert_eq!(spec.counts.binding_free, 1);
        assert_eq!(spec.counts.exists_single_bound, 1);
    }

    #[test]
    fn classifies_inline_specification_next_expression() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tla = dir.path().join("Test.tla");
        let cfg = dir.path().join("Test.cfg");
        std::fs::write(
            &tla,
            r#"---- MODULE Test ----
VARIABLE x
vars == <<x>>
Init == x = 0
Spec == Init /\ [][\E i \in {1, 2}: x' = i]_vars
===="#,
        )
        .expect("write tla");
        std::fs::write(&cfg, "SPECIFICATION Spec\n").expect("write cfg");

        let spec = analyze_spec(None, &tla, Some(&cfg)).expect("analyze spec");

        assert_eq!(spec.counts.total_actions, 1);
        assert_eq!(spec.counts.exists_single_bound, 1);
    }
}
