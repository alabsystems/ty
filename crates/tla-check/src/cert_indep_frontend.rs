// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! A SECOND, fully independent TLA+ front end for the certifiable SCALAR fragment.
//!
//! Certifying verification's Leg D part-2 binds the embedded AY obligation to the
//! spec by (4) re-translating through TY's own `BmcTranslator` and (5) an
//! engine-diverse `tla-eval` probe cross-check. Both (4) and (5) still parse the
//! spec with `tla_core::{parse_to_syntax_tree, lower}` — so a *front-end* bug is
//! invisible to them.
//!
//! This module is a THIRD truth source that shares NOTHING with `tla_core` or
//! `BmcTranslator`: a hand-written tokenizer + recursive-descent parser + direct
//! evaluator for the scalar fragment (integer/boolean arithmetic, comparisons,
//! boolean connectives, primed variables, and 0-ary operator definitions). The
//! caller evaluates the obligation at probe states through BOTH this independent
//! path and the embedded AY obligation; agreement removes ALL of TY's
//! front-end+translator trust for scalar specs — a genuinely third-party-style
//! check, in-tree.
//!
//! FAIL-CLOSED: any construct outside the supported fragment (sets, functions,
//! records, sequences, quantifiers, `LET`, `IF`, `CHOOSE`, unknown operators,
//! multi-line oddities) makes parsing return `None`, and the caller keeps the
//! existing bindings. The module never *accepts* — it can only refute.

use std::collections::HashMap;

/// A scalar value in the independent evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IVal {
    Int(i64),
    Bool(bool),
}

/// A concrete state: state-variable name -> scalar value.
pub(crate) type IState = HashMap<String, IVal>;

/// The independent AST for the scalar fragment.
#[derive(Clone, Debug)]
enum IExpr {
    Int(i64),
    Bool(bool),
    Var(String),
    Prime(Box<IExpr>),
    NegArith(Box<IExpr>),
    Add(Box<IExpr>, Box<IExpr>),
    Sub(Box<IExpr>, Box<IExpr>),
    Mul(Box<IExpr>, Box<IExpr>),
    Div(Box<IExpr>, Box<IExpr>),
    Mod(Box<IExpr>, Box<IExpr>),
    Eq(Box<IExpr>, Box<IExpr>),
    Neq(Box<IExpr>, Box<IExpr>),
    Lt(Box<IExpr>, Box<IExpr>),
    Le(Box<IExpr>, Box<IExpr>),
    Gt(Box<IExpr>, Box<IExpr>),
    Ge(Box<IExpr>, Box<IExpr>),
    Not(Box<IExpr>),
    And(Box<IExpr>, Box<IExpr>),
    Or(Box<IExpr>, Box<IExpr>),
    Implies(Box<IExpr>, Box<IExpr>),
    Iff(Box<IExpr>, Box<IExpr>),
}

// ===========================================================================
// Tokenizer
// ===========================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok {
    Int(i64),
    Bool(bool),
    Ident(String),
    Plus,
    Minus,
    Star,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    Implies,
    Iff,
    LParen,
    RParen,
    Prime,
    DefEq,
}

/// Tokenize the supported scalar fragment. `None` on any character/operator
/// outside the fragment (fail-closed).
fn tokenize(src: &str) -> Option<Vec<Tok>> {
    let b = src.as_bytes();
    let mut i = 0;
    let n = b.len();
    let mut out = Vec::new();
    while i < n {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // line comment `\* ... EOL`
        if c == b'\\' && i + 1 < n && b[i + 1] == b'*' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // block comment `(* ... *)`
        if c == b'(' && i + 1 < n && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b')') {
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            while i < n && b[i].is_ascii_digit() {
                i += 1;
            }
            let s = std::str::from_utf8(&b[start..i]).ok()?;
            out.push(Tok::Int(s.parse().ok()?));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let s = std::str::from_utf8(&b[start..i]).ok()?;
            match s {
                "TRUE" => out.push(Tok::Bool(true)),
                "FALSE" => out.push(Tok::Bool(false)),
                _ => out.push(Tok::Ident(s.to_string())),
            }
            continue;
        }
        // backslash word operators
        if c == b'\\' && i + 1 < n && b[i + 1] != b'/' {
            let start = i + 1;
            let mut j = start;
            while j < n && b[j].is_ascii_alphabetic() {
                j += 1;
            }
            let word = std::str::from_utf8(&b[start..j]).ok()?;
            match word {
                "div" => out.push(Tok::Div),
                "leq" => out.push(Tok::Le),
                "geq" => out.push(Tok::Ge),
                "neq" => out.push(Tok::Neq),
                "land" => out.push(Tok::And),
                "lor" => out.push(Tok::Or),
                "lnot" | "neg" => out.push(Tok::Not),
                _ => return None,
            }
            i = j;
            continue;
        }
        // two/three-char operators
        if i + 1 < n {
            match [c, b[i + 1]] {
                [b'=', b'='] => {
                    out.push(Tok::DefEq);
                    i += 2;
                    continue;
                }
                [b'/', b'\\'] => {
                    out.push(Tok::And);
                    i += 2;
                    continue;
                }
                [b'\\', b'/'] => {
                    out.push(Tok::Or);
                    i += 2;
                    continue;
                }
                [b'<', b'='] => {
                    if i + 2 < n && b[i + 2] == b'>' {
                        out.push(Tok::Iff);
                        i += 3;
                    } else {
                        out.push(Tok::Le);
                        i += 2;
                    }
                    continue;
                }
                [b'>', b'='] => {
                    out.push(Tok::Ge);
                    i += 2;
                    continue;
                }
                [b'=', b'>'] => {
                    out.push(Tok::Implies);
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }
        match c {
            b'+' => out.push(Tok::Plus),
            b'-' => out.push(Tok::Minus),
            b'*' => out.push(Tok::Star),
            b'%' => out.push(Tok::Mod),
            b'=' => out.push(Tok::Eq),
            b'#' => out.push(Tok::Neq),
            b'<' => out.push(Tok::Lt),
            b'>' => out.push(Tok::Gt),
            b'~' => out.push(Tok::Not),
            b'(' => out.push(Tok::LParen),
            b')' => out.push(Tok::RParen),
            b'\'' => out.push(Tok::Prime),
            _ => return None,
        }
        i += 1;
    }
    Some(out)
}

// ===========================================================================
// Recursive-descent parser (precedence: <=> , => , \/ , /\ , ~ , compare ,
// +- , */div% , unary- , prime , atom). 0-ary operator references are expanded
// inline from `defs`.
// ===========================================================================

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    defs: &'a HashMap<String, Vec<Tok>>,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn bump(&mut self) -> Option<&Tok> {
        let t = self.toks.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_iff(&mut self) -> Option<IExpr> {
        let mut lhs = self.parse_implies()?;
        while self.eat(&Tok::Iff) {
            let rhs = self.parse_implies()?;
            lhs = IExpr::Iff(Box::new(lhs), Box::new(rhs));
        }
        Some(lhs)
    }
    fn parse_implies(&mut self) -> Option<IExpr> {
        let lhs = self.parse_or()?;
        // right-associative
        if self.eat(&Tok::Implies) {
            let rhs = self.parse_implies()?;
            Some(IExpr::Implies(Box::new(lhs), Box::new(rhs)))
        } else {
            Some(lhs)
        }
    }
    fn parse_or(&mut self) -> Option<IExpr> {
        let mut lhs = self.parse_and()?;
        while self.eat(&Tok::Or) {
            let rhs = self.parse_and()?;
            lhs = IExpr::Or(Box::new(lhs), Box::new(rhs));
        }
        Some(lhs)
    }
    fn parse_and(&mut self) -> Option<IExpr> {
        let mut lhs = self.parse_not()?;
        while self.eat(&Tok::And) {
            let rhs = self.parse_not()?;
            lhs = IExpr::And(Box::new(lhs), Box::new(rhs));
        }
        Some(lhs)
    }
    fn parse_not(&mut self) -> Option<IExpr> {
        if self.eat(&Tok::Not) {
            let e = self.parse_not()?;
            Some(IExpr::Not(Box::new(e)))
        } else {
            self.parse_compare()
        }
    }
    fn parse_compare(&mut self) -> Option<IExpr> {
        let lhs = self.parse_add()?;
        let op = match self.peek() {
            Some(Tok::Eq) => Some(0),
            Some(Tok::Neq) => Some(1),
            Some(Tok::Lt) => Some(2),
            Some(Tok::Le) => Some(3),
            Some(Tok::Gt) => Some(4),
            Some(Tok::Ge) => Some(5),
            _ => None,
        };
        if let Some(op) = op {
            self.pos += 1;
            let rhs = self.parse_add()?;
            let (l, r) = (Box::new(lhs), Box::new(rhs));
            Some(match op {
                0 => IExpr::Eq(l, r),
                1 => IExpr::Neq(l, r),
                2 => IExpr::Lt(l, r),
                3 => IExpr::Le(l, r),
                4 => IExpr::Gt(l, r),
                _ => IExpr::Ge(l, r),
            })
        } else {
            Some(lhs)
        }
    }
    fn parse_add(&mut self) -> Option<IExpr> {
        let mut lhs = self.parse_mul()?;
        loop {
            if self.eat(&Tok::Plus) {
                let rhs = self.parse_mul()?;
                lhs = IExpr::Add(Box::new(lhs), Box::new(rhs));
            } else if self.eat(&Tok::Minus) {
                let rhs = self.parse_mul()?;
                lhs = IExpr::Sub(Box::new(lhs), Box::new(rhs));
            } else {
                break;
            }
        }
        Some(lhs)
    }
    fn parse_mul(&mut self) -> Option<IExpr> {
        let mut lhs = self.parse_unary()?;
        loop {
            if self.eat(&Tok::Star) {
                let rhs = self.parse_unary()?;
                lhs = IExpr::Mul(Box::new(lhs), Box::new(rhs));
            } else if self.eat(&Tok::Div) {
                let rhs = self.parse_unary()?;
                lhs = IExpr::Div(Box::new(lhs), Box::new(rhs));
            } else if self.eat(&Tok::Mod) {
                let rhs = self.parse_unary()?;
                lhs = IExpr::Mod(Box::new(lhs), Box::new(rhs));
            } else {
                break;
            }
        }
        Some(lhs)
    }
    fn parse_unary(&mut self) -> Option<IExpr> {
        if self.eat(&Tok::Minus) {
            let e = self.parse_unary()?;
            Some(IExpr::NegArith(Box::new(e)))
        } else {
            self.parse_postfix()
        }
    }
    fn parse_postfix(&mut self) -> Option<IExpr> {
        let mut e = self.parse_atom()?;
        while self.eat(&Tok::Prime) {
            e = IExpr::Prime(Box::new(e));
        }
        Some(e)
    }
    fn parse_atom(&mut self) -> Option<IExpr> {
        match self.bump()?.clone() {
            Tok::Int(v) => Some(IExpr::Int(v)),
            Tok::Bool(b) => Some(IExpr::Bool(b)),
            Tok::LParen => {
                let e = self.parse_iff()?;
                if self.eat(&Tok::RParen) {
                    Some(e)
                } else {
                    None
                }
            }
            Tok::Ident(name) => {
                // A 0-ary operator reference is expanded inline; otherwise it is a
                // state variable. Guard recursion depth to bound mutual references.
                if let Some(body) = self.defs.get(&name) {
                    if self.depth > 64 {
                        return None;
                    }
                    let mut sub = Parser {
                        toks: body,
                        pos: 0,
                        defs: self.defs,
                        depth: self.depth + 1,
                    };
                    let e = sub.parse_iff()?;
                    if sub.pos != sub.toks.len() {
                        return None;
                    }
                    Some(e)
                } else {
                    Some(IExpr::Var(name))
                }
            }
            _ => None,
        }
    }
}

fn parse_expr(toks: &[Tok], defs: &HashMap<String, Vec<Tok>>) -> Option<IExpr> {
    let mut p = Parser {
        toks,
        pos: 0,
        defs,
        depth: 0,
    };
    let e = p.parse_iff()?;
    if p.pos == p.toks.len() {
        Some(e)
    } else {
        None
    }
}

// ===========================================================================
// Definition extraction from the module text (independent of tla_core).
// ===========================================================================

/// Scan a TLA module body for top-level 0-ary definitions `Name == <body>`,
/// returning `Name -> tokens(body)`. A definition's body runs until the next
/// top-level `Name ==` line, a `----`/`====` rule, or a declaration keyword. Any
/// body that fails to tokenize is dropped (it is simply unavailable for
/// expansion; if the obligation needs it, the overall parse fails -> `None`).
fn extract_defs(spec_src: &str) -> HashMap<String, Vec<Tok>> {
    let mut defs = HashMap::new();
    let lines: Vec<&str> = spec_src.lines().collect();
    // Identify the indices of lines that START a top-level definition.
    let is_def_start = |line: &str| -> Option<(String, usize)> {
        // `Name == ...` where Name is at the left margin (allow leading spaces).
        let t = line.trim_start();
        let indent = line.len() - t.len();
        // Heuristic: top-level defs are at indent 0; nested/continuation lines are
        // indented. This keeps multi-line bodies attached to their header.
        if indent != 0 {
            return None;
        }
        let bytes = t.as_bytes();
        if bytes.is_empty() || !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
            return None;
        }
        let mut k = 0;
        while k < t.len() && (bytes[k].is_ascii_alphanumeric() || bytes[k] == b'_') {
            k += 1;
        }
        let name = &t[..k];
        let rest = t[k..].trim_start();
        if let Some(body0) = rest.strip_prefix("==") {
            Some((name.to_string(), {
                // byte offset of body start within the original line
                let header_len = line.len() - body0.len();
                header_len
            }))
        } else {
            None
        }
    };
    let is_boundary = |line: &str| -> bool {
        let t = line.trim_start();
        t.starts_with("----")
            || t.starts_with("====")
            || t.starts_with("VARIABLE")
            || t.starts_with("VARIABLES")
            || t.starts_with("CONSTANT")
            || t.starts_with("CONSTANTS")
            || t.starts_with("EXTENDS")
            || t.starts_with("ASSUME")
            || t.starts_with("THEOREM")
            || t.starts_with("INSTANCE")
    };

    let mut i = 0;
    while i < lines.len() {
        if let Some((name, body_off)) = is_def_start(lines[i]) {
            let mut body = String::new();
            body.push_str(&lines[i][body_off..]);
            let mut j = i + 1;
            while j < lines.len() {
                let l = lines[j];
                if is_def_start(l).is_some() || is_boundary(l) {
                    break;
                }
                body.push('\n');
                body.push_str(l);
                j += 1;
            }
            if let Some(toks) = tokenize(&body) {
                defs.insert(name, toks);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    defs
}

// ===========================================================================
// Direct evaluator.
// ===========================================================================

fn eval(expr: &IExpr, cur: &IState, next: Option<&IState>) -> Option<IVal> {
    use IVal::{Bool, Int};
    let as_int = |v: IVal| -> Option<i64> {
        match v {
            Int(i) => Some(i),
            Bool(_) => None,
        }
    };
    let as_bool = |v: IVal| -> Option<bool> {
        match v {
            Bool(b) => Some(b),
            Int(_) => None,
        }
    };
    let bin_int = |a: &IExpr, b: &IExpr| -> Option<(i64, i64)> {
        Some((as_int(eval(a, cur, next)?)?, as_int(eval(b, cur, next)?)?))
    };
    let bin_bool = |a: &IExpr, b: &IExpr| -> Option<(bool, bool)> {
        Some((as_bool(eval(a, cur, next)?)?, as_bool(eval(b, cur, next)?)?))
    };
    Some(match expr {
        IExpr::Int(v) => Int(*v),
        IExpr::Bool(b) => Bool(*b),
        IExpr::Var(name) => *cur.get(name)?,
        IExpr::Prime(inner) => {
            // Only a primed *variable* is supported; `(x+1)'` is out of fragment.
            let next = next?;
            match inner.as_ref() {
                IExpr::Var(name) => *next.get(name)?,
                _ => return None,
            }
        }
        IExpr::NegArith(a) => Int(as_int(eval(a, cur, next)?)?.checked_neg()?),
        IExpr::Add(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Int(x.checked_add(y)?)
        }
        IExpr::Sub(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Int(x.checked_sub(y)?)
        }
        IExpr::Mul(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Int(x.checked_mul(y)?)
        }
        IExpr::Div(a, b) => {
            let (x, y) = bin_int(a, b)?;
            if y == 0 {
                return None;
            }
            // TLA+ integer division is Euclidean (floor toward -inf); match it.
            Int(x.div_euclid(y))
        }
        IExpr::Mod(a, b) => {
            let (x, y) = bin_int(a, b)?;
            if y == 0 {
                return None;
            }
            Int(x.rem_euclid(y))
        }
        IExpr::Eq(a, b) => Bool(eval(a, cur, next)? == eval(b, cur, next)?),
        IExpr::Neq(a, b) => Bool(eval(a, cur, next)? != eval(b, cur, next)?),
        IExpr::Lt(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Bool(x < y)
        }
        IExpr::Le(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Bool(x <= y)
        }
        IExpr::Gt(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Bool(x > y)
        }
        IExpr::Ge(a, b) => {
            let (x, y) = bin_int(a, b)?;
            Bool(x >= y)
        }
        IExpr::Not(a) => Bool(!as_bool(eval(a, cur, next)?)?),
        IExpr::And(a, b) => {
            let (x, y) = bin_bool(a, b)?;
            Bool(x && y)
        }
        IExpr::Or(a, b) => {
            let (x, y) = bin_bool(a, b)?;
            Bool(x || y)
        }
        IExpr::Implies(a, b) => {
            let (x, y) = bin_bool(a, b)?;
            Bool(!x || y)
        }
        IExpr::Iff(a, b) => {
            let (x, y) = bin_bool(a, b)?;
            Bool(x == y)
        }
    })
}

// ===========================================================================
// Public API: a parsed obligation source, evaluable at probe states.
// ===========================================================================

/// The spec's obligation predicates parsed by the INDEPENDENT front end.
pub(crate) struct IndepSpec {
    init: IExpr,
    next: IExpr,
    safety: IExpr,
    j: IExpr,
}

impl IndepSpec {
    /// Parse the obligation predicates independently of `tla_core`. `init_name`/
    /// `next_name` are operator names; `invariant_names` are conjoined into the
    /// safety predicate; `j_tla` is the invariant `J` as TLA text. Returns `None`
    /// if anything is outside the scalar fragment (fail-closed).
    pub(crate) fn parse(
        spec_src: &str,
        init_name: &str,
        next_name: &str,
        invariant_names: &[String],
        j_tla: &str,
    ) -> Option<IndepSpec> {
        let defs = extract_defs(spec_src);
        let init = parse_expr(defs.get(init_name)?, &defs)?;
        let next = parse_expr(defs.get(next_name)?, &defs)?;
        // safety = conjunction of the named invariants.
        let mut safety: Option<IExpr> = None;
        for name in invariant_names {
            let e = parse_expr(defs.get(name)?, &defs)?;
            safety = Some(match safety {
                None => e,
                Some(acc) => IExpr::And(Box::new(acc), Box::new(e)),
            });
        }
        let safety = safety?;
        let j = parse_expr(&tokenize(j_tla)?, &defs)?;
        Some(IndepSpec {
            init,
            next,
            safety,
            j,
        })
    }

    fn eval_bool(&self, e: &IExpr, cur: &IState, next: Option<&IState>) -> Option<bool> {
        match eval(e, cur, next)? {
            IVal::Bool(b) => Some(b),
            IVal::Int(_) => None,
        }
    }

    /// `Init(s0) /\ ~J(s0)`.
    pub(crate) fn initiation_truth(&self, s0: &IState) -> Option<bool> {
        Some(self.eval_bool(&self.init, s0, None)? && !self.eval_bool(&self.j, s0, None)?)
    }
    /// `J(s0) /\ ~Safety(s0)`.
    pub(crate) fn safety_truth(&self, s0: &IState) -> Option<bool> {
        Some(self.eval_bool(&self.j, s0, None)? && !self.eval_bool(&self.safety, s0, None)?)
    }
    /// `J(s0) /\ Next(s0,s1) /\ ~J(s1)`.
    pub(crate) fn consecution_truth(&self, s0: &IState, s1: &IState) -> Option<bool> {
        let j0 = self.eval_bool(&self.j, s0, None)?;
        let next = self.eval_bool(&self.next, s0, Some(s1))?;
        let j1 = self.eval_bool(&self.j, s1, None)?;
        Some(j0 && next && !j1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(pairs: &[(&str, i64)]) -> IState {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), IVal::Int(*v)))
            .collect()
    }

    const ACC: &str = "---- MODULE Acc ----\n\
                       EXTENDS Integers\n\
                       VARIABLE x\n\
                       Init == x = 0\n\
                       Next == x' = x + 1\n\
                       Safety == x >= 0\n\
                       ====\n";

    #[test]
    fn parses_and_evaluates_accumulator() {
        let spec = IndepSpec::parse(ACC, "Init", "Next", &["Safety".to_string()], "x >= 0")
            .expect("scalar fragment must parse");

        // Initiation = Init /\ ~J = (x=0) /\ (x<0): false everywhere.
        assert_eq!(spec.initiation_truth(&st(&[("x", 0)])), Some(false));
        assert_eq!(spec.initiation_truth(&st(&[("x", -1)])), Some(false));

        // Safety = J /\ ~Safety = (x>=0) /\ (x<0): false everywhere.
        assert_eq!(spec.safety_truth(&st(&[("x", 5)])), Some(false));

        // Consecution = J /\ Next /\ ~J' = (x>=0) /\ (x'=x+1) /\ (x'<0).
        // s0={x:0}, s1={x:1}: true /\ (1=0+1=true) /\ (1<0=false) -> false.
        assert_eq!(
            spec.consecution_truth(&st(&[("x", 0)]), &st(&[("x", 1)])),
            Some(false)
        );
        // s0={x:0}, s1={x:5}: Next false (5 != 1) -> whole thing false.
        assert_eq!(
            spec.consecution_truth(&st(&[("x", 0)]), &st(&[("x", 5)])),
            Some(false)
        );
    }

    #[test]
    fn out_of_fragment_is_none() {
        // A set construct must fail-closed (None), not mis-parse.
        let spec = "---- MODULE S ----\nVARIABLE x\nInit == x \\in {1,2,3}\n====\n";
        assert!(IndepSpec::parse(spec, "Init", "Init", &[], "x >= 0").is_none());
    }

    /// INDEPENDENCE GUARD: this module must import NOTHING from the production
    /// front end or translator — its whole purpose is to be a front-end-INDEPENDENT
    /// truth source. A future refactor that adds such an import would silently
    /// re-collapse the trust; this test fails closed on any `use` of the forbidden
    /// crates/paths. (Doc-comment mentions are fine; only `use` statements count.)
    #[test]
    fn test_indep_independence_grep() {
        let src = include_str!("cert_indep_frontend.rs");
        const FORBIDDEN: &[&str] = &[
            "tla_core",
            "tla_eval",
            "crate::eval",
            "BmcTranslator",
            "ay_bmc",
            "tla_ay",
            "negate_normalized",
        ];
        for line in src.lines() {
            let t = line.trim();
            if t.starts_with("use ") {
                for f in FORBIDDEN {
                    assert!(
                        !t.contains(f),
                        "independence violated: `{t}` imports forbidden `{f}`",
                    );
                }
            }
        }
    }

    #[test]
    fn evaluator_polarity_and_arithmetic() {
        // Direct checks of operator semantics independent of tla_core.
        let defs = HashMap::new();
        let e = parse_expr(&tokenize("x + 2 * 3 >= 7").unwrap(), &defs).unwrap();
        // x=1 -> 1+6=7 >= 7 true; x=0 -> 6>=7 false.
        assert_eq!(eval(&e, &st(&[("x", 1)]), None), Some(IVal::Bool(true)));
        assert_eq!(eval(&e, &st(&[("x", 0)]), None), Some(IVal::Bool(false)));
    }
}
