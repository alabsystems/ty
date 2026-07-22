// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! MCC property XML parser.
//!
//! Parses `<Examination>.xml` files from MCC model directories.
//! Each file contains a `<property-set>` with multiple `<property>` elements,
//! each having an `<id>` and a `<formula>`.
//!
//! Supports:
//! - `<place-bound>`: list of places for UpperBounds examination
//! - `<exists-path><finally>` / `<all-paths><globally>`: Reachability path
//!   quantifiers wrapping boolean/integer/fireability state predicates

pub(crate) mod ast;
mod ctl;
mod ltl;
mod parse_common;
mod reachability;

#[cfg(test)]
mod tests;

use std::path::Path;

use crate::error::PnmlError;

pub(crate) use ast::{
    CtlFormula, Formula, IntExpr, LtlFormula, PathQuantifier, Property, ReachabilityFormula,
    StatePredicate,
};

#[derive(Debug, Clone, Copy)]
enum FormulaKind {
    Ctl,
    Ltl,
}

/// Parse an examination XML file from a model directory.
///
/// Reads `<model_dir>/<examination>.xml` and extracts all properties.
/// Dispatches to CTL/LTL-specific parsers based on examination name.
pub(crate) fn parse_properties(
    model_dir: &Path,
    examination: &str,
) -> Result<Vec<Property>, PnmlError> {
    let xml_path = model_dir.join(format!("{examination}.xml"));
    let content = std::fs::read_to_string(&xml_path).map_err(|e| PnmlError::Io {
        path: xml_path.clone(),
        source: e,
    })?;
    match examination {
        "CTLCardinality" | "CTLFireability" => parse_properties_typed(&content, FormulaKind::Ctl),
        "LTLCardinality" | "LTLFireability" => parse_properties_typed(&content, FormulaKind::Ltl),
        _ => parse_properties_xml(&content),
    }
}

/// Parse only MCC property identifiers from an examination XML file.
///
/// Fallback/error paths use this when formula parsing fails but the XML still
/// contains enough structure to emit one `CANNOT_COMPUTE` line per property.
pub(crate) fn parse_property_ids(
    model_dir: &Path,
    examination: &str,
) -> Result<Vec<String>, PnmlError> {
    let xml_path = model_dir.join(format!("{examination}.xml"));
    let content = std::fs::read_to_string(&xml_path).map_err(|e| PnmlError::Io {
        path: xml_path.clone(),
        source: e,
    })?;
    let doc =
        roxmltree::Document::parse(&content).map_err(|e| PnmlError::XmlParse(e.to_string()))?;
    parse_property_ids_xml(doc.root_element())
}

/// Parse property XML content string.
pub(crate) fn parse_properties_xml(content: &str) -> Result<Vec<Property>, PnmlError> {
    let doc =
        roxmltree::Document::parse(content).map_err(|e| PnmlError::XmlParse(e.to_string()))?;
    parse_common::parse_property_set(doc.root_element(), reachability::parse_root_formula)
}

/// Parse property XML with CTL or LTL formula interpretation.
fn parse_properties_typed(content: &str, kind: FormulaKind) -> Result<Vec<Property>, PnmlError> {
    let doc =
        roxmltree::Document::parse(content).map_err(|e| PnmlError::XmlParse(e.to_string()))?;
    match kind {
        FormulaKind::Ctl => {
            parse_common::parse_property_set(doc.root_element(), ctl::parse_root_formula)
        }
        FormulaKind::Ltl => {
            parse_common::parse_property_set(doc.root_element(), ltl::parse_root_formula)
        }
    }
}

fn parse_property_ids_xml(root: roxmltree::Node<'_, '_>) -> Result<Vec<String>, PnmlError> {
    if root.tag_name().name() != "property-set" {
        return Err(PnmlError::MissingElement(format!(
            "expected <property-set> root, got <{}>",
            root.tag_name().name()
        )));
    }

    let mut ids = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for prop_node in root.children().filter(|node| node.has_tag_name("property")) {
        let id_node = parse_common::only_named_child(&prop_node, "id", "property")?;
        let id = parse_common::parse_property_id(&id_node, &mut seen)?;
        ids.push(id);
    }

    if ids.is_empty() {
        return Err(PnmlError::MissingElement(
            "property-set: missing <property>".into(),
        ));
    }

    Ok(ids)
}
