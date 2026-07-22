// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! NUPN metadata parsing for MCC PNML files.
//!
//! MCC P/T models may include a `toolspecific tool="nupn"` section declaring
//! Nested-Unit Petri Net structure. The `safe="true"` pragma gives free
//! unit-safety facts that can be consumed by structural checks and future
//! compact encodings.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::PnmlError;
use crate::petri_net::{PetriNet, PlaceIdx};

/// Declared size metadata from a NUPN toolspecific block.
///
/// Each field is the declared count when the annotation provided it; `None`
/// when absent.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NupnSize {
    /// Declared number of places.
    pub places: Option<usize>,
    /// Declared number of transitions.
    pub transitions: Option<usize>,
    /// Declared number of arcs.
    pub arcs: Option<usize>,
}

/// One NUPN unit after resolving place and subunit references.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NupnUnit {
    id: String,
    places: Vec<PlaceIdx>,
    subunits: Vec<usize>,
}

impl NupnUnit {
    /// Unit identifier from the PNML annotation.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Local places directly owned by this unit.
    #[must_use]
    pub fn places(&self) -> &[PlaceIdx] {
        &self.places
    }

    /// Child unit indices into [`NupnStructure::units`].
    #[must_use]
    pub fn subunits(&self) -> &[usize] {
        &self.subunits
    }
}

/// Parsed NUPN structure for a P/T model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NupnStructure {
    unit_safe: bool,
    root: Option<usize>,
    units: Vec<NupnUnit>,
    size: Option<NupnSize>,
}

impl NupnStructure {
    /// Whether the annotation declares the net unit-safe.
    #[must_use]
    pub fn unit_safe(&self) -> bool {
        self.unit_safe
    }

    /// Root unit, if the annotation declared one.
    #[must_use]
    pub fn root_unit(&self) -> Option<&NupnUnit> {
        self.root.map(|idx| &self.units[idx])
    }

    /// All units in PNML order.
    #[must_use]
    pub fn units(&self) -> &[NupnUnit] {
        &self.units
    }

    /// Optional declared size metadata.
    #[must_use]
    pub fn size(&self) -> Option<&NupnSize> {
        self.size.as_ref()
    }

    /// Number of distinct P/T places mentioned by local unit place lists.
    #[must_use]
    pub fn covered_place_count(&self) -> usize {
        let mut covered = HashSet::new();
        for unit in &self.units {
            covered.extend(unit.places.iter().map(|place| place.0));
        }
        covered.len()
    }

    /// Whether every place in a net of `num_places` places has a local NUPN unit.
    #[must_use]
    pub fn covers_all_places(&self, num_places: usize) -> bool {
        if num_places == 0 {
            return true;
        }

        let mut covered = vec![false; num_places];
        for unit in &self.units {
            for place in &unit.places {
                let idx = place.0 as usize;
                if idx < num_places {
                    covered[idx] = true;
                }
            }
        }
        covered.into_iter().all(|is_covered| is_covered)
    }

    /// Per-place upper bounds implied by a unit-safe annotation.
    #[must_use]
    pub fn one_safe_place_bounds(&self, num_places: usize) -> Vec<Option<u64>> {
        if !self.unit_safe {
            return vec![None; num_places];
        }

        let mut bounds = vec![None; num_places];
        for unit in &self.units {
            for place in &unit.places {
                let idx = place.0 as usize;
                if idx < num_places {
                    bounds[idx] = Some(1);
                }
            }
        }
        bounds
    }

    /// Check that the initial marking satisfies every local unit-safe group.
    #[must_use]
    pub fn initial_marking_respects_unit_safety(&self, initial_marking: &[u64]) -> bool {
        if !self.unit_safe {
            return false;
        }

        self.units.iter().all(|unit| {
            let mut sum = 0u64;
            for place in &unit.places {
                let Some(tokens) = initial_marking.get(place.0 as usize) else {
                    return false;
                };
                let Some(next) = sum.checked_add(*tokens) else {
                    return false;
                };
                sum = next;
            }
            sum <= 1
        })
    }

    /// Whether this annotation is sufficient to prove individual P/T OneSafe.
    pub(crate) fn proves_individual_one_safe(&self, net: &PetriNet) -> bool {
        self.unit_safe
            && self.covers_all_places(net.num_places())
            && self.initial_marking_respects_unit_safety(&net.initial_marking)
    }
}

/// Parse NUPN metadata from a PNML file, if present.
pub(crate) fn parse_nupn_file(
    path: &Path,
    net: &PetriNet,
) -> Result<Option<NupnStructure>, PnmlError> {
    let content = std::fs::read_to_string(path).map_err(|e| PnmlError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    parse_nupn(&content, net)
}

/// Parse NUPN metadata from PNML XML content, if present.
pub(crate) fn parse_nupn(
    content: &str,
    net: &PetriNet,
) -> Result<Option<NupnStructure>, PnmlError> {
    let doc =
        roxmltree::Document::parse(content).map_err(|e| PnmlError::XmlParse(e.to_string()))?;

    let Some(toolspecific) = doc.descendants().find(|node| {
        node.is_element()
            && node.tag_name().name() == "toolspecific"
            && node
                .attribute("tool")
                .is_some_and(|tool| tool.eq_ignore_ascii_case("nupn"))
    }) else {
        return Ok(None);
    };

    let Some(structure) = toolspecific
        .children()
        .find(|node| node.is_element() && node.tag_name().name() == "structure")
    else {
        return Ok(None);
    };

    let place_by_id: HashMap<&str, PlaceIdx> = net
        .places
        .iter()
        .enumerate()
        .map(|(idx, place)| (place.id.as_str(), PlaceIdx(idx as u32)))
        .collect();

    let unit_nodes: Vec<_> = structure
        .children()
        .filter(|node| node.is_element() && node.tag_name().name() == "unit")
        .collect();
    let mut unit_by_id = HashMap::with_capacity(unit_nodes.len());
    for (idx, unit) in unit_nodes.iter().enumerate() {
        let id = unit
            .attribute("id")
            .ok_or_else(|| invalid_nupn("unit without id"))?;
        if unit_by_id.insert(id.to_string(), idx).is_some() {
            return Err(invalid_nupn(format!("duplicate unit id `{id}`")));
        }
    }

    let mut place_owner = vec![None::<String>; net.num_places()];
    let mut units = Vec::with_capacity(unit_nodes.len());
    for unit in &unit_nodes {
        let id = unit
            .attribute("id")
            .expect("unit id collected in first pass")
            .to_string();
        let places = parse_place_list(unit, &place_by_id, &id, &mut place_owner)?;
        let subunits = parse_subunit_list(unit, &unit_by_id, &id)?;
        units.push(NupnUnit {
            id,
            places,
            subunits,
        });
    }

    let root = match structure.attribute("root") {
        Some(root_id) => Some(
            unit_by_id
                .get(root_id)
                .copied()
                .ok_or_else(|| invalid_nupn(format!("root unit `{root_id}` is not declared")))?,
        ),
        None => None,
    };

    Ok(Some(NupnStructure {
        unit_safe: structure
            .attribute("safe")
            .is_some_and(|safe| safe.eq_ignore_ascii_case("true")),
        root,
        units,
        size: parse_size(&toolspecific),
    }))
}

fn parse_place_list(
    unit: &roxmltree::Node<'_, '_>,
    place_by_id: &HashMap<&str, PlaceIdx>,
    unit_id: &str,
    place_owner: &mut [Option<String>],
) -> Result<Vec<PlaceIdx>, PnmlError> {
    let Some(places_node) = child(unit, "places") else {
        return Ok(Vec::new());
    };
    let mut seen_in_unit = HashSet::new();
    let mut places = Vec::new();
    for place_id in split_node_text(&places_node) {
        let place = place_by_id.get(place_id).copied().ok_or_else(|| {
            invalid_nupn(format!(
                "unit `{unit_id}` references unknown place `{place_id}`"
            ))
        })?;
        if !seen_in_unit.insert(place.0) {
            return Err(invalid_nupn(format!(
                "unit `{unit_id}` repeats place `{place_id}`"
            )));
        }

        let owner_slot = &mut place_owner[place.0 as usize];
        if let Some(owner) = owner_slot {
            return Err(invalid_nupn(format!(
                "place `{place_id}` is listed in both unit `{owner}` and unit `{unit_id}`"
            )));
        }
        *owner_slot = Some(unit_id.to_string());
        places.push(place);
    }
    Ok(places)
}

fn parse_subunit_list(
    unit: &roxmltree::Node<'_, '_>,
    unit_by_id: &HashMap<String, usize>,
    unit_id: &str,
) -> Result<Vec<usize>, PnmlError> {
    let Some(subunits_node) = child(unit, "subunits") else {
        return Ok(Vec::new());
    };
    let mut seen = HashSet::new();
    let mut subunits = Vec::new();
    for subunit_id in split_node_text(&subunits_node) {
        let subunit = unit_by_id.get(subunit_id).copied().ok_or_else(|| {
            invalid_nupn(format!(
                "unit `{unit_id}` references unknown subunit `{subunit_id}`"
            ))
        })?;
        if !seen.insert(subunit) {
            return Err(invalid_nupn(format!(
                "unit `{unit_id}` repeats subunit `{subunit_id}`"
            )));
        }
        subunits.push(subunit);
    }
    Ok(subunits)
}

fn child<'a>(node: &roxmltree::Node<'a, 'a>, tag: &str) -> Option<roxmltree::Node<'a, 'a>> {
    node.children()
        .find(|child| child.is_element() && child.tag_name().name() == tag)
}

fn split_node_text<'a>(node: &'a roxmltree::Node<'_, '_>) -> impl Iterator<Item = &'a str> {
    node.text()
        .unwrap_or("")
        .split_whitespace()
        .filter(|item| !item.is_empty())
}

fn parse_size(toolspecific: &roxmltree::Node<'_, '_>) -> Option<NupnSize> {
    let size = child(toolspecific, "size")?;
    Some(NupnSize {
        places: parse_usize_attr(&size, "places"),
        transitions: parse_usize_attr(&size, "transitions"),
        arcs: parse_usize_attr(&size, "arcs"),
    })
}

fn parse_usize_attr(node: &roxmltree::Node<'_, '_>, attr: &str) -> Option<usize> {
    node.attribute(attr)
        .and_then(|value| value.parse::<usize>().ok())
}

fn invalid_nupn(reason: impl Into<String>) -> PnmlError {
    PnmlError::InvalidNupn {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::{Arc, PlaceInfo, TransitionInfo};

    fn net() -> PetriNet {
        PetriNet {
            name: Some("nupn-test".to_string()),
            places: vec![
                PlaceInfo {
                    id: "P0".to_string(),
                    name: None,
                },
                PlaceInfo {
                    id: "P1".to_string(),
                    name: None,
                },
            ],
            transitions: vec![TransitionInfo {
                id: "T0".to_string(),
                name: None,
                inputs: vec![Arc {
                    place: PlaceIdx(0),
                    weight: 1,
                }],
                outputs: vec![Arc {
                    place: PlaceIdx(1),
                    weight: 1,
                }],
            }],
            initial_marking: vec![1, 0],
        }
    }

    const NUPN_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="nupn-test" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="page0">
      <place id="P0"/>
      <place id="P1"/>
      <transition id="T0"/>
      <toolspecific tool="nupn" version="1.1">
        <size places="2" transitions="1" arcs="2"/>
        <structure units="2" root="u0" safe="true">
          <unit id="u0">
            <places/>
            <subunits>u1</subunits>
          </unit>
          <unit id="u1">
            <places>P0 P1</places>
            <subunits/>
          </unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;

    #[test]
    fn test_parse_nupn_unit_safe_metadata() {
        let net = net();
        let nupn = parse_nupn(NUPN_PNML, &net)
            .expect("valid NUPN should parse")
            .expect("NUPN should be present");

        assert!(nupn.unit_safe());
        assert_eq!(nupn.root_unit().map(NupnUnit::id), Some("u0"));
        assert_eq!(nupn.units().len(), 2);
        assert_eq!(nupn.units()[1].places(), &[PlaceIdx(0), PlaceIdx(1)]);
        assert_eq!(nupn.covered_place_count(), 2);
        assert!(nupn.covers_all_places(net.num_places()));
        assert_eq!(
            nupn.one_safe_place_bounds(net.num_places()),
            vec![Some(1), Some(1)]
        );
        assert!(nupn.proves_individual_one_safe(&net));
    }

    #[test]
    fn test_parse_nupn_rejects_unknown_place_reference() {
        let net = net();
        let bad = NUPN_PNML.replace("P0 P1", "P0 Missing");
        let error = parse_nupn(&bad, &net).expect_err("unknown place should fail");
        assert!(
            matches!(error, PnmlError::InvalidNupn { .. }),
            "expected InvalidNupn, got {error:?}"
        );
    }
}
