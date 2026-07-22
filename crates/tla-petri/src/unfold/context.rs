// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sort/value algebra and shared lookup context for colored-net unfolding.

use std::collections::HashMap;

use crate::error::PnmlError;
use crate::hlpnml::{ColorSort, ColorTerm, ColoredNet};

use super::ColorValue;

/// Lookup context for sort definitions and variable-to-sort mappings.
pub(super) struct UnfoldContext {
    /// Sort id → sort definition.
    pub(super) sorts: HashMap<String, ColorSort>,
    /// Variable id → sort id.
    pub(super) var_sorts: HashMap<String, String>,
}

impl UnfoldContext {
    pub(super) fn new(colored: &ColoredNet) -> Result<Self, PnmlError> {
        let mut sorts = HashMap::new();
        for sort in &colored.sorts {
            sorts.insert(sort.id().to_string(), sort.clone());
        }

        let mut var_sorts = HashMap::new();
        for var in &colored.variables {
            var_sorts.insert(var.id.clone(), var.sort_id.clone());
        }

        Ok(UnfoldContext { sorts, var_sorts })
    }

    /// Get the sort for a colored place.
    pub(super) fn sort_for_place(
        &self,
        place: &crate::hlpnml::ColoredPlace,
    ) -> Result<&ColorSort, PnmlError> {
        self.sorts.get(&place.sort_id).ok_or_else(|| {
            PnmlError::MissingElement(format!("sort '{}' for place '{}'", place.sort_id, place.id))
        })
    }

    /// Get the sort for a variable.
    pub(super) fn sort_for_variable(&self, var_id: &str) -> Result<&ColorSort, PnmlError> {
        let sort_id = self.var_sorts.get(var_id).ok_or_else(|| {
            PnmlError::MissingElement(format!("variable '{var_id}' not declared"))
        })?;
        self.sorts.get(sort_id).ok_or_else(|| {
            PnmlError::MissingElement(format!("sort '{sort_id}' for variable '{var_id}'"))
        })
    }

    /// Validate an `<all>` term against the place sort it contributes to.
    pub(super) fn validate_all_sort_for_target(
        &self,
        sort_id: &str,
        target_sort: &ColorSort,
    ) -> Result<(), PnmlError> {
        if sort_id.is_empty() {
            return Ok(());
        }
        if !self.sorts.contains_key(sort_id) {
            return Err(PnmlError::MissingElement(format!(
                "sort '{sort_id}' not found"
            )));
        }
        if sort_id != target_sort.id() {
            return Err(PnmlError::InvalidMarking(format!(
                "all sort '{sort_id}' does not match target place sort '{}'",
                target_sort.id()
            )));
        }
        Ok(())
    }

    /// Compute cardinality of a sort.
    pub(super) fn sort_cardinality(&self, sort: &ColorSort) -> Result<usize, PnmlError> {
        match sort {
            ColorSort::Dot { .. } => Ok(1),
            ColorSort::CyclicEnum { constants, .. } => Ok(constants.len()),
            ColorSort::FiniteIntRange { start, end, .. } => {
                if end < start {
                    return Err(PnmlError::UnsupportedNetType {
                        net_type: format!(
                            "finiteintrange '{}' has end ({end}) < start ({start})",
                            sort.name()
                        ),
                    });
                }
                let span = (*end as i128) - (*start as i128) + 1;
                usize::try_from(span).map_err(|_| PnmlError::UnsupportedNetType {
                    net_type: format!(
                        "finiteintrange '{}' cardinality {span} exceeds usize",
                        sort.name()
                    ),
                })
            }
            ColorSort::Product { components, .. } => {
                let mut cardinality = 1usize;
                for component_id in components {
                    let component_sort = self.sorts.get(component_id).ok_or_else(|| {
                        PnmlError::MissingElement(format!(
                            "component sort '{component_id}' for product sort '{}'",
                            sort.name()
                        ))
                    })?;
                    cardinality = cardinality
                        .checked_mul(self.sort_cardinality(component_sort)?)
                        .ok_or_else(|| PnmlError::UnsupportedNetType {
                            net_type: format!(
                                "product sort '{}' cardinality overflow",
                                sort.name()
                            ),
                        })?;
                }
                Ok(cardinality)
            }
        }
    }

    pub(super) fn sort_value_names(&self, sort: &ColorSort) -> Result<Vec<String>, PnmlError> {
        let cardinality = self.sort_cardinality(sort)?;
        (0..cardinality)
            .map(|value| self.sort_value_name(sort, value))
            .collect()
    }

    pub(super) fn sort_value_name(
        &self,
        sort: &ColorSort,
        value: ColorValue,
    ) -> Result<String, PnmlError> {
        match sort {
            ColorSort::Dot { .. } => Ok("dot".to_string()),
            ColorSort::CyclicEnum { constants, .. } => constants
                .get(value)
                .map(|constant| constant.name.clone())
                .ok_or_else(|| {
                    PnmlError::InvalidMarking(format!(
                        "color value {value} out of range for sort '{}'",
                        sort.name()
                    ))
                }),
            ColorSort::FiniteIntRange { start, end, .. } => {
                let span = (*end as i128) - (*start as i128) + 1;
                if (value as i128) >= span {
                    return Err(PnmlError::InvalidMarking(format!(
                        "color value {value} out of range for sort '{}'",
                        sort.name()
                    )));
                }
                Ok(((*start as i128) + value as i128).to_string())
            }
            ColorSort::Product { components, .. } => {
                let component_values = self.unflatten_product_value(components, value)?;
                let mut names = Vec::with_capacity(components.len());
                for (component_id, component_value) in components.iter().zip(component_values) {
                    let component_sort = self.sorts.get(component_id).ok_or_else(|| {
                        PnmlError::MissingElement(format!(
                            "component sort '{component_id}' for product sort '{}'",
                            sort.name()
                        ))
                    })?;
                    names.push(self.sort_value_name(component_sort, component_value)?);
                }
                Ok(names.join("_"))
            }
        }
    }

    pub(super) fn flatten_product_value(
        &self,
        component_sort_ids: &[String],
        component_values: &[ColorValue],
    ) -> Result<ColorValue, PnmlError> {
        if component_sort_ids.len() != component_values.len() {
            return Err(PnmlError::InvalidMarking(format!(
                "tuple arity {} does not match product sort arity {}",
                component_values.len(),
                component_sort_ids.len()
            )));
        }

        let mut value = 0usize;
        for (component_sort_id, component_value) in component_sort_ids.iter().zip(component_values)
        {
            let component_sort = self.sorts.get(component_sort_id).ok_or_else(|| {
                PnmlError::MissingElement(format!(
                    "component sort '{component_sort_id}' for product value"
                ))
            })?;
            let radix = self.sort_cardinality(component_sort)?;
            if *component_value >= radix {
                return Err(PnmlError::InvalidMarking(format!(
                    "component value {} out of range for sort '{component_sort_id}'",
                    component_value
                )));
            }
            value = value
                .checked_mul(radix)
                .and_then(|prefix| prefix.checked_add(*component_value))
                .ok_or_else(|| PnmlError::UnsupportedNetType {
                    net_type: format!("product sort '{}' flattening overflow", component_sort_id),
                })?;
        }
        Ok(value)
    }

    pub(super) fn unflatten_product_value(
        &self,
        component_sort_ids: &[String],
        value: ColorValue,
    ) -> Result<Vec<ColorValue>, PnmlError> {
        let mut radices = Vec::with_capacity(component_sort_ids.len());
        let mut total = 1usize;
        for component_sort_id in component_sort_ids {
            let component_sort = self.sorts.get(component_sort_id).ok_or_else(|| {
                PnmlError::MissingElement(format!(
                    "component sort '{component_sort_id}' for product value"
                ))
            })?;
            let radix = self.sort_cardinality(component_sort)?;
            radices.push(radix);
            total = total
                .checked_mul(radix)
                .ok_or_else(|| PnmlError::UnsupportedNetType {
                    net_type: format!("product sort '{}' cardinality overflow", component_sort_id),
                })?;
        }

        if value >= total {
            return Err(PnmlError::InvalidMarking(format!(
                "product value {value} out of range for cardinality {total}"
            )));
        }

        let mut remaining = value;
        let mut component_values = vec![0usize; component_sort_ids.len()];
        for index in (0..radices.len()).rev() {
            let radix = radices[index];
            component_values[index] = remaining % radix;
            remaining /= radix;
        }
        Ok(component_values)
    }

    /// Find the index of a named constant in a sort.
    ///
    /// Dot sort returns `None` because the dot constant is represented by
    /// `ColorTerm::DotConstant`, not `ColorTerm::UserConstant`. Returning
    /// `Some(0)` here would poison all UserConstant lookups when HashMap
    /// iteration visits the Dot sort first (since any constant_id would
    /// match), causing incorrect guard evaluation.
    pub(super) fn constant_index(&self, sort: &ColorSort, constant_id: &str) -> Option<usize> {
        match sort {
            ColorSort::Dot { .. } => None,
            ColorSort::CyclicEnum { constants, .. } => {
                constants.iter().position(|c| c.id == constant_id)
            }
            // FiniteIntRange values are bare integers (e.g. "0", "1", "2",
            // …) and may appear as useroperator declarations referencing the
            // literal value string. Match the integer-as-string against the
            // index within `[start, end]`.
            ColorSort::FiniteIntRange { start, end, .. } => {
                let parsed: i64 = constant_id.parse().ok()?;
                if parsed < *start || parsed > *end {
                    return None;
                }
                Some((parsed - *start) as usize)
            }
            ColorSort::Product { .. } => None,
        }
    }

    /// Evaluate a color term to a concrete color value under a binding.
    pub(super) fn eval_color_value(
        &self,
        term: &ColorTerm,
        binding: &HashMap<&str, ColorValue>,
    ) -> Option<ColorValue> {
        match term {
            ColorTerm::Variable(var_id) => binding.get(var_id.as_str()).copied(),
            ColorTerm::Tuple(_) => None,
            ColorTerm::UserConstant(decl_id) => {
                for sort in self.sorts.values() {
                    if let Some(idx) = self.constant_index(sort, decl_id) {
                        return Some(idx);
                    }
                }
                None
            }
            ColorTerm::IntegerConstant(_) => None,
            ColorTerm::All(_) => None,
            ColorTerm::DotConstant => Some(0),
            ColorTerm::Predecessor(inner) => {
                let val = self.eval_color_value(inner, binding)?;
                let sort = self.sort_for_term(inner)?;
                let card = self.sort_cardinality(sort).ok()?;
                Some(if val == 0 { card - 1 } else { val - 1 })
            }
            ColorTerm::Successor(inner) => {
                let val = self.eval_color_value(inner, binding)?;
                let sort = self.sort_for_term(inner)?;
                let card = self.sort_cardinality(sort).ok()?;
                Some((val + 1) % card)
            }
        }
    }

    pub(super) fn eval_color_value_for_sort(
        &self,
        term: &ColorTerm,
        binding: &HashMap<&str, ColorValue>,
        sort: &ColorSort,
    ) -> Result<Option<ColorValue>, PnmlError> {
        match (sort, term) {
            (ColorSort::FiniteIntRange { start, end, .. }, ColorTerm::IntegerConstant(value)) => {
                if value < start || value > end {
                    return Err(PnmlError::InvalidMarking(format!(
                        "integer color constant {value} out of range {start}..={end}"
                    )));
                }
                usize::try_from((*value as i128) - (*start as i128))
                    .map(Some)
                    .map_err(|_| PnmlError::UnsupportedNetType {
                        net_type: format!(
                            "integer color constant {value} cannot be represented as usize"
                        ),
                    })
            }
            (_, ColorTerm::All(_)) => Ok(None),
            (ColorSort::Product { components, .. }, ColorTerm::Tuple(component_terms)) => {
                if component_terms.len() != components.len() {
                    return Err(PnmlError::InvalidMarking(format!(
                        "tuple arity {} does not match product sort arity {}",
                        component_terms.len(),
                        components.len()
                    )));
                }

                let mut component_values = Vec::with_capacity(component_terms.len());
                for (component_term, component_sort_id) in
                    component_terms.iter().zip(components.iter())
                {
                    let component_sort = self.sorts.get(component_sort_id).ok_or_else(|| {
                        PnmlError::MissingElement(format!(
                            "component sort '{component_sort_id}' for tuple"
                        ))
                    })?;
                    let component_value = self
                        .eval_color_value_for_sort(component_term, binding, component_sort)?
                        .ok_or_else(|| {
                            PnmlError::InvalidMarking(format!(
                                "tuple component for sort '{component_sort_id}' did not resolve"
                            ))
                        })?;
                    component_values.push(component_value);
                }

                Ok(Some(
                    self.flatten_product_value(components, &component_values)?,
                ))
            }
            (ColorSort::Product { .. }, ColorTerm::Variable(var_id)) => {
                Ok(binding.get(var_id.as_str()).copied())
            }
            (ColorSort::Product { .. }, ColorTerm::Predecessor(_) | ColorTerm::Successor(_)) => {
                Ok(None)
            }
            // A single-component tuple over a non-product (scalar) sort. The
            // GreatSPN editor wraps every arc inscription in `<tuple>`, even for
            // a scalar-sorted place (e.g. `<a>` on a `finiteintrange` place is
            // emitted as a 1-tuple). Treat it as the bare inner term so the arc
            // is not silently dropped. Without this, the `_` arm below falls
            // through to `eval_color_value`, which returns `None` for any
            // `Tuple`, dropping the arc and corrupting the unfolded net (a place
            // becomes spuriously isolated / constant).
            (sort, ColorTerm::Tuple(component_terms))
                if !matches!(sort, ColorSort::Product { .. }) && component_terms.len() == 1 =>
            {
                self.eval_color_value_for_sort(&component_terms[0], binding, sort)
            }
            (_, ColorTerm::Predecessor(inner)) => {
                let val = self
                    .eval_color_value_for_sort(inner, binding, sort)?
                    .ok_or_else(|| {
                        PnmlError::InvalidMarking(String::from(
                            "predecessor subterm did not resolve",
                        ))
                    })?;
                let card = self.sort_cardinality(sort)?;
                Ok(Some(if val == 0 { card - 1 } else { val - 1 }))
            }
            (_, ColorTerm::Successor(inner)) => {
                let val = self
                    .eval_color_value_for_sort(inner, binding, sort)?
                    .ok_or_else(|| {
                        PnmlError::InvalidMarking(String::from("successor subterm did not resolve"))
                    })?;
                let card = self.sort_cardinality(sort)?;
                Ok(Some((val + 1) % card))
            }
            _ => Ok(self.eval_color_value(term, binding)),
        }
    }

    pub(super) fn eval_color_values_for_sort(
        &self,
        term: &ColorTerm,
        binding: &HashMap<&str, ColorValue>,
        sort: &ColorSort,
    ) -> Result<Vec<ColorValue>, PnmlError> {
        match (sort, term) {
            (_, ColorTerm::All(sort_id)) => {
                self.validate_all_sort_for_target(sort_id, sort)?;
                Ok((0..self.sort_cardinality(sort)?).collect())
            }
            (ColorSort::Product { components, .. }, ColorTerm::Tuple(component_terms)) => {
                if component_terms.len() != components.len() {
                    return Err(PnmlError::InvalidMarking(format!(
                        "tuple arity {} does not match product sort arity {}",
                        component_terms.len(),
                        components.len()
                    )));
                }

                let mut component_values = Vec::with_capacity(component_terms.len());
                for (component_term, component_sort_id) in
                    component_terms.iter().zip(components.iter())
                {
                    let component_sort = self.sorts.get(component_sort_id).ok_or_else(|| {
                        PnmlError::MissingElement(format!(
                            "component sort '{component_sort_id}' for tuple"
                        ))
                    })?;
                    let values =
                        self.eval_color_values_for_sort(component_term, binding, component_sort)?;
                    if values.is_empty() {
                        return Ok(vec![]);
                    }
                    component_values.push(values);
                }

                let mut values = Vec::new();
                let mut current = Vec::with_capacity(component_values.len());
                self.expand_product_values(
                    components,
                    &component_values,
                    0,
                    &mut current,
                    &mut values,
                )?;
                Ok(values)
            }
            // Single-component tuple over a non-product (scalar) sort: unwrap
            // and evaluate the inner term, preserving multi-valued results such
            // as `<all>`. See the matching note in `eval_color_value_for_sort`.
            (sort, ColorTerm::Tuple(component_terms))
                if !matches!(sort, ColorSort::Product { .. }) && component_terms.len() == 1 =>
            {
                self.eval_color_values_for_sort(&component_terms[0], binding, sort)
            }
            _ => Ok(self
                .eval_color_value_for_sort(term, binding, sort)?
                .into_iter()
                .collect()),
        }
    }

    fn expand_product_values(
        &self,
        component_sort_ids: &[String],
        component_values: &[Vec<ColorValue>],
        index: usize,
        current: &mut Vec<ColorValue>,
        values: &mut Vec<ColorValue>,
    ) -> Result<(), PnmlError> {
        if index == component_values.len() {
            return self
                .flatten_product_value(component_sort_ids, current)
                .map(|value| values.push(value));
        }

        for value in &component_values[index] {
            current.push(*value);
            self.expand_product_values(
                component_sort_ids,
                component_values,
                index + 1,
                current,
                values,
            )?;
            current.pop();
        }
        Ok(())
    }

    /// Determine the sort of a color term (for predecessor/successor wraparound).
    pub(super) fn sort_for_term(&self, term: &ColorTerm) -> Option<&ColorSort> {
        match term {
            ColorTerm::Variable(var_id) => self.sort_for_variable(var_id).ok(),
            ColorTerm::UserConstant(decl_id) => {
                for sort in self.sorts.values() {
                    if self.constant_index(sort, decl_id).is_some() {
                        return Some(sort);
                    }
                }
                None
            }
            ColorTerm::DotConstant => self
                .sorts
                .values()
                .find(|s| matches!(s, ColorSort::Dot { .. })),
            ColorTerm::IntegerConstant(_) => None,
            ColorTerm::All(sort_id) => self.sorts.get(sort_id),
            ColorTerm::Tuple(_) => None,
            ColorTerm::Predecessor(inner) | ColorTerm::Successor(inner) => {
                self.sort_for_term(inner)
            }
        }
    }
}
