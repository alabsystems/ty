// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::{Path, PathBuf};

use crate::hlpnml::ColoredNet;
use crate::nupn::NupnStructure;
use crate::petri_net::PetriNet;

use super::aliases::PropertyAliases;
use super::diagnostics::ColoredLoadDiagnostics;

/// The kind of source net that was loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SourceNetKind {
    /// Standard Place/Transition net (`ptnet` type attribute).
    Pt,
    /// Colored symmetric net (`symmetricnet` type attribute).
    SymmetricNet,
}

/// A model directory fully prepared for MCC examination execution.
///
/// Wraps a [`PetriNet`] with the model name, directory path, source
/// net kind, and property alias tables. Created by [`super::load_model_dir`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedModel {
    model_name: String,
    model_dir: PathBuf,
    source_kind: SourceNetKind,
    net: PetriNet,
    nupn: Option<NupnStructure>,
    pub(crate) aliases: PropertyAliases,
    /// Parsed colored source before query-independent load reductions.
    ///
    /// Colored UpperBounds, reachability, and OneSafe use this as the semantic
    /// baseline so original MCC identifiers resolve to every color instance
    /// even when the executable load-time net collapsed a place to Dot.
    pub(crate) colored_source: Option<ColoredNet>,
    colored_load_diagnostics: Option<ColoredLoadDiagnostics>,
    /// True when load-time unfolding aborted over budget (size cap or
    /// deadline) and `net` is therefore an empty PLACEHOLDER. Examinations
    /// that lack a colored-source-aware path MUST NOT run on this net; they
    /// emit CANNOT_COMPUTE instead. Only the colored OneSafe structural
    /// shortcut (which never touches `net`) and the re-unfolding paths
    /// (which re-abort to per-examination CC) are valid here.
    pub(crate) colored_unfold_unavailable: bool,
}

impl PreparedModel {
    pub(super) fn new(
        model_name: String,
        model_dir: PathBuf,
        source_kind: SourceNetKind,
        net: PetriNet,
        nupn: Option<NupnStructure>,
        aliases: PropertyAliases,
        colored_source: Option<ColoredNet>,
        colored_load_diagnostics: Option<ColoredLoadDiagnostics>,
    ) -> Self {
        Self {
            model_name,
            model_dir,
            source_kind,
            net,
            nupn,
            aliases,
            colored_source,
            colored_load_diagnostics,
            colored_unfold_unavailable: false,
        }
    }

    /// Mark this model's executable net as an over-budget placeholder.
    /// See [`PreparedModel::colored_unfold_unavailable`].
    pub(super) fn with_colored_unfold_unavailable(mut self) -> Self {
        self.colored_unfold_unavailable = true;
        self
    }

    /// Whether load-time unfolding aborted over budget (placeholder net).
    #[must_use]
    pub(crate) fn colored_unfold_unavailable(&self) -> bool {
        self.colored_unfold_unavailable
    }

    /// The model name (derived from the directory name).
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The model directory path.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// The kind of source net that was loaded.
    #[must_use]
    pub fn source_kind(&self) -> SourceNetKind {
        self.source_kind
    }

    /// The executable P/T net (after unfolding for colored nets).
    #[must_use]
    pub fn net(&self) -> &PetriNet {
        &self.net
    }

    /// Parsed NUPN metadata for P/T inputs, if present.
    #[must_use]
    pub fn nupn(&self) -> Option<&NupnStructure> {
        self.nupn.as_ref()
    }

    #[must_use]
    pub(crate) fn aliases(&self) -> &PropertyAliases {
        &self.aliases
    }

    #[must_use]
    pub(super) fn colored_load_diagnostics(&self) -> Option<&ColoredLoadDiagnostics> {
        self.colored_load_diagnostics.as_ref()
    }
}
