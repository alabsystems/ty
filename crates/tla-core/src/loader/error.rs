// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error types and data structures for the module loader.

use std::path::PathBuf;

use crate::ast::Module;
use crate::span::FileId;
use crate::syntax::SyntaxNode;

/// Error during module loading
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LoadError {
    /// Module file not found
    NotFound {
        /// Name of the module that could not be located.
        module: String,
        /// Directories searched while looking for the module file.
        search_paths: Vec<PathBuf>,
    },
    /// IO error reading file
    IoError {
        /// Path that failed to read.
        path: PathBuf,
        /// The underlying I/O error message.
        message: String,
    },
    /// Parse error in module
    ParseError {
        /// Path of the module that failed to parse.
        path: PathBuf,
        /// Rendered parse error messages.
        errors: Vec<String>,
    },
    /// Lower error in module
    LowerError {
        /// Path of the module that failed to lower (CST -> AST).
        path: PathBuf,
        /// Rendered lowering error messages.
        errors: Vec<String>,
    },
    /// Circular dependency detected
    CircularDependency {
        /// The dependency cycle, as module names in import order.
        chain: Vec<String>,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::NotFound {
                module,
                search_paths,
            } => {
                write!(f, "Module '{module}' not found. Searched in:")?;
                for path in search_paths {
                    write!(f, "\n  - {}", path.display())?;
                }
                Ok(())
            }
            LoadError::IoError { path, message } => {
                write!(f, "Error reading {}: {}", path.display(), message)
            }
            LoadError::ParseError { path, errors } => {
                write!(f, "Parse errors in {}:", path.display())?;
                for err in errors {
                    write!(f, "\n  {err}")?;
                }
                Ok(())
            }
            LoadError::LowerError { path, errors } => {
                write!(f, "Lower errors in {}:", path.display())?;
                for err in errors {
                    write!(f, "\n  {err}")?;
                }
                Ok(())
            }
            LoadError::CircularDependency { chain } => {
                write!(
                    f,
                    "Circular module dependency detected: {}",
                    chain.join(" -> ")
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Loaded module with metadata
#[derive(Debug, Clone)]
pub struct LoadedModule {
    /// The parsed and lowered module
    pub module: Module,
    /// The syntax tree (for SPECIFICATION resolution)
    pub syntax_tree: SyntaxNode,
    /// Path to the source file
    pub path: PathBuf,
    /// File ID for span tracking
    pub file_id: FileId,
}
