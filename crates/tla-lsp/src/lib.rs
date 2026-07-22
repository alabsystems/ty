// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! TY Language Server Protocol implementation for TLA+.
//!
//! This crate is the IDE-integration layer of the `ty` toolchain. It speaks the
//! [Language Server Protocol] over stdin/stdout (via [`tower_lsp`]) so that any
//! LSP-capable editor — VS Code, Neovim, Emacs, etc. — can offer TLA+ language
//! intelligence backed by `ty`'s own front end ([`tla_core`]) rather than the
//! Java SANY tooling.
//!
//! # What it provides
//!
//! - **Diagnostics** — parse, lowering, name-resolution, and lightweight
//!   semantic errors/warnings/hints, each tagged with a stable diagnostic code
//!   (see the `diagnostics` module).
//! - **Document symbols** — an outline of a module's CONSTANTs, VARIABLEs,
//!   operators, and theorems.
//! - **Go to definition** and **find references** — driven by the resolver's
//!   use→definition map.
//! - **Hover** — kind and signature information for the symbol under the cursor.
//! - **Completion** — TLA+ keywords, standard-library modules and their
//!   operators, and locally declared symbols, triggered on `\`, `.`, and `_`.
//! - **Workspace symbol search** — fuzzy lookup across all open documents.
//!
//! # How it works
//!
//! Each open document is parsed and analyzed on every change through the
//! full front-end pipeline — `parse` → `lower` → `resolve` → semantic
//! analysis — and the results are cached per URI. Text synchronization is
//! *full* (the whole buffer is re-sent and re-analyzed on each edit), keeping
//! the server stateless and simple at the cost of re-parsing on every keystroke.
//! All language features are answered from the cached analysis of the relevant
//! document.
//!
//! # Public API
//!
//! The surface is intentionally tiny. [`run_server`] is the entry point used by
//! the `ty` CLI; it constructs a [`TlaBackend`] and serves it over the process's
//! standard streams until the client disconnects. [`TlaBackend`] is exposed so
//! that embedders can host the language server over a transport other than
//! stdio.
//!
//! ```no_run
//! # async fn run() {
//! // Serve the TLA+ language server on this process's stdin/stdout.
//! tla_lsp::run_server().await;
//! # }
//! ```
//!
//! [Language Server Protocol]: https://microsoft.github.io/language-server-protocol/

mod analysis;
mod backend;
mod completions;
mod diagnostics;
mod document;
mod handlers;
mod position;
mod server;
mod symbols;

#[cfg(test)]
mod tests;

pub use backend::TlaBackend;
pub use server::run_server;
