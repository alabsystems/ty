// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use dashmap::DashMap;
use tower_lsp::lsp_types::Url;
use tower_lsp::Client;

use crate::document::DocumentState;

/// TLA+ language server backend.
///
/// Implements [`tower_lsp::LanguageServer`] and holds the server's mutable
/// state: a handle to the LSP `Client` (used to push diagnostics and log
/// messages) and a per-URI cache of analyzed [open documents][crate]. The
/// cache is a [`DashMap`], so the request handlers — which `tower_lsp` may run
/// concurrently — can read and update document state without an outer lock.
///
/// One backend instance lives for the duration of a client session. Construct
/// it with [`TlaBackend::new`] (typically via [`crate::run_server`], which wires
/// it to the process's standard streams).
pub struct TlaBackend {
    /// LSP client handle, used to publish diagnostics and send notifications.
    pub(crate) client: Client,
    /// Cache of currently open documents, keyed by their URI. Populated on
    /// `didOpen`/`didChange` and removed on `didClose`.
    pub(crate) documents: DashMap<Url, DocumentState>,
}

impl TlaBackend {
    /// Create a backend bound to the given LSP `client`.
    ///
    /// The document cache starts empty; documents are added as the client sends
    /// `didOpen` notifications. The `client` handle is retained for the lifetime
    /// of the session so the backend can push diagnostics back to the editor.
    ///
    /// This matches the `tower_lsp::LspService::new` factory signature, so it is
    /// usually passed there by reference rather than called directly.
    pub fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
        }
    }
}
