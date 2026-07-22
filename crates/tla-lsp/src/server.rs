// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::backend::TlaBackend;

/// Run the TLA+ language server on this process's standard streams.
///
/// Builds a [`TlaBackend`] wired to a `tower_lsp` service and serves LSP
/// requests over stdin/stdout, using the standard `Content-Length`-framed
/// JSON-RPC transport that editors speak. This is the entry point invoked by
/// the `ty` CLI's language-server subcommand.
///
/// The returned future resolves when the client closes the connection (e.g.
/// after a `shutdown`/`exit` exchange) or when stdin reaches EOF. Because it
/// owns the process's stdin and stdout, only one server should run per process.
///
/// ```no_run
/// # async fn run() {
/// tla_lsp::run_server().await;
/// # }
/// ```
pub async fn run_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = tower_lsp::LspService::new(TlaBackend::new);
    tower_lsp::Server::new(stdin, stdout, socket)
        .serve(service)
        .await;
}
