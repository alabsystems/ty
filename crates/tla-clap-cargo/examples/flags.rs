// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use clap::Parser;

#[derive(Debug, Parser)]
struct Cli {
    #[clap(flatten)]
    manifest: clap_cargo::Manifest,
    #[clap(flatten)]
    workspace: clap_cargo::Workspace,
    #[clap(flatten)]
    features: clap_cargo::Features,
}

fn main() {
    let args = Cli::parse();
    println!("args = {args:#?}");
}
