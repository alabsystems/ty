// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty install-tlc proof-library`: install the **upstream** TLA+ proof-system
//! module library (`tlaplus/tlapm`'s `library/`) so TLC can parse the corpus
//! specs that `EXTENDS TLAPS` / `FiniteSetTheorems` / `NaturalsInduction`.
//!
//! # Why this exists
//!
//! 25 of the 141 eligible rows in
//! `tests/tlc_comparison/strict_corpus_manifest.json` do not parse under the
//! declared TLC toolchain (`tytools.jar` + `CommunityModules.jar` + a JDK).
//! Running SANY across the eligible corpus gives exactly one cause for all 25:
//!
//! | missing module | rows |
//! |---|---|
//! | `TLAPS` | MCBakery, MCBoulanger, MCFindHighest, MCBinarySearch, MCQuicksort, MCConsensus, MCPaxos, MCVoting, MCConsensus_2, MCPaxosSmall, MCPaxosTiny, MCVoting_2, Simple, SimpleRegular, MCTwoPhase, Barriers, PConProof, Lock, Peterson |
//! | `FiniteSetTheorems` | bcastByz, bcastByzNoBcast, BPConProof, Consensus |
//! | `NaturalsInduction` | VoteProof, LockHS |
//!
//! Historically the comparison harness closed that gap by injecting the repo's
//! own `test_specs/tla_library` (a first-party 77-line `TLAPS.tla` stub that
//! defines every proof pragma as `TRUE`, plus companions) into BOTH tools'
//! module paths. That is semantically defensible — both tools see the identical
//! module — but it makes ~18% of the claim corpus depend on a **TY-authored
//! artifact that the recorded toolchain never mentions**. A reader of the claim
//! could not tell that a fifth of the corpus was mediated by a first-party file.
//!
//! Installing the genuine upstream library removes the problem at the root
//! rather than documenting around it: with `library/` pinned at
//! [`TLAPM_PIN`], **all 141 eligible rows parse** under SANY, so the strict
//! comparison never needs the stub.
//!
//! # Pinning
//!
//! GitHub's `codeload` tarballs are not byte-stable across recompression, so the
//! tarball digest is *not* the authority. Each `.tla` module is pinned by its
//! own sha256 in [`TLAPS_LIBRARY_SHA256`] and verified after extraction. That is
//! immune to recompression and pins exactly what the tools will read.
//!
//! Follows the repo convention of shelling out to `curl`/`tar`/`java` rather
//! than adding an HTTP client dependency (cf. `cmd_corpus`, `cmd_tlc`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use super::{run, sha256_hex};

/// Upstream proof-library source, pinned to an exact commit.
const TLAPM_REPO: &str = "https://github.com/tlaplus/tlapm";
/// The pinned `tlaplus/tlapm` commit whose `library/` this installs.
pub(crate) const TLAPM_PIN: &str = "3ab43c7ff31db4ced850619d4746fa4c841a7681";

/// Directory name under the TLC install dir where the library lands.
pub(crate) const PROOF_LIBRARY_DIR: &str = "tla-library";

/// Per-file sha256 of every `.tla` module in `tlapm@TLAPM_PIN:library/`.
///
/// This is the pin authority — see the module docs on why the tarball digest is
/// not. Regenerate with:
/// `cd <library> && for f in *.tla; do echo "$f $(sha256sum "$f")"; done`
pub(crate) const TLAPS_LIBRARY_SHA256: &[(&str, &str)] = &[
    (
        "Bags.tla",
        "6a2b0328d64eef6fe3c7f08f382bc4ed11a61e5231cf39f88c41affc3cad91b8",
    ),
    (
        "BagsTheorems.tla",
        "af67c8abfb23b0f3ee0d0fa6d0427948f268a14de1b9bb408e7acda23a1d1db4",
    ),
    (
        "BagsTheorems_proofs.tla",
        "e416eba583b338a98718b97bd5077fd5a057b310649066d43b77dfc6c6b9c0d4",
    ),
    (
        "FiniteSetTheorems.tla",
        "484bf0f9ab6a69ef45f7282f7f92dcf1e6ae139e44117b0d5a4427635818e773",
    ),
    (
        "FiniteSetTheorems_proofs.tla",
        "a0c71d0b98986784096f9f56b4395edb76dd013bf621d6fa0b9659c1cdbf80a6",
    ),
    (
        "FiniteSets.tla",
        "eca905df95c16fb12f0f205a38b429c216a11b5c8429d270af044f05600a6380",
    ),
    (
        "NaturalsInduction.tla",
        "08f52420cdaaf11292ed366782b5ce5b596bb7cbe789526a1cfd8806dbf98624",
    ),
    (
        "NaturalsInduction_proofs.tla",
        "8b48da8d05a2eb61698c5cd512a25c07cfe64ad6295976a853b3697e9640cddb",
    ),
    (
        "RealTime.tla",
        "23078bd5961b783bb888cc5654a5c92ae6296fae4f426dc6e1f57a29109c52f0",
    ),
    (
        "SequenceTheorems.tla",
        "1fdbed9077bba9db329e499535be29f8d2e6fba3a2b338e364c3b0ec56596bf9",
    ),
    (
        "SequenceTheorems_proofs.tla",
        "8f30d6f458ac623df4db14c88ab758fcb2a97889e42427e67634dcfc61510b20",
    ),
    (
        "TLAPS.tla",
        "5cc604533e49792c1c3d050a38d845d08d9c209879ca20c86de04975bc4bc563",
    ),
    (
        "WellFoundedInduction.tla",
        "6f2f274c2e987d1edcf004d8e37b053f1f82b912e66d6a51bae0af8012ddcbec",
    ),
    (
        "WellFoundedInduction_proofs.tla",
        "f3af7b047663143f50d33c2587ca96cfeb27d47a80f365701cc0f1f7f569584c",
    ),
];

/// Resolve the proof-library directory under a TLC install dir.
pub(crate) fn proof_library_path(dest: &Path) -> PathBuf {
    dest.join(PROOF_LIBRARY_DIR)
}

/// Provenance of whichever TLA library a run actually resolved.
///
/// The strict TY-vs-TLC claim records its inputs; a library that mediates 18% of
/// the corpus is an input. This is what makes it nameable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LibraryProvenance {
    /// The pinned upstream `tlaplus/tlapm` `library/` — what strict evidence wants.
    UpstreamTlaps,
    /// The repo's own `test_specs/tla_library` stub set — usable, but first-party.
    FirstPartyStub,
    /// An operator-supplied directory (explicit flag or `TLA_LIBRARY`/`TLA_PLUS_LIBRARY`).
    OperatorSupplied,
}

impl LibraryProvenance {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::UpstreamTlaps => "upstream_tlaps",
            Self::FirstPartyStub => "first_party_stub",
            Self::OperatorSupplied => "operator_supplied",
        }
    }

    /// Whether this provenance is acceptable for a strict, publishable claim.
    /// Only the pinned upstream library is: it is the artifact a third party can
    /// independently obtain and check.
    pub(crate) fn is_strict_qualified(self) -> bool {
        matches!(self, Self::UpstreamTlaps)
    }
}

/// Classify an already-resolved library directory by content.
///
/// Identity is decided by digest, not by path: a directory is the upstream
/// library iff every pinned module is present with its pinned sha256. That keeps
/// the classification honest if someone points `TLA_LIBRARY` at a copy.
pub(crate) fn classify_library(dir: &Path) -> LibraryProvenance {
    if verify_pinned_modules(dir).is_ok() {
        return LibraryProvenance::UpstreamTlaps;
    }
    // The repo stub is recognizable by a companion module the upstream library
    // does not ship at all.
    if dir.join("Apalache.tla").is_file() && dir.join("TLAPS.tla").is_file() {
        return LibraryProvenance::FirstPartyStub;
    }
    LibraryProvenance::OperatorSupplied
}

/// Verify every pinned module is present at `dir` with its pinned sha256.
pub(crate) fn verify_pinned_modules(dir: &Path) -> Result<()> {
    for (name, expected) in TLAPS_LIBRARY_SHA256 {
        let path = dir.join(name);
        if !path.is_file() {
            bail!("missing pinned proof-library module {}", path.display());
        }
        let got = sha256_hex(&path)
            .with_context(|| format!("hashing proof-library module {}", path.display()))?;
        if !got.eq_ignore_ascii_case(expected) {
            bail!(
                "proof-library module {} sha256 mismatch: got {got}, expected {expected}",
                path.display()
            );
        }
    }
    Ok(())
}

/// Install the pinned upstream proof library into `<dest>/tla-library`.
pub(crate) fn do_install(dest: &Path, force: bool) -> Result<()> {
    let lib = proof_library_path(dest);

    if lib.is_dir() && verify_pinned_modules(&lib).is_ok() && !force {
        println!(
            "TLA+ proof library already installed at {} (tlapm@{}); use --force to re-install",
            lib.display(),
            &TLAPM_PIN[..12]
        );
        return Ok(());
    }

    let work = std::env::temp_dir().join("ty-tlaps-library-dl");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).with_context(|| format!("creating {}", work.display()))?;
    let tarball = work.join("tlapm.tar.gz");
    let url = format!("https://codeload.github.com/tlaplus/tlapm/tar.gz/{TLAPM_PIN}");

    eprintln!("downloading TLA+ proof library from {TLAPM_REPO} @ {TLAPM_PIN}");
    run(
        Command::new("curl")
            .args(["-fsSL", "--retry", "3", "-o"])
            .arg(&tarball)
            .arg(&url),
        "curl download of tlapm archive",
    )?;

    // Extract only `library/`: the rest of tlapm is an OCaml prover we do not need.
    run(
        Command::new("tar")
            .arg("-xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(&work)
            .arg(format!("tlapm-{TLAPM_PIN}/library")),
        "tar extract of tlapm library/",
    )?;

    let extracted = work.join(format!("tlapm-{TLAPM_PIN}")).join("library");
    if !extracted.is_dir() {
        bail!(
            "tlapm archive did not contain library/ at {}",
            extracted.display()
        );
    }

    // Verify BEFORE installing: never land an unpinned artifact.
    verify_pinned_modules(&extracted)
        .context("pinned proof-library digests — refusing to install an altered library")?;

    if lib.exists() {
        std::fs::remove_dir_all(&lib)
            .with_context(|| format!("removing previous proof library at {}", lib.display()))?;
    }
    std::fs::create_dir_all(&lib)
        .with_context(|| format!("creating proof library dir {}", lib.display()))?;
    for (name, _) in TLAPS_LIBRARY_SHA256 {
        std::fs::copy(extracted.join(name), lib.join(name))
            .with_context(|| format!("installing proof-library module {name}"))?;
    }
    let _ = std::fs::remove_dir_all(&work);

    println!(
        "TLA+ proof library installed: {} ({} modules, tlapm@{})",
        lib.display(),
        TLAPS_LIBRARY_SHA256.len(),
        &TLAPM_PIN[..12]
    );
    println!(
        "  this is the upstream library, so TLC parses every eligible corpus row without the\n  \
         repo's first-party test_specs/tla_library stub. Run `ty corpus doctor` to confirm."
    );
    Ok(())
}

/// Verify an installed proof library: pinned digests plus, when a JDK and TLC
/// jar are available, that SANY actually resolves `TLAPS` through it.
pub(crate) fn do_verify(dest: &Path) -> Result<()> {
    let lib = proof_library_path(dest);
    if !lib.is_dir() {
        bail!(
            "TLA+ proof library NOT found: {} does not exist. Run `ty install-tlc proof-library`.",
            lib.display()
        );
    }
    verify_pinned_modules(&lib)?;
    println!(
        "TLA+ proof library OK at {} ({} pinned modules, tlapm@{})",
        lib.display(),
        TLAPS_LIBRARY_SHA256.len(),
        &TLAPM_PIN[..12]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_manifest_is_sorted_and_unique() {
        let names: Vec<&str> = TLAPS_LIBRARY_SHA256.iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            names, sorted,
            "pin manifest must be sorted and duplicate-free"
        );
    }

    #[test]
    fn pin_manifest_covers_the_modules_the_corpus_needs() {
        // The three modules the eligible-corpus SANY sweep named as missing.
        for required in [
            "TLAPS.tla",
            "FiniteSetTheorems.tla",
            "NaturalsInduction.tla",
        ] {
            assert!(
                TLAPS_LIBRARY_SHA256.iter().any(|(n, _)| *n == required),
                "pin manifest must include {required}"
            );
        }
    }

    #[test]
    fn pin_digests_are_hex_sha256() {
        for (name, digest) in TLAPS_LIBRARY_SHA256 {
            assert_eq!(digest.len(), 64, "{name} digest must be 64 hex chars");
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{name} digest must be hex"
            );
        }
    }

    #[test]
    fn only_upstream_provenance_is_strict_qualified() {
        assert!(LibraryProvenance::UpstreamTlaps.is_strict_qualified());
        assert!(!LibraryProvenance::FirstPartyStub.is_strict_qualified());
        assert!(!LibraryProvenance::OperatorSupplied.is_strict_qualified());
    }

    #[test]
    fn missing_directory_fails_pin_verification() {
        let missing = std::env::temp_dir().join("ty-nonexistent-proof-library-xyzzy");
        assert!(verify_pinned_modules(&missing).is_err());
        assert_eq!(
            classify_library(&missing),
            LibraryProvenance::OperatorSupplied
        );
    }
}
