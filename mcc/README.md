# MCC 2026 Submission

This directory supports the official MCC 2026 VM/VMDK submission flow.

Official identity:

- `BK_TOOL=TY`
- tool kind: `parallel`
- artifact: `TY-2026.vmdk`
- companion input image: `mcc2026-input.vmdk`

The authoritative sources are the MCC 2026 rules PDF and the
`MCC2026-SubmissionManual.pdf` shipped in the official submission kit:

- https://mcc.lip6.fr/2026/pdf/rules.pdf
- https://mcc.lip6.fr/2026/archives/SubmissionKit.tar.gz

The Docker image in this directory is a local build/smoke helper. It is not the
official MCC submission artifact.

## Official CLI Flow

```bash
cargo run -p tla-petri --bin ty-mccctl -- doctor --strict
cargo run -p tla-petri --bin ty-mccctl -- build
cargo run -p tla-petri --bin ty-mccctl -- smoke \
  --real-bin /tmp/ty-mcc-agent/agent/ty-mcc
cargo run -p tla-petri --bin ty-mccctl -- submit
```

`submit` is the normal path: it runs final preflight, rebuilds the official
packet, and uploads the latest build to Nuage. It replaces same-named files in
the Nuage drop by default, which is what a corrected resubmission needs.

`preflight` checks the official VM/VMDK contract: the tool name, the declared
parallel class, the `TY-2026.vmdk` filename, the `mcc2026-input.vmdk` input
image, the VMDK SHA-256 sidecar, and `qemu-img check` on both disk images.

`submission` writes:

- `/tmp/mcc2026/submission/TY-2026.vmdk`
- `/tmp/mcc2026/submission/TY-2026.vmdk.sha256`
- `/tmp/mcc2026/submission/TY-2026.SUBMISSION-NOTICE.txt`
- `/tmp/mcc2026/submission/TY-2026.vmdk.tgz`
- `/tmp/mcc2026/submission/TY-2026.vmdk.tgz.sha256`

The raw VMDK is the official upload artifact. The `.tgz` bundle is only a
local transfer convenience. The notice says explicitly that the
organizer-facing `BK_TOOL` value is `TY` and that the submitted tool is
parallel.

`upload` and `submit` use the organizer-provided Nuage/Nextcloud file drop.
The share token can be passed explicitly with `--share-token` or `--share-url`,
or read from the saved `/tmp/mcc2026/submission/upload-state/upload.env` from a
previous successful upload.

## BenchKit Wrapper

The VM must install this file as:

```bash
/home/mcc/BenchKit/BenchKit_head.sh
```

BenchKit sets `BK_EXAMINATION`, `BK_TOOL`, `BK_TIME_CONFINEMENT`,
`BK_MEMORY_CONFINEMENT`, and `BK_BIN_PATH`. The wrapper records those values in
`BK_LOG_FILE` when BenchKit provides it and runs the installed `ty-mcc`
binary.

For a local wrapper smoke:

```bash
BK_INPUT="$PWD/tests/mcc_benchmarks/mutex" \
BK_EXAMINATION=ReachabilityDeadlock \
BK_TOOL=TY \
BK_TIME_CONFINEMENT=60 \
TY_MCC_BIN=/tmp/ty-mcc-agent/agent/ty-mcc \
./mcc/BenchKit_head.sh
```
