// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared native successor-kernel ABI.
//!
//! This module describes the stable shape that trust_cg, trust-ir adapters, TLA
//! checkers, MCC/Petri adapters, and future hardware-model adapters can share:
//! one flat input state plus optional constants and scratch space produces a
//! flat successor buffer plus machine-readable metadata.
//!
//! The types here are deliberately domain-neutral. Domain crates own semantic
//! lowering, witness validation, and parity policy; this crate owns the native
//! call shape and pure-data kernel descriptors.
//!
//! Kernel artifact adoption evidence in this module intentionally mirrors only
//! stable strings from trust_cg's `KernelArtifactContract` surface. Keeping this
//! crate independent of `trust_cg-codegen` preserves the near-leaf ABI boundary
//! while giving TY, MCC, and later consumers one data shape for native
//! successor/predicate adoption decisions.

use std::{collections::BTreeMap, fmt};

use thiserror::Error;

use crate::JitRuntimeErrorKind;

/// Stable trust-codegen schema name for native kernel artifact contracts.
///
/// Mirrored from trust-codegen `KernelArtifactContract`; do not change without a
/// coordinated trust_cg/TY schema migration.
pub const KERNEL_ARTIFACT_CONTRACT_SCHEMA: &str = "trust_cg.kernel_artifact_contract/v1";

/// Current trust-codegen native kernel artifact contract schema version.
pub const KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// TY consumer name used inside trust-codegen native kernel artifact contracts.
pub const TY_KERNEL_ARTIFACT_CONSUMER: &str = "ty";

/// Stable contract kind string for native successor kernels.
pub const SUCCESSOR_KERNEL_ARTIFACT_KIND: &str = "successor_kernel";

/// Stable contract kind string for native predicate kernels.
pub const PREDICATE_KERNEL_ARTIFACT_KIND: &str = "predicate_kernel";

/// Stable contract kind string for analytical helper kernels.
pub const ANALYTICAL_KERNEL_ARTIFACT_KIND: &str = "analytical_kernel";

/// Stable contract kind string for AY/symbolic helper kernels.
pub const AY_SYMBOLIC_KERNEL_ARTIFACT_KIND: &str = "ay_symbolic_kernel";

/// Stable contract kind string for compiled fingerprint helper kernels.
pub const FINGERPRINT_KERNEL_ARTIFACT_KIND: &str = "fingerprint_kernel";

/// Stable contract kind string for proof or trace replay helper kernels.
pub const REPLAY_KERNEL_ARTIFACT_KIND: &str = "replay_kernel";

/// Stable contract kind string for native helper kernels.
pub const NATIVE_HELPER_KERNEL_ARTIFACT_KIND: &str = "native_helper_kernel";

/// Manifest metadata key expected before TY adopts a successor kernel.
pub const TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA: &str = "ty.successor_kernel.evidence";

/// Manifest metadata key expected before TY adopts a predicate kernel.
pub const TY_PREDICATE_KERNEL_EVIDENCE_METADATA: &str = "ty.predicate_kernel.evidence";

/// Stable ABI string for non-variadic C-callable native kernel symbols.
pub const KERNEL_SYMBOL_ABI_EXTERN_C: &str = "extern_c";

/// Stable trust-codegen ABI value string for a 32-bit integer.
pub const KERNEL_ABI_VALUE_I32: &str = "i32";

/// Stable trust-codegen ABI value string for a native pointer.
pub const KERNEL_ABI_VALUE_PTR: &str = "ptr";

/// Stable trust-codegen ABI value string for no returned value.
pub const KERNEL_ABI_VALUE_VOID: &str = "void";

/// Native successor-kernel execution status.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuccessorKernelStatus {
    /// Kernel ran to completion and wrote all enabled successors that fit.
    Ok = 0,
    /// The current state has no enabled successor in this kernel lane.
    Disabled = 1,
    /// Kernel produced more successors than the caller-provided buffer allowed.
    BufferOverflow = 2,
    /// Kernel hit a runtime error such as overflow or division by zero.
    RuntimeError = 3,
    /// Kernel exists but this state/path must fall back to the interpreter.
    FallbackNeeded = 4,
    /// Kernel was not executable for this model or property.
    Unsupported = 5,
}

/// Domain-neutral reason a successor kernel cannot be used.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuccessorKernelUnsupportedReason {
    /// No unsupported reason applies.
    None = 0,
    /// Lowering could not represent a source operation.
    UnsupportedOperation = 1,
    /// The state layout does not fit the native flat-state ABI.
    UnsupportedStateLayout = 2,
    /// The transition relation requires compound values outside this kernel.
    CompoundValue = 3,
    /// The generated kernel would exceed configured size or arity limits.
    TooLarge = 4,
    /// Runtime parity or validation policy has not promoted this kernel.
    ParityRequired = 5,
    /// Backend-specific compilation is unavailable in this build.
    BackendUnavailable = 6,
}

/// Product-facing kernel intent for native artifacts shared across TY-like consumers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelArtifactKind {
    /// Native successor enumeration or expansion kernel.
    SuccessorKernel,
    /// Native state predicate, invariant, or filter kernel.
    PredicateKernel,
    /// Analytical helper kernel used for solver-independent model analysis.
    AnalyticalKernel,
    /// AY or symbolic backend helper kernel.
    AYSymbolicKernel,
    /// Compiled fingerprint helper kernel.
    FingerprintKernel,
    /// Proof, witness, trace, or native artifact replay helper kernel.
    ReplayKernel,
    /// Native helper kernel that does not fit a more specific role.
    NativeHelperKernel,
    /// Downstream-defined kernel kind.
    Other(String),
}

impl KernelArtifactKind {
    /// Stable contract string for this kernel kind.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::SuccessorKernel => SUCCESSOR_KERNEL_ARTIFACT_KIND,
            Self::PredicateKernel => PREDICATE_KERNEL_ARTIFACT_KIND,
            Self::AnalyticalKernel => ANALYTICAL_KERNEL_ARTIFACT_KIND,
            Self::AYSymbolicKernel => AY_SYMBOLIC_KERNEL_ARTIFACT_KIND,
            Self::FingerprintKernel => FINGERPRINT_KERNEL_ARTIFACT_KIND,
            Self::ReplayKernel => REPLAY_KERNEL_ARTIFACT_KIND,
            Self::NativeHelperKernel => NATIVE_HELPER_KERNEL_ARTIFACT_KIND,
            Self::Other(value) => value.as_str(),
        }
    }
}

/// Stable 128-bit checksum value supplied by trust-codegen artifact manifests.
///
/// This wrapper stores the value only. TY intentionally does not reimplement
/// trust_cg's manifest hashing here because the manifest producer owns canonical
/// encoding and checksum generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelArtifactChecksum(u128);

impl KernelArtifactChecksum {
    /// Create a checksum from its raw 128-bit value.
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Return the raw 128-bit checksum value.
    pub const fn get(self) -> u128 {
        self.0
    }

    /// Whether this checksum carries the zero placeholder value.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for KernelArtifactChecksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trust_cg-stable128:{:032x}", self.0)
    }
}

/// Artifact descriptor checksums that TY records before native adoption.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelArtifactChecksums {
    /// Target descriptor checksum expected by the consumer.
    pub target: KernelArtifactChecksum,
    /// ABI descriptor checksum expected by the consumer.
    pub abi: KernelArtifactChecksum,
    /// Layout manifest checksum expected by the consumer.
    pub layout: KernelArtifactChecksum,
    /// Proof policy checksum expected by the consumer.
    pub proof_policy: KernelArtifactChecksum,
    /// Stable checksum for the transition relation or predicate source.
    pub semantic: KernelArtifactChecksum,
}

impl KernelArtifactChecksums {
    /// Create descriptor and semantic checksum evidence.
    pub const fn new(
        target: KernelArtifactChecksum,
        abi: KernelArtifactChecksum,
        layout: KernelArtifactChecksum,
        proof_policy: KernelArtifactChecksum,
        semantic: KernelArtifactChecksum,
    ) -> Self {
        Self {
            target,
            abi,
            layout,
            proof_policy,
            semantic,
        }
    }
}

/// One ABI value in a native artifact symbol signature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelAbiValue {
    /// Stable value-kind string mirrored from trust_cg's `AbiValueKind` contract.
    pub kind: String,
    /// Whether null is a valid value. Relevant for pointers.
    pub nullable: bool,
}

impl KernelAbiValue {
    /// Create a non-nullable ABI value.
    #[must_use]
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            nullable: false,
        }
    }

    /// Mark the value as nullable.
    #[must_use]
    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
}

/// Canonical callable-symbol signature for a native kernel artifact.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelSymbolSignature {
    /// ABI or calling convention this signature expects.
    pub abi: String,
    /// Positional parameter values.
    pub params: Vec<KernelAbiValue>,
    /// Positional return values.
    pub returns: Vec<KernelAbiValue>,
    /// Whether the function is variadic.
    pub variadic: bool,
}

impl KernelSymbolSignature {
    /// Create a non-variadic `extern "C"` signature.
    #[must_use]
    pub fn extern_c(params: Vec<KernelAbiValue>, returns: Vec<KernelAbiValue>) -> Self {
        Self {
            abi: KERNEL_SYMBOL_ABI_EXTERN_C.to_owned(),
            params,
            returns,
            variadic: false,
        }
    }

    /// Stable signature for [`SuccessorKernelFn`].
    #[must_use]
    pub fn native_successor_kernel() -> Self {
        Self::extern_c(
            vec![
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_I32),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR).nullable(),
                KernelAbiValue::new(KERNEL_ABI_VALUE_I32),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR).nullable(),
                KernelAbiValue::new(KERNEL_ABI_VALUE_I32),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_I32),
            ],
            vec![KernelAbiValue::new(KERNEL_ABI_VALUE_VOID)],
        )
    }

    /// Stable signature for state predicate kernels using `JitInvariantFn`.
    #[must_use]
    pub fn native_state_predicate_kernel() -> Self {
        Self::extern_c(
            vec![
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_I32),
            ],
            vec![KernelAbiValue::new(KERNEL_ABI_VALUE_VOID)],
        )
    }
}

/// Finite-domain facts for native kernel artifacts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelStateDomain {
    /// State space is finite with a known upper bound.
    Finite {
        /// Number of state variables encoded by the kernel.
        variable_count: u32,
        /// Optional upper bound on distinct states.
        max_state_count: Option<u64>,
    },
    /// State space is bounded by a named downstream invariant.
    BoundedByInvariant {
        /// Stable invariant or proof fact name.
        invariant: String,
    },
    /// Domain evidence is intentionally not exposed yet.
    Unknown,
}

/// Stable schema name for frontend-neutral whole-program kernel metadata.
pub const WHOLE_PROGRAM_KERNEL_SCHEMA: &str = "ty.whole_program_kernel/v1";

/// Current whole-program kernel metadata schema version.
pub const WHOLE_PROGRAM_KERNEL_SCHEMA_VERSION: u32 = 1;

/// Source-level role for one contiguous slot range in a flat kernel state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelStateSlotKind {
    /// TLA+ state variable slot range, keyed by VarIdx.
    TlaStateVar {
        /// State variable index in checker/TIR order.
        var_idx: u32,
    },
    /// Petri-net marking place, keyed by place index.
    PetriPlace {
        /// Place index in the source net.
        place_idx: u32,
    },
    /// Hardware-model register vector lane or scalar register.
    HardwareRegister {
        /// Register-vector descriptor name.
        vector: String,
        /// Register index inside the vector.
        register_idx: u32,
    },
    /// Downstream-defined slot kind.
    Other(String),
}

/// One source-level state slot range as consumed or produced by native kernels.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelStateSlot {
    /// Stable ordinal in source-state order.
    pub ordinal: u32,
    /// Starting i64 slot in the flat kernel state.
    pub offset: u32,
    /// Number of contiguous i64 slots occupied by this source value.
    pub slot_count: u32,
    /// Source role for this slot range.
    pub kind: KernelStateSlotKind,
    /// Optional semantic type or layout tag, for example `i64`, `bool`, `u16`,
    /// `compact_set_bitmask`, or a frontend-owned layout name.
    pub value_kind: Option<String>,
    /// Downstream extension metadata. Keys are deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl KernelStateSlot {
    /// Create a state-slot descriptor.
    #[must_use]
    pub fn new(ordinal: u32, offset: u32, slot_count: u32, kind: KernelStateSlotKind) -> Self {
        Self {
            ordinal,
            offset,
            slot_count,
            kind,
            value_kind: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Create a TLA+ state variable slot descriptor.
    #[must_use]
    pub fn tla_state_var(var_idx: u32, offset: u32, slot_count: u32) -> Self {
        Self::new(
            var_idx,
            offset,
            slot_count,
            KernelStateSlotKind::TlaStateVar { var_idx },
        )
    }

    /// Create a Petri marking place slot descriptor.
    #[must_use]
    pub fn petri_place(place_idx: u32, offset: u32, slot_count: u32) -> Self {
        Self::new(
            place_idx,
            offset,
            slot_count,
            KernelStateSlotKind::PetriPlace { place_idx },
        )
    }

    /// Create a hardware register slot descriptor.
    #[must_use]
    pub fn hardware_register(
        ordinal: u32,
        offset: u32,
        slot_count: u32,
        vector: impl Into<String>,
        register_idx: u32,
    ) -> Self {
        Self::new(
            ordinal,
            offset,
            slot_count,
            KernelStateSlotKind::HardwareRegister {
                vector: vector.into(),
                register_idx,
            },
        )
    }

    /// Attach a semantic value/layout tag.
    #[must_use]
    pub fn with_value_kind(mut self, value_kind: impl Into<String>) -> Self {
        self.value_kind = Some(value_kind.into());
        self
    }
}

/// Packed Petri marking layout facts for a whole-program kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelMarkingLayout {
    /// Descriptor name, for example the Petri net name or adapter lane.
    pub name: String,
    /// Total places in the source net.
    pub place_count: u32,
    /// Places stored in the packed marking.
    pub packed_place_count: u32,
    /// Token storage width per packed place.
    pub token_width_bits: u32,
    /// Source place indexes omitted from the packed representation because
    /// they can be reconstructed from marking invariants.
    pub implied_places: Vec<u32>,
    /// Downstream extension metadata. Keys are deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl KernelMarkingLayout {
    /// Create a Petri marking layout descriptor.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        place_count: u32,
        packed_place_count: u32,
        token_width_bits: u32,
    ) -> Self {
        Self {
            name: name.into(),
            place_count,
            packed_place_count,
            token_width_bits,
            implied_places: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Attach implied-place indexes in deterministic order.
    #[must_use]
    pub fn with_implied_places(mut self, implied_places: impl IntoIterator<Item = u32>) -> Self {
        self.implied_places = implied_places.into_iter().collect();
        self.implied_places.sort_unstable();
        self.implied_places.dedup();
        self
    }
}

/// Hardware register-vector layout facts for a whole-program kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelRegisterVectorLayout {
    /// Descriptor name referenced by [`KernelStateSlotKind::HardwareRegister`].
    pub name: String,
    /// Architecture or adapter register class, for example `gpr`, `simd`, or
    /// a hardware-model-specific register-file name.
    pub register_class: String,
    /// Number of logical registers in this vector.
    pub register_count: u32,
    /// Number of lanes per logical register.
    pub lane_count: u32,
    /// Width of each lane in bits.
    pub lane_width_bits: u32,
    /// Downstream extension metadata. Keys are deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl KernelRegisterVectorLayout {
    /// Create a hardware register-vector layout descriptor.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        register_class: impl Into<String>,
        register_count: u32,
        lane_count: u32,
        lane_width_bits: u32,
    ) -> Self {
        Self {
            name: name.into(),
            register_class: register_class.into(),
            register_count,
            lane_count,
            lane_width_bits,
            metadata: BTreeMap::new(),
        }
    }

    /// Total logical payload bits represented by the vector.
    #[must_use]
    pub fn total_payload_bits(&self) -> Option<u64> {
        u64::from(self.register_count)
            .checked_mul(u64::from(self.lane_count))?
            .checked_mul(u64::from(self.lane_width_bits))
    }
}

/// Stable pure-data descriptor for a native predicate kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateKernelDescriptor {
    /// Human-readable predicate or invariant name.
    pub name: String,
    /// Flat i64 slots per input state.
    pub state_len: u32,
    /// Predicate produces deterministic answers for parity checks.
    pub deterministic: bool,
    /// Predicate must pass domain parity before it is trusted for answers.
    pub requires_parity: bool,
}

impl PredicateKernelDescriptor {
    /// Create a descriptor with conservative defaults for new adapters.
    #[must_use]
    pub fn new(name: impl Into<String>, state_len: u32) -> Self {
        Self {
            name: name.into(),
            state_len,
            deterministic: true,
            requires_parity: true,
        }
    }
}

/// Domain-neutral flat-buffer shape for helper kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenericKernelShape {
    /// Flat i64 input slots consumed by the kernel.
    pub input_len: u32,
    /// Flat i64 output slots written by the kernel.
    pub output_len: u32,
    /// Flat i64 constants consumed by the kernel.
    pub constants_len: u32,
    /// Flat i64 scratch slots required by the kernel.
    pub scratch_len: u32,
}

impl GenericKernelShape {
    /// Create a generic helper-kernel shape.
    #[must_use]
    pub fn new(input_len: u32, output_len: u32, constants_len: u32, scratch_len: u32) -> Self {
        Self {
            input_len,
            output_len,
            constants_len,
            scratch_len,
        }
    }

    /// Total caller-provided i64 slots excluding scratch.
    #[must_use]
    pub fn io_slots(&self) -> Option<usize> {
        (self.input_len as usize)
            .checked_add(self.output_len as usize)?
            .checked_add(self.constants_len as usize)
    }
}

/// Stable pure-data descriptor for analytical, solver, replay, and helper kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenericKernelDescriptor {
    /// Human-readable kernel or helper name.
    pub name: String,
    /// Product-facing helper role.
    pub kind: KernelArtifactKind,
    /// Flat-buffer ABI shape.
    pub shape: GenericKernelShape,
    /// Kernel produces deterministic outputs for identical inputs.
    pub deterministic: bool,
    /// Kernel is expected not to mutate memory except declared output/scratch buffers.
    pub side_effect_free: bool,
    /// Kernel adoption requires frontend- or backend-owned validation evidence.
    pub requires_validation: bool,
    /// Downstream extension metadata. Keys are deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl GenericKernelDescriptor {
    /// Create a generic helper descriptor with conservative validation defaults.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        kind: KernelArtifactKind,
        shape: GenericKernelShape,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            shape,
            deterministic: true,
            side_effect_free: true,
            requires_validation: true,
            metadata: BTreeMap::new(),
        }
    }

    /// Create an analytical helper-kernel descriptor.
    #[must_use]
    pub fn analytical(name: impl Into<String>, shape: GenericKernelShape) -> Self {
        Self::new(name, KernelArtifactKind::AnalyticalKernel, shape)
    }

    /// Create a AY/symbolic helper-kernel descriptor.
    #[must_use]
    pub fn ay_symbolic(name: impl Into<String>, shape: GenericKernelShape) -> Self {
        Self::new(name, KernelArtifactKind::AYSymbolicKernel, shape)
    }

    /// Create a compiled fingerprint helper-kernel descriptor.
    #[must_use]
    pub fn fingerprint(name: impl Into<String>, shape: GenericKernelShape) -> Self {
        Self::new(name, KernelArtifactKind::FingerprintKernel, shape)
    }

    /// Create a proof or trace replay helper-kernel descriptor.
    #[must_use]
    pub fn replay(name: impl Into<String>, shape: GenericKernelShape) -> Self {
        Self::new(name, KernelArtifactKind::ReplayKernel, shape)
    }

    /// Create a native helper-kernel descriptor.
    #[must_use]
    pub fn native_helper(name: impl Into<String>, shape: GenericKernelShape) -> Self {
        Self::new(name, KernelArtifactKind::NativeHelperKernel, shape)
    }

    /// Override whether identical inputs deterministically produce identical outputs.
    #[must_use]
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Override the declared side-effect policy.
    #[must_use]
    pub fn with_side_effect_free(mut self, side_effect_free: bool) -> Self {
        self.side_effect_free = side_effect_free;
        self
    }

    /// Override whether adoption requires validation evidence.
    #[must_use]
    pub fn with_requires_validation(mut self, requires_validation: bool) -> Self {
        self.requires_validation = requires_validation;
        self
    }

    /// Attach deterministic extension metadata.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A callable kernel entry in a whole-program kernel bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelEntry {
    /// Successor enumeration or expansion kernel.
    Successor {
        /// Callable symbol or logical entry name.
        entry_symbol: String,
        /// Successor-kernel descriptor.
        descriptor: SuccessorKernelDescriptor,
    },
    /// State predicate, invariant, or filter kernel.
    Predicate {
        /// Callable symbol or logical entry name.
        entry_symbol: String,
        /// Predicate-kernel descriptor.
        descriptor: PredicateKernelDescriptor,
    },
    /// Analytical, solver, replay, fingerprint, or native helper kernel.
    Generic {
        /// Callable symbol or logical entry name.
        entry_symbol: String,
        /// Expected callable signature.
        signature: KernelSymbolSignature,
        /// Helper-kernel descriptor.
        descriptor: GenericKernelDescriptor,
    },
}

impl KernelEntry {
    /// Create a successor-kernel entry.
    #[must_use]
    pub fn successor(
        entry_symbol: impl Into<String>,
        descriptor: SuccessorKernelDescriptor,
    ) -> Self {
        Self::Successor {
            entry_symbol: entry_symbol.into(),
            descriptor,
        }
    }

    /// Create a predicate-kernel entry.
    #[must_use]
    pub fn predicate(
        entry_symbol: impl Into<String>,
        descriptor: PredicateKernelDescriptor,
    ) -> Self {
        Self::Predicate {
            entry_symbol: entry_symbol.into(),
            descriptor,
        }
    }

    /// Create a generic helper-kernel entry.
    #[must_use]
    pub fn generic(
        entry_symbol: impl Into<String>,
        signature: KernelSymbolSignature,
        descriptor: GenericKernelDescriptor,
    ) -> Self {
        Self::Generic {
            entry_symbol: entry_symbol.into(),
            signature,
            descriptor,
        }
    }

    /// Product-facing kind for this entry.
    #[must_use]
    pub fn kind(&self) -> KernelArtifactKind {
        match self {
            Self::Successor { .. } => KernelArtifactKind::SuccessorKernel,
            Self::Predicate { .. } => KernelArtifactKind::PredicateKernel,
            Self::Generic { descriptor, .. } => descriptor.kind.clone(),
        }
    }

    /// Callable symbol or logical entry name.
    #[must_use]
    pub fn entry_symbol(&self) -> &str {
        match self {
            Self::Successor { entry_symbol, .. }
            | Self::Predicate { entry_symbol, .. }
            | Self::Generic { entry_symbol, .. } => entry_symbol,
        }
    }

    /// Expected callable signature for this entry.
    #[must_use]
    pub fn signature(&self) -> KernelSymbolSignature {
        match self {
            Self::Successor { .. } => KernelSymbolSignature::native_successor_kernel(),
            Self::Predicate { .. } => KernelSymbolSignature::native_state_predicate_kernel(),
            Self::Generic { signature, .. } => signature.clone(),
        }
    }
}

/// Frontend-neutral metadata for one whole-program native-kernel bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WholeProgramKernel {
    /// Metadata schema name.
    pub schema: String,
    /// Metadata schema version.
    pub schema_version: u32,
    /// Human-readable program or model name.
    pub name: String,
    /// Flat i64 slots per full state.
    pub state_len: u32,
    /// Source-level slot ranges inside the flat state.
    pub state_slots: Vec<KernelStateSlot>,
    /// Petri marking layouts referenced by slots or entries.
    pub marking_layouts: Vec<KernelMarkingLayout>,
    /// Hardware register-vector layouts referenced by slots or entries.
    pub register_vectors: Vec<KernelRegisterVectorLayout>,
    /// Callable successor and predicate entries.
    pub entries: Vec<KernelEntry>,
    /// Downstream extension metadata. Keys are deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl WholeProgramKernel {
    /// Create an empty whole-program kernel descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, state_len: u32) -> Self {
        Self {
            schema: WHOLE_PROGRAM_KERNEL_SCHEMA.to_owned(),
            schema_version: WHOLE_PROGRAM_KERNEL_SCHEMA_VERSION,
            name: name.into(),
            state_len,
            state_slots: Vec::new(),
            marking_layouts: Vec::new(),
            register_vectors: Vec::new(),
            entries: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Add a state-slot descriptor.
    #[must_use]
    pub fn with_state_slot(mut self, slot: KernelStateSlot) -> Self {
        self.state_slots.push(slot);
        self
    }

    /// Add a Petri marking layout descriptor.
    #[must_use]
    pub fn with_marking_layout(mut self, layout: KernelMarkingLayout) -> Self {
        self.marking_layouts.push(layout);
        self
    }

    /// Add a hardware register-vector layout descriptor.
    #[must_use]
    pub fn with_register_vector(mut self, layout: KernelRegisterVectorLayout) -> Self {
        self.register_vectors.push(layout);
        self
    }

    /// Add a callable kernel entry.
    #[must_use]
    pub fn with_entry(mut self, entry: KernelEntry) -> Self {
        self.entries.push(entry);
        self
    }
}

/// Data-only evidence TY needs before adopting a native kernel artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelArtifactAdoptionEvidence {
    /// Kernel contract schema name.
    pub schema: String,
    /// Kernel contract schema version.
    pub schema_version: u32,
    /// Downstream consumer, such as `ty`.
    pub consumer: String,
    /// Kernel kind.
    pub kind: KernelArtifactKind,
    /// Callable entry symbol in the artifact manifest.
    pub entry_symbol: String,
    /// Expected callable signature.
    pub signature: KernelSymbolSignature,
    /// Target, ABI, layout, proof-policy, and semantic checksums.
    pub checksums: KernelArtifactChecksums,
    /// Finite-domain or bounded-domain evidence for safe state exploration.
    pub state_domain: KernelStateDomain,
    /// Manifest metadata keys that must be present before consumer adoption.
    pub required_manifest_metadata: Vec<String>,
    /// Downstream extension metadata. Keys are deterministic.
    pub metadata: BTreeMap<String, String>,
}

impl KernelArtifactAdoptionEvidence {
    /// Create kernel adoption evidence for any stable or downstream kernel kind.
    #[must_use]
    pub fn kernel(
        consumer: impl Into<String>,
        kind: KernelArtifactKind,
        entry_symbol: impl Into<String>,
        signature: KernelSymbolSignature,
        checksums: KernelArtifactChecksums,
        state_domain: KernelStateDomain,
    ) -> Self {
        Self::new(
            consumer,
            kind,
            entry_symbol,
            signature,
            checksums,
            state_domain,
        )
    }

    /// Create successor-kernel adoption evidence bound to artifact checksums.
    #[must_use]
    pub fn successor_kernel(
        consumer: impl Into<String>,
        entry_symbol: impl Into<String>,
        signature: KernelSymbolSignature,
        checksums: KernelArtifactChecksums,
        state_domain: KernelStateDomain,
    ) -> Self {
        Self::kernel(
            consumer,
            KernelArtifactKind::SuccessorKernel,
            entry_symbol,
            signature,
            checksums,
            state_domain,
        )
    }

    /// Create predicate-kernel adoption evidence bound to artifact checksums.
    #[must_use]
    pub fn predicate_kernel(
        consumer: impl Into<String>,
        entry_symbol: impl Into<String>,
        signature: KernelSymbolSignature,
        checksums: KernelArtifactChecksums,
        state_domain: KernelStateDomain,
    ) -> Self {
        Self::kernel(
            consumer,
            KernelArtifactKind::PredicateKernel,
            entry_symbol,
            signature,
            checksums,
            state_domain,
        )
    }

    fn new(
        consumer: impl Into<String>,
        kind: KernelArtifactKind,
        entry_symbol: impl Into<String>,
        signature: KernelSymbolSignature,
        checksums: KernelArtifactChecksums,
        state_domain: KernelStateDomain,
    ) -> Self {
        Self {
            schema: KERNEL_ARTIFACT_CONTRACT_SCHEMA.to_owned(),
            schema_version: KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION,
            consumer: consumer.into(),
            kind,
            entry_symbol: entry_symbol.into(),
            signature,
            checksums,
            state_domain,
            required_manifest_metadata: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Require a manifest metadata key before this kernel can be adopted.
    #[must_use]
    pub fn with_required_manifest_metadata(mut self, key: impl Into<String>) -> Self {
        self.required_manifest_metadata.push(key.into());
        self.required_manifest_metadata = normalize_string_set(self.required_manifest_metadata);
        self
    }

    /// Validate the mirrored trust-codegen schema name and version.
    pub fn validate_schema(&self) -> Result<(), KernelArtifactAdoptionError> {
        if self.schema == KERNEL_ARTIFACT_CONTRACT_SCHEMA
            && self.schema_version == KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION
        {
            return Ok(());
        }

        Err(KernelArtifactAdoptionError::SchemaMismatch {
            expected_schema: KERNEL_ARTIFACT_CONTRACT_SCHEMA,
            expected_version: KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION,
            actual_schema: self.schema.clone(),
            actual_version: self.schema_version,
        })
    }

    /// Validate required manifest metadata without depending on trust-codegen manifest types.
    pub fn validate_required_manifest_metadata(
        &self,
        manifest_metadata: &BTreeMap<String, String>,
    ) -> Result<(), KernelArtifactAdoptionError> {
        for key in &self.required_manifest_metadata {
            if !manifest_metadata.contains_key(key) {
                return Err(KernelArtifactAdoptionError::MissingManifestMetadata {
                    key: key.clone(),
                });
            }
        }

        Ok(())
    }

    /// Validate schema and required manifest metadata for data-only adoption.
    pub fn validate_adoption_metadata(
        &self,
        manifest_metadata: &BTreeMap<String, String>,
    ) -> Result<(), KernelArtifactAdoptionError> {
        self.validate_schema()?;
        self.validate_required_manifest_metadata(manifest_metadata)
    }
}

/// Validation errors for data-only native kernel artifact adoption evidence.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KernelArtifactAdoptionError {
    /// Kernel artifact schema does not match the mirrored trust-codegen contract.
    #[error(
        "kernel artifact schema mismatch: expected {expected_schema} version {expected_version}, got {actual_schema} version {actual_version}"
    )]
    SchemaMismatch {
        /// Expected schema string.
        expected_schema: &'static str,
        /// Expected schema version.
        expected_version: u32,
        /// Actual schema string.
        actual_schema: String,
        /// Actual schema version.
        actual_version: u32,
    },
    /// Required manifest metadata was not present.
    #[error("missing kernel artifact manifest metadata key {key}")]
    MissingManifestMetadata {
        /// Missing manifest metadata key.
        key: String,
    },
}

/// Stable native successor-kernel output written through an out-pointer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuccessorKernelOut {
    /// Execution status for this kernel call.
    pub status: SuccessorKernelStatus,
    /// Number of successors written to the output buffer.
    pub successor_count: u32,
    /// Number of successors the kernel generated before capacity filtering.
    pub generated_count: u32,
    /// Flat i64 slots per successor.
    pub state_len: u32,
    /// Number of generated successors omitted because of buffer capacity.
    pub overflow_count: u32,
    /// Runtime error kind when `status == RuntimeError`.
    pub runtime_error: JitRuntimeErrorKind,
    /// Unsupported reason when `status == Unsupported`.
    pub unsupported_reason: SuccessorKernelUnsupportedReason,
    /// Reserved machine-readable flags for backend-specific evidence.
    pub metadata_bits: u64,
}

impl Default for SuccessorKernelOut {
    fn default() -> Self {
        Self {
            status: SuccessorKernelStatus::Ok,
            successor_count: 0,
            generated_count: 0,
            state_len: 0,
            overflow_count: 0,
            runtime_error: JitRuntimeErrorKind::DivisionByZero,
            unsupported_reason: SuccessorKernelUnsupportedReason::None,
            metadata_bits: 0,
        }
    }
}

impl SuccessorKernelOut {
    /// Create a successful output summary.
    #[must_use]
    pub fn ok(successor_count: u32, state_len: u32) -> Self {
        Self {
            status: SuccessorKernelStatus::Ok,
            successor_count,
            generated_count: successor_count,
            state_len,
            ..Self::default()
        }
    }

    /// Create a disabled-state output summary.
    #[must_use]
    pub fn disabled(state_len: u32) -> Self {
        Self {
            status: SuccessorKernelStatus::Disabled,
            state_len,
            ..Self::default()
        }
    }

    /// Create a buffer-overflow output summary.
    #[must_use]
    pub fn buffer_overflow(written_count: u32, generated_count: u32, state_len: u32) -> Self {
        Self {
            status: SuccessorKernelStatus::BufferOverflow,
            successor_count: written_count,
            generated_count,
            state_len,
            overflow_count: generated_count.saturating_sub(written_count),
            ..Self::default()
        }
    }

    /// Create a runtime-error output summary.
    #[must_use]
    pub fn runtime_error(runtime_error: JitRuntimeErrorKind, state_len: u32) -> Self {
        Self {
            status: SuccessorKernelStatus::RuntimeError,
            state_len,
            runtime_error,
            ..Self::default()
        }
    }

    /// Create an unsupported-kernel output summary.
    #[must_use]
    pub fn unsupported(reason: SuccessorKernelUnsupportedReason, state_len: u32) -> Self {
        Self {
            status: SuccessorKernelStatus::Unsupported,
            state_len,
            unsupported_reason: reason,
            ..Self::default()
        }
    }

    /// Whether this output requires a non-native fallback path.
    #[must_use]
    pub fn needs_fallback(&self) -> bool {
        matches!(
            self.status,
            SuccessorKernelStatus::FallbackNeeded
                | SuccessorKernelStatus::RuntimeError
                | SuccessorKernelStatus::Unsupported
        )
    }
}

/// Function pointer for a native successor kernel.
pub type SuccessorKernelFn = unsafe extern "C" fn(
    out: *mut SuccessorKernelOut,
    state_in: *const i64,
    state_len: u32,
    constants: *const i64,
    constants_len: u32,
    scratch: *mut i64,
    scratch_len: u32,
    successors: *mut i64,
    successor_capacity: u32,
);

/// Domain-neutral flat-state kernel shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuccessorKernelShape {
    /// Flat i64 slots per state.
    pub state_len: u32,
    /// Flat i64 constants consumed by the kernel.
    pub constants_len: u32,
    /// Flat i64 scratch slots required by the kernel.
    pub scratch_len: u32,
    /// Maximum successors this kernel may emit for one input state.
    pub max_successors: u32,
}

impl SuccessorKernelShape {
    /// Create a kernel shape.
    #[must_use]
    pub fn new(state_len: u32, constants_len: u32, scratch_len: u32, max_successors: u32) -> Self {
        Self {
            state_len,
            constants_len,
            scratch_len,
            max_successors,
        }
    }

    /// Required i64 slots for a fully sized successor buffer.
    #[must_use]
    pub fn successor_buffer_slots(&self) -> Option<usize> {
        (self.state_len as usize).checked_mul(self.max_successors as usize)
    }

    /// Whether the shape can emit at least one full successor.
    #[must_use]
    pub fn can_emit_successors(&self) -> bool {
        self.state_len > 0 && self.max_successors > 0
    }
}

/// Stable pure-data descriptor for a native successor kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuccessorKernelDescriptor {
    /// Human-readable kernel name or action/model lane.
    pub name: String,
    /// Flat-state ABI shape.
    pub shape: SuccessorKernelShape,
    /// Kernel writes full successor states rather than sparse deltas.
    pub writes_full_state: bool,
    /// Kernel produces deterministic successor ordering for parity checks.
    pub deterministic_order: bool,
    /// Kernel must pass domain parity before it is trusted for answers.
    pub requires_parity: bool,
}

impl SuccessorKernelDescriptor {
    /// Create a descriptor with conservative defaults for new adapters.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: SuccessorKernelShape) -> Self {
        Self {
            name: name.into(),
            shape,
            writes_full_state: true,
            deterministic_order: true,
            requires_parity: true,
        }
    }
}

fn normalize_string_set(items: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
    let mut values = items.into_iter().map(Into::into).collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_reports_successor_buffer_slots() {
        let shape = SuccessorKernelShape::new(3, 2, 4, 8);

        assert_eq!(shape.successor_buffer_slots(), Some(24));
        assert!(shape.can_emit_successors());
    }

    #[test]
    fn output_constructors_preserve_metadata() {
        let ok = SuccessorKernelOut::ok(2, 3);
        assert_eq!(ok.status, SuccessorKernelStatus::Ok);
        assert_eq!(ok.successor_count, 2);
        assert_eq!(ok.generated_count, 2);
        assert_eq!(ok.state_len, 3);
        assert!(!ok.needs_fallback());

        let overflow = SuccessorKernelOut::buffer_overflow(4, 7, 3);
        assert_eq!(overflow.status, SuccessorKernelStatus::BufferOverflow);
        assert_eq!(overflow.successor_count, 4);
        assert_eq!(overflow.generated_count, 7);
        assert_eq!(overflow.overflow_count, 3);
    }

    #[test]
    fn unsupported_output_requests_fallback() {
        let out = SuccessorKernelOut::unsupported(
            SuccessorKernelUnsupportedReason::UnsupportedOperation,
            5,
        );

        assert_eq!(out.status, SuccessorKernelStatus::Unsupported);
        assert_eq!(
            out.unsupported_reason,
            SuccessorKernelUnsupportedReason::UnsupportedOperation
        );
        assert!(out.needs_fallback());
    }

    #[test]
    fn descriptor_defaults_require_parity() {
        let descriptor = SuccessorKernelDescriptor::new(
            "petri-fire-transition",
            SuccessorKernelShape::new(4, 0, 2, 3),
        );

        assert_eq!(descriptor.name, "petri-fire-transition");
        assert!(descriptor.writes_full_state);
        assert!(descriptor.deterministic_order);
        assert!(descriptor.requires_parity);
    }

    #[test]
    fn kernel_artifact_schema_and_kind_names_match_trust_cg_contract() {
        assert_eq!(
            KERNEL_ARTIFACT_CONTRACT_SCHEMA,
            "trust_cg.kernel_artifact_contract/v1"
        );
        assert_eq!(KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION, 1);
        assert_eq!(TY_KERNEL_ARTIFACT_CONSUMER, "ty");
        assert_eq!(
            KernelArtifactKind::SuccessorKernel.as_str(),
            SUCCESSOR_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::PredicateKernel.as_str(),
            PREDICATE_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::AnalyticalKernel.as_str(),
            ANALYTICAL_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::AYSymbolicKernel.as_str(),
            AY_SYMBOLIC_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::FingerprintKernel.as_str(),
            FINGERPRINT_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::ReplayKernel.as_str(),
            REPLAY_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::NativeHelperKernel.as_str(),
            NATIVE_HELPER_KERNEL_ARTIFACT_KIND
        );
        assert_eq!(
            KernelArtifactKind::Other("mcc.fire_transition".to_owned()).as_str(),
            "mcc.fire_transition"
        );
    }

    #[test]
    fn native_kernel_signatures_use_stable_trust_cg_abi_value_names() {
        let successor = KernelSymbolSignature::native_successor_kernel();

        assert_eq!(successor.abi, KERNEL_SYMBOL_ABI_EXTERN_C);
        assert_eq!(successor.params.len(), 9);
        assert_eq!(successor.params[0].kind.as_str(), KERNEL_ABI_VALUE_PTR);
        assert_eq!(successor.params[2].kind.as_str(), KERNEL_ABI_VALUE_I32);
        assert!(successor.params[3].nullable);
        assert!(successor.params[5].nullable);
        assert_eq!(successor.returns[0].kind.as_str(), KERNEL_ABI_VALUE_VOID);
        assert!(!successor.variadic);

        let predicate = KernelSymbolSignature::native_state_predicate_kernel();
        assert_eq!(predicate.params.len(), 3);
        assert_eq!(predicate.params[0].kind.as_str(), KERNEL_ABI_VALUE_PTR);
        assert_eq!(predicate.params[2].kind.as_str(), KERNEL_ABI_VALUE_I32);
        assert_eq!(predicate.returns[0].kind.as_str(), KERNEL_ABI_VALUE_VOID);
    }

    #[test]
    fn successor_adoption_evidence_binds_checksums_role_and_metadata() {
        let checksums = KernelArtifactChecksums::new(
            KernelArtifactChecksum::new(0x11),
            KernelArtifactChecksum::new(0x22),
            KernelArtifactChecksum::new(0x33),
            KernelArtifactChecksum::new(0x44),
            KernelArtifactChecksum::new(0x55),
        );
        let evidence = KernelArtifactAdoptionEvidence::successor_kernel(
            TY_KERNEL_ARTIFACT_CONSUMER,
            "ty_next_entry",
            KernelSymbolSignature::native_successor_kernel(),
            checksums,
            KernelStateDomain::Finite {
                variable_count: 3,
                max_state_count: Some(1024),
            },
        )
        .with_required_manifest_metadata(TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA)
        .with_required_manifest_metadata(TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA);

        assert_eq!(evidence.schema, KERNEL_ARTIFACT_CONTRACT_SCHEMA);
        assert_eq!(
            evidence.schema_version,
            KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION
        );
        assert_eq!(evidence.consumer, TY_KERNEL_ARTIFACT_CONSUMER);
        assert_eq!(evidence.kind, KernelArtifactKind::SuccessorKernel);
        assert_eq!(evidence.entry_symbol, "ty_next_entry");
        assert_eq!(
            evidence.checksums.semantic,
            KernelArtifactChecksum::new(0x55)
        );
        assert_eq!(
            evidence.required_manifest_metadata,
            vec![TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA.to_owned()]
        );
        assert_eq!(
            evidence.checksums.semantic.to_string(),
            "trust_cg-stable128:00000000000000000000000000000055"
        );
        assert!(!evidence.checksums.semantic.is_zero());

        let mut manifest_metadata = BTreeMap::new();
        manifest_metadata.insert(
            TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA.to_owned(),
            "finite-domain-v1".to_owned(),
        );
        evidence
            .validate_adoption_metadata(&manifest_metadata)
            .expect("matching metadata should allow data-only adoption");

        manifest_metadata.clear();
        let err = evidence
            .validate_adoption_metadata(&manifest_metadata)
            .expect_err("missing manifest metadata must reject adoption");
        assert_eq!(
            err,
            KernelArtifactAdoptionError::MissingManifestMetadata {
                key: TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA.to_owned()
            }
        );
    }

    #[test]
    fn predicate_adoption_evidence_reports_schema_drift_without_trust_cg_dependency() {
        let checksums = KernelArtifactChecksums::new(
            KernelArtifactChecksum::new(1),
            KernelArtifactChecksum::new(2),
            KernelArtifactChecksum::new(3),
            KernelArtifactChecksum::new(4),
            KernelArtifactChecksum::new(5),
        );
        let mut evidence = KernelArtifactAdoptionEvidence::predicate_kernel(
            TY_KERNEL_ARTIFACT_CONSUMER,
            "ty_typeok_entry",
            KernelSymbolSignature::native_state_predicate_kernel(),
            checksums,
            KernelStateDomain::BoundedByInvariant {
                invariant: "TypeOK".to_owned(),
            },
        )
        .with_required_manifest_metadata(TY_PREDICATE_KERNEL_EVIDENCE_METADATA);
        evidence.schema = "trust_cg.kernel_artifact_contract/v0".to_owned();
        evidence.schema_version = 0;

        let err = evidence
            .validate_schema()
            .expect_err("schema drift must be visible before native adoption");
        assert_eq!(
            err,
            KernelArtifactAdoptionError::SchemaMismatch {
                expected_schema: KERNEL_ARTIFACT_CONTRACT_SCHEMA,
                expected_version: KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION,
                actual_schema: "trust_cg.kernel_artifact_contract/v0".to_owned(),
                actual_version: 0,
            }
        );
    }

    #[test]
    fn whole_program_kernel_metadata_spans_frontend_slot_kinds_and_entries() {
        let marking = KernelMarkingLayout::new("mcc-net", 4, 3, 16).with_implied_places([3, 3]);
        let registers = KernelRegisterVectorLayout::new("x86-gpr", "gpr", 16, 1, 64);
        let successor =
            SuccessorKernelDescriptor::new("fire_t0", SuccessorKernelShape::new(4, 0, 2, 1));
        let predicate = PredicateKernelDescriptor::new("TypeOK", 4);

        let kernel = WholeProgramKernel::new("mixed-program", 4)
            .with_state_slot(KernelStateSlot::tla_state_var(0, 0, 1).with_value_kind("i64"))
            .with_state_slot(KernelStateSlot::petri_place(1, 1, 1).with_value_kind("token_u16"))
            .with_state_slot(KernelStateSlot::hardware_register(2, 2, 2, "x86-gpr", 0))
            .with_marking_layout(marking)
            .with_register_vector(registers)
            .with_entry(KernelEntry::successor("succ_fire_t0", successor))
            .with_entry(KernelEntry::predicate("pred_typeok", predicate));

        assert_eq!(kernel.schema, WHOLE_PROGRAM_KERNEL_SCHEMA);
        assert_eq!(kernel.schema_version, WHOLE_PROGRAM_KERNEL_SCHEMA_VERSION);
        assert_eq!(kernel.state_len, 4);
        assert_eq!(kernel.state_slots.len(), 3);
        assert_eq!(kernel.marking_layouts[0].implied_places, vec![3]);
        assert_eq!(kernel.register_vectors[0].total_payload_bits(), Some(1024));
        assert_eq!(kernel.entries.len(), 2);
        assert_eq!(
            kernel.entries[0].kind(),
            KernelArtifactKind::SuccessorKernel
        );
        assert_eq!(
            kernel.entries[1].kind(),
            KernelArtifactKind::PredicateKernel
        );
        assert_eq!(kernel.entries[0].entry_symbol(), "succ_fire_t0");
    }

    #[test]
    fn generic_helper_kernel_metadata_covers_solver_and_replay_roles() {
        let helper_shape = GenericKernelShape::new(4, 1, 2, 8);
        let helper_signature = KernelSymbolSignature::extern_c(
            vec![
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_I32),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR),
                KernelAbiValue::new(KERNEL_ABI_VALUE_PTR).nullable(),
            ],
            vec![KernelAbiValue::new(KERNEL_ABI_VALUE_I32)],
        );

        let kernel = WholeProgramKernel::new("solver-helper-program", 4)
            .with_entry(KernelEntry::generic(
                "analysis_scc_summary",
                helper_signature.clone(),
                GenericKernelDescriptor::analytical("scc_summary", helper_shape)
                    .with_metadata("domain", "reachability"),
            ))
            .with_entry(KernelEntry::generic(
                "ay_chc_solve",
                helper_signature.clone(),
                GenericKernelDescriptor::ay_symbolic("chc_solver", helper_shape)
                    .with_deterministic(false)
                    .with_metadata("solver", "ay"),
            ))
            .with_entry(KernelEntry::generic(
                "fp64_state",
                helper_signature.clone(),
                GenericKernelDescriptor::fingerprint("compiled_fp64", helper_shape)
                    .with_requires_validation(false),
            ))
            .with_entry(KernelEntry::generic(
                "replay_native_trace",
                helper_signature.clone(),
                GenericKernelDescriptor::replay("native_trace_replay", helper_shape),
            ))
            .with_entry(KernelEntry::generic(
                "native_helper_bridge",
                helper_signature.clone(),
                GenericKernelDescriptor::native_helper("bridge_helper", helper_shape)
                    .with_side_effect_free(false),
            ));

        assert_eq!(helper_shape.io_slots(), Some(7));
        assert_eq!(kernel.entries.len(), 5);
        assert_eq!(
            kernel
                .entries
                .iter()
                .map(KernelEntry::kind)
                .map(|kind| kind.as_str().to_owned())
                .collect::<Vec<_>>(),
            vec![
                ANALYTICAL_KERNEL_ARTIFACT_KIND.to_owned(),
                AY_SYMBOLIC_KERNEL_ARTIFACT_KIND.to_owned(),
                FINGERPRINT_KERNEL_ARTIFACT_KIND.to_owned(),
                REPLAY_KERNEL_ARTIFACT_KIND.to_owned(),
                NATIVE_HELPER_KERNEL_ARTIFACT_KIND.to_owned(),
            ]
        );
        assert_eq!(kernel.entries[1].entry_symbol(), "ay_chc_solve");
        assert_eq!(kernel.entries[1].signature(), helper_signature);

        let KernelEntry::Generic { descriptor, .. } = &kernel.entries[1] else {
            panic!("expected a generic helper entry");
        };
        assert_eq!(descriptor.kind, KernelArtifactKind::AYSymbolicKernel);
        assert!(!descriptor.deterministic);
        assert_eq!(
            descriptor.metadata.get("solver").map(String::as_str),
            Some("ay")
        );

        let KernelEntry::Generic { descriptor, .. } = &kernel.entries[2] else {
            panic!("expected a generic helper entry");
        };
        assert!(!descriptor.requires_validation);

        let KernelEntry::Generic { descriptor, .. } = &kernel.entries[4] else {
            panic!("expected a generic helper entry");
        };
        assert!(!descriptor.side_effect_free);
    }

    #[test]
    fn generic_adoption_evidence_accepts_helper_kernel_kinds() {
        let checksums = KernelArtifactChecksums::new(
            KernelArtifactChecksum::new(10),
            KernelArtifactChecksum::new(20),
            KernelArtifactChecksum::new(30),
            KernelArtifactChecksum::new(40),
            KernelArtifactChecksum::new(50),
        );
        let evidence = KernelArtifactAdoptionEvidence::kernel(
            TY_KERNEL_ARTIFACT_CONSUMER,
            KernelArtifactKind::ReplayKernel,
            "replay_native_trace",
            KernelSymbolSignature::extern_c(
                vec![KernelAbiValue::new(KERNEL_ABI_VALUE_PTR)],
                vec![KernelAbiValue::new(KERNEL_ABI_VALUE_I32)],
            ),
            checksums,
            KernelStateDomain::Unknown,
        )
        .with_required_manifest_metadata("ty.replay_kernel.evidence");

        assert_eq!(evidence.kind.as_str(), REPLAY_KERNEL_ARTIFACT_KIND);
        assert_eq!(evidence.entry_symbol, "replay_native_trace");
        assert_eq!(evidence.checksums.semantic, KernelArtifactChecksum::new(50));
        assert_eq!(
            evidence.required_manifest_metadata,
            vec!["ty.replay_kernel.evidence".to_owned()]
        );
    }
}
