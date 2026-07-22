---- MODULE TrustEngineAdmission ----
\* Public-package native evidence contract for tRust `trust.ty.native-evidence.v1`.
\* This spec is intentionally a gate over evidence facts. It does not prove that
\* the current TY native runtime is accepted unless the install-gate manifest
\* and replay evidence facts are present and accepted.

EXTENDS TLC

CONSTANTS
  EvidenceSchema,
  SpecModule,
  PropertyName,
  RefinementMapping,
  CheckedFact,
  ArtifactHash,
  Assumptions,
  Configuration,
  ReplayInvocation,
  ResultStatus,
  AdmissionDisposition,
  AdmissionReasonCode,
  RequestedAuthority,
  InstallAuthority,
  AdmissionFailClosed,
  ProductionSelected,
  TyNativeActivate,
  NativeInstallGateManifestPresent,
  ReplayRootSha256,
  ProofReportSha256,
  AdmissionEvidenceSha256

VARIABLE admission_state

PublicResultStatuses == {"accepted", "checked", "proved", "verified"}

KnownAdmissionDispositions == {"installable", "rejected"}

PresentText(value) == value # "" /\ value # "none"

NativeEvidenceRequiredFieldsPresent ==
  /\ EvidenceSchema = "trust.ty.native-evidence.v1"
  /\ SpecModule = "TrustEngineAdmission"
  /\ PropertyName = "TyPackageReplayAccepted"
  /\ RefinementMapping = "TrustEngineAdmissionRefinement"
  /\ CheckedFact = "Temporal package replay evidence is bound to this source revision"
  /\ PresentText(ArtifactHash)
  /\ PresentText(Assumptions)
  /\ PresentText(Configuration)
  /\ PresentText(ReplayInvocation)
  /\ PresentText(ResultStatus)

NativeEvidenceShapeComplete ==
  /\ NativeEvidenceRequiredFieldsPresent
  /\ ResultStatus \in PublicResultStatuses

TrustEngineAdmissionRefinement ==
  /\ ArtifactHash = ReplayRootSha256
  /\ RequestedAuthority = "active_callable"
  /\ AdmissionDisposition \in KnownAdmissionDispositions
  /\ AdmissionFailClosed \in BOOLEAN
  /\ ProductionSelected \in BOOLEAN
  /\ TyNativeActivate \in BOOLEAN
  /\ NativeInstallGateManifestPresent \in BOOLEAN

NativeInstallGateAccepted ==
  /\ NativeInstallGateManifestPresent
  /\ AdmissionDisposition = "installable"
  /\ AdmissionReasonCode = "none"
  /\ InstallAuthority # "none"
  /\ AdmissionFailClosed = FALSE
  /\ ProductionSelected = TRUE
  /\ TyNativeActivate = TRUE
  /\ PresentText(ReplayRootSha256)
  /\ PresentText(ProofReportSha256)
  /\ PresentText(AdmissionEvidenceSha256)

Init ==
  admission_state =
    IF NativeEvidenceShapeComplete
       /\ TrustEngineAdmissionRefinement
       /\ NativeInstallGateAccepted
    THEN "accepted"
    ELSE "blocked"

Next == UNCHANGED admission_state

TypeOK == admission_state \in {"accepted", "blocked"}

TyPackageReplayAccepted ==
  /\ admission_state = "accepted"
  /\ NativeEvidenceShapeComplete
  /\ TrustEngineAdmissionRefinement
  /\ NativeInstallGateAccepted

TyPackageReplayBlocked ==
  /\ admission_state = "blocked"
  /\ NativeEvidenceRequiredFieldsPresent
  /\ TrustEngineAdmissionRefinement
  /\ ~TyPackageReplayAccepted

====
