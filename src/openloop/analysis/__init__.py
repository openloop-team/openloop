"""Staged inputs, surface uploads, and sealed-analysis artifacts.

Controller-provisioned input files are addressed by capability ref, surface
uploads are metadata-only until an approved run fetches their bytes (Phase 4
lazy staging), and the sole read-out report is retained keyed by the analysis
job.  Tool arguments and workflow state carry only refs.
"""

from openloop.analysis.inputs import (
    ANALYSIS_ARGS_VERSION,
    AnalysisInput,
    AnalysisReportArgs,
    ExecutableAnalysisRequest,
    GithubInput,
    StagedInput,
    UploadInput,
)
from openloop.analysis.provision import (
    GithubProvisioner,
    ProvisionError,
    Provisioner,
    RequestIdentity,
    StagedProvisioner,
    UploadFetcher,
    UploadProvisioner,
    provision_inputs,
)
from openloop.analysis.store import (
    AnalysisAttempt,
    AnalysisAttemptStore,
    AnalysisArtifact,
    ArtifactStore,
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InputFile,
    InputManifest,
    InputStore,
    materialize_inputs,
)
from openloop.analysis.uploads import (
    InMemoryUploadStore,
    UploadRecord,
    UploadStore,
)

__all__ = [
    "ANALYSIS_ARGS_VERSION",
    "AnalysisAttempt",
    "AnalysisAttemptStore",
    "AnalysisArtifact",
    "AnalysisInput",
    "AnalysisReportArgs",
    "ArtifactStore",
    "ExecutableAnalysisRequest",
    "GithubInput",
    "GithubProvisioner",
    "InMemoryAnalysisAttemptStore",
    "InMemoryArtifactStore",
    "InMemoryInputStore",
    "InMemoryUploadStore",
    "InputFile",
    "InputManifest",
    "InputStore",
    "ProvisionError",
    "Provisioner",
    "RequestIdentity",
    "StagedInput",
    "StagedProvisioner",
    "UploadFetcher",
    "UploadInput",
    "UploadProvisioner",
    "UploadRecord",
    "UploadStore",
    "materialize_inputs",
    "provision_inputs",
]
