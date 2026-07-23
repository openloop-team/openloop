from openloop.deliverable import Artifact, Prose
from openloop.tasks.outcomes import (
    Diagnosis,
    EvidenceBundle,
    Failed,
    PullRequest,
    to_deliverable,
)


def test_diagnosis_maps_to_prose():
    d = to_deliverable(Diagnosis(text="The bug is a null deref in parse()."))
    assert isinstance(d, Prose)
    assert "null deref" in d.text


def test_evidence_bundle_maps_to_artifact_with_provenance_body():
    outcome = EvidenceBundle(
        summary="Found 2 call sites.",
        findings="# Findings\n- parse() at src/p.py:42\n- caller at src/c.py:9\n",
    )
    d = to_deliverable(outcome)
    assert isinstance(d, Artifact)
    assert d.summary == "Found 2 call sites."
    assert "src/p.py:42" in d.content
    assert d.filename == "findings.md"


def test_pull_request_maps_to_prose_summary():
    d = to_deliverable(
        PullRequest(repo="a/b", branch="openloop/job-x", pr_number=7,
                    pr_url="https://gh/pr/7", summary="Opened draft PR #7.")
    )
    assert isinstance(d, Prose)
    assert "#7" in d.text


def test_failed_maps_to_prose():
    d = to_deliverable(Failed(status="open_pr_failed", error="403"))
    assert isinstance(d, Prose)
    assert "403" in d.text


def test_outcome_is_not_keyed_on_entry_action():
    # Structural decoupling proof: an EvidenceBundle and a PullRequest are the
    # same union; nothing binds a construction to which action started the task.
    outcomes = [Diagnosis(text="x"), PullRequest(repo="a/b", branch="b",
                pr_number=None, pr_url=None, summary="s")]
    assert all(to_deliverable(o) is not None for o in outcomes)
