"""Unit coverage for controller-owned Phase 1 analysis stores."""

import pytest

from openloop.analysis import (
    InMemoryAnalysisAttemptStore,
    InMemoryArtifactStore,
    InMemoryInputStore,
    InputFile,
    InputManifest,
)


async def test_input_manifest_is_job_and_ref_scoped_and_materializes(tmp_path):
    store = InMemoryInputStore()
    manifest = InputManifest(
        job_id="job-1",
        input_ref="upload:abc",
        files=(InputFile("sales.csv", b"amount\n42\n"),),
    )
    await store.stage(manifest)

    assert await store.get("job-1", "other") is None
    restored = await store.get("job-1", "upload:abc")
    assert restored == manifest

    destination = tmp_path / "inputs"
    restored.materialize(destination)
    assert (destination / "sales.csv").read_bytes() == b"amount\n42\n"


@pytest.mark.parametrize("name", ["../secret", "nested/data.csv", "/abs.csv", "x\\y"])
def test_input_filenames_cannot_escape_the_inputs_directory(name):
    with pytest.raises(ValueError, match="bare filename"):
        InputFile(name, b"x")


async def test_artifact_store_uses_stable_job_key_and_defensive_byte_copy():
    store = InMemoryArtifactStore()
    body = bytearray(b"# first\n")
    ref = await store.put("job-1", body)
    body[:] = b"changed\n"

    assert ref == "analysis://job-1/report.md"
    assert (await store.get(ref)).body == b"# first\n"

    same_ref = await store.put("job-1", b"# replacement\n")
    assert same_ref == ref
    assert (await store.get(ref)).body == b"# replacement\n"


async def test_attempt_store_tracks_charge_and_settles_idempotently():
    store = InMemoryAnalysisAttemptStore()

    attempt, created = await store.begin("attempt-1", "job-1")
    assert created
    assert attempt.status == "started"

    charged = await store.charge(
        "attempt-1",
        cost_usd=0.42,
        prompt_tokens=120,
        completion_tokens=30,
    )
    assert charged.status == "charged"
    # Replaying the same observed provider telemetry is safe after a crash
    # between the charge checkpoint and usage settlement.
    assert await store.charge(
        "attempt-1",
        cost_usd=0.42,
        prompt_tokens=120,
        completion_tokens=30,
    ) is charged

    settled = await store.settle("attempt-1")
    assert settled.status == "settled"
    assert await store.settle("attempt-1") is settled

    existing, created = await store.begin("attempt-1", "job-1")
    assert not created
    assert existing is settled
    with pytest.raises(RuntimeError, match="different charge"):
        await store.charge(
            "attempt-1",
            cost_usd=0.43,
            prompt_tokens=120,
            completion_tokens=30,
        )


async def test_attempt_store_accepts_growing_cumulative_charges_until_settled():
    store = InMemoryAnalysisAttemptStore()
    await store.begin("attempt-1", "job-1")

    first = await store.charge(
        "attempt-1", cost_usd=0.20, prompt_tokens=100, completion_tokens=25
    )
    assert first.charged_at is not None
    # The iterative strategy re-charges the cumulative attempt total after
    # every completion: growth is legal while charged...
    grown = await store.charge(
        "attempt-1", cost_usd=0.40, prompt_tokens=200, completion_tokens=50
    )
    assert grown.status == "charged"
    assert grown.cost_usd == 0.40
    assert grown.charged_at == first.charged_at  # first observation time kept
    # ...an equal replay is a safe crash retry...
    assert (await store.charge(
        "attempt-1", cost_usd=0.40, prompt_tokens=200, completion_tokens=50
    )).cost_usd == 0.40
    # ...but a decrease means the caller lost track of observed spend.
    with pytest.raises(RuntimeError, match="decrease"):
        await store.charge(
            "attempt-1", cost_usd=0.30, prompt_tokens=200, completion_tokens=50
        )

    await store.settle("attempt-1")
    # After the ledger settle, growth would be spend that never reaches the
    # idempotent usage row — refuse anything but the settled figure.
    with pytest.raises(RuntimeError, match="different charge"):
        await store.charge(
            "attempt-1", cost_usd=0.60, prompt_tokens=300, completion_tokens=75
        )


async def test_unknown_attempt_cannot_be_charged():
    store = InMemoryAnalysisAttemptStore()
    await store.begin("attempt-unknown", "job-1")
    await store.mark_unknown("attempt-unknown", "interrupted before telemetry")

    with pytest.raises(RuntimeError, match="is unknown; cannot charge"):
        await store.charge(
            "attempt-unknown",
            cost_usd=0.42,
            prompt_tokens=120,
            completion_tokens=30,
        )
