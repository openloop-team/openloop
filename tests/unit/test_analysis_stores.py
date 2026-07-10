"""Unit coverage for controller-owned Phase 1 analysis stores."""

import pytest

from openloop.analysis import (
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
