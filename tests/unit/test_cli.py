"""CLI tests — the sealed-analysis operator staging path."""

import io
import subprocess
import tarfile

import pytest

from openloop.cli import main
from openloop.config import Settings


@pytest.fixture
def fake_store(monkeypatch):
    """Swap the durable input store for a recorder and force the pg backend."""

    class _FakeInputStore:
        instances = []

        def __init__(self, dsn):
            self.dsn = dsn
            self.staged = []
            self.setup_called = False
            self.closed = False
            _FakeInputStore.instances.append(self)

        async def setup(self):
            self.setup_called = True

        async def stage(self, manifest):
            self.staged.append(manifest)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "openloop.config.get_settings",
        lambda: Settings(
            memory_backend="postgres",
            database_url="postgresql://unused-in-tests",
        ),
    )
    monkeypatch.setattr(
        "openloop.analysis.postgres.PostgresInputStore", _FakeInputStore
    )
    return _FakeInputStore


def test_stage_files_builds_manifest_and_prints_the_invocation(
    tmp_path, fake_store, capsys
):
    (tmp_path / "sales.csv").write_text("amount\n42\n")
    (tmp_path / "regions.csv").write_text("region\nwest\n")

    rc = main([
        "analysis", "stage",
        str(tmp_path / "sales.csv"), str(tmp_path / "regions.csv"),
    ])

    assert rc == 0
    (store,) = fake_store.instances
    assert store.setup_called and store.closed
    (manifest,) = store.staged
    # The capability ref is generated (high-entropy), not operator-chosen.
    assert manifest.input_ref.startswith("staged:")
    assert len(manifest.input_ref) > len("staged:") + 20
    assert [f.name for f in manifest.files] == ["sales.csv", "regions.csv"]
    out = capsys.readouterr().out
    # The follow-up invocation is ready to paste with the generated ref in the
    # new inputs shape, and no job_id (a caller can't bind one).
    assert f'"input_ref": "{manifest.input_ref}"' in out
    assert '"source": "staged"' in out
    assert '"job_id"' not in out
    assert "analysis.report:write" in out


def test_stage_archive_stages_one_tarball_of_committed_content(
    tmp_path, fake_store
):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "committed.txt").write_text("hello\n")
    subprocess.run(["git", "add", "committed.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    (tmp_path / "uncommitted.txt").write_text("not in HEAD\n")

    rc = main([
        "analysis", "stage", "--archive", str(tmp_path),
    ])

    assert rc == 0
    (manifest,) = fake_store.instances[0].staged
    assert manifest.input_ref.startswith("staged:")
    (file,) = manifest.files
    assert file.name == f"{tmp_path.name}.tar"
    members = tarfile.open(fileobj=io.BytesIO(file.content)).getnames()
    assert "committed.txt" in members
    assert "uncommitted.txt" not in members  # git archive HEAD only


def test_stage_refuses_the_process_local_backend(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "openloop.config.get_settings",
        lambda: Settings(memory_backend="memory"),
    )
    (tmp_path / "data.csv").write_text("x\n")

    rc = main([
        "analysis", "stage",
        str(tmp_path / "data.csv"),
    ])

    assert rc == 1
    assert "MEMORY_BACKEND=postgres" in capsys.readouterr().err


def test_stage_with_nothing_to_stage_errors(capsys):
    rc = main(["analysis", "stage"])

    assert rc == 1
    assert "nothing to stage" in capsys.readouterr().err


def test_stage_rejects_duplicate_bare_names(tmp_path, fake_store, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "data.csv").write_text("x\n")
    (tmp_path / "b" / "data.csv").write_text("y\n")

    rc = main([
        "analysis", "stage",
        str(tmp_path / "a" / "data.csv"), str(tmp_path / "b" / "data.csv"),
    ])

    assert rc == 1
    assert "duplicate" in capsys.readouterr().err
    assert fake_store.instances == []  # nothing staged


def test_stage_missing_file_errors(tmp_path, fake_store, capsys):
    rc = main([
        "analysis", "stage",
        str(tmp_path / "absent.csv"),
    ])

    assert rc == 1
    assert "absent.csv" in capsys.readouterr().err
    assert fake_store.instances == []
