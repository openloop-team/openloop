"""CLI tests — `openloop agents id issue` (durable agent identity minting).

The command stamps a minted ``uuid4().hex`` id into an existing hand-authored
agent YAML with a surgical text insert (comments and formatting preserved),
confirms before writing (``--yes`` opts out), keeps a present well-formed
unique id untouched, and refuses duplicates/malformed ids. The `agents/`
directory the file lives in is the uniqueness authority.
"""

import re
from pathlib import Path

import pytest

from openloop.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]

ID_RE = re.compile(r"^[0-9a-f]{32}$")
VALID_ID = "9f2c1d4e8a7b4c3d9e0f1a2b3c4d5e6f"


def _write_agent(path: Path, *, name="foo", id=None) -> Path:
    id_line = f"  id: {id}\n" if id else ""
    path.write_text(
        "# hand-authored agent config\n"
        "apiVersion: openloop.team/v1alpha1\n"
        "kind: Agent\n"
        "metadata:\n"
        "  # the human handle\n"
        f"  name: {name}\n"
        "  workspace: acme\n"
        f"{id_line}"
        "\n"
        "spec:\n"
        "  model_policy:\n"
        "    default: m   # trailing comment\n"
    )
    return path


def _stamped_id(path: Path) -> str:
    (match,) = re.findall(r"^  id: (\S+)$", path.read_text(), re.MULTILINE)
    return match


def test_yes_mints_and_inserts_under_metadata(tmp_path, capsys):
    file = _write_agent(tmp_path / "foo.yaml")

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 0
    minted = _stamped_id(file)
    assert ID_RE.match(minted)
    # Inserted after the last metadata child, before the blank line.
    assert f"  workspace: acme\n  id: {minted}\n\nspec:" in file.read_text()
    out = capsys.readouterr().out
    assert f"issued identity {minted}" in out
    assert "commit" in out  # the reminder that identity must land in git


def test_insert_preserves_comments_and_formatting(tmp_path):
    # A fixture with real comment density: the committed brand-designer file.
    original = (REPO_ROOT / "agents" / "brand-designer.yaml").read_text()
    without_id = "".join(
        line
        for line in original.splitlines(keepends=True)
        if not line.startswith("  id: ")
    )
    file = tmp_path / "brand-designer.yaml"
    file.write_text(without_id)

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 0
    stamped = file.read_text()
    # Everything except the freshly minted id line is byte-identical — no
    # safe_dump round-trip stripping comments or reflowing formatting.
    assert "".join(
        line
        for line in stamped.splitlines(keepends=True)
        if not line.startswith("  id: ")
    ) == without_id
    assert ID_RE.match(_stamped_id(file))


def test_tty_confirm_yes_writes(tmp_path, monkeypatch, capsys):
    file = _write_agent(tmp_path / "foo.yaml")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    rc = main(["agents", "id", "issue", "-f", str(file)])

    assert rc == 0
    assert ID_RE.match(_stamped_id(file))
    # The preview names the permanence contract before asking.
    assert "permanent identity" in capsys.readouterr().out


def test_tty_confirm_no_aborts_file_untouched(tmp_path, monkeypatch):
    file = _write_agent(tmp_path / "foo.yaml")
    before = file.read_bytes()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    rc = main(["agents", "id", "issue", "-f", str(file)])

    assert rc == 1
    assert file.read_bytes() == before


def test_tty_default_empty_answer_aborts(tmp_path, monkeypatch):
    # [y/N] — anything but an explicit yes aborts.
    file = _write_agent(tmp_path / "foo.yaml")
    before = file.read_bytes()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "")

    rc = main(["agents", "id", "issue", "-f", str(file)])

    assert rc == 1
    assert file.read_bytes() == before


def test_non_tty_without_yes_aborts(tmp_path, monkeypatch, capsys):
    # Never hang on stdin, never write silently.
    file = _write_agent(tmp_path / "foo.yaml")
    before = file.read_bytes()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    rc = main(["agents", "id", "issue", "-f", str(file)])

    assert rc == 1
    assert file.read_bytes() == before
    assert "--yes" in capsys.readouterr().err


def test_present_unique_id_is_kept(tmp_path, capsys):
    # Idempotent-keep: safe to re-run, or to sweep a whole directory.
    file = _write_agent(tmp_path / "foo.yaml", id=VALID_ID)
    before = file.read_bytes()

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 0
    assert file.read_bytes() == before
    assert f"already has identity {VALID_ID}" in capsys.readouterr().out


def test_duplicate_id_among_siblings_refuses(tmp_path, capsys):
    file = _write_agent(tmp_path / "foo.yaml", id=VALID_ID)
    _write_agent(tmp_path / "other.yaml", name="other", id=VALID_ID)

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "already used by" in err and "other.yaml" in err


def test_malformed_existing_id_refuses(tmp_path, capsys):
    file = _write_agent(tmp_path / "foo.yaml", id="not-an-identity")
    before = file.read_bytes()

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 1
    assert file.read_bytes() == before
    assert "not a valid identity" in capsys.readouterr().err


def test_name_collision_with_sibling_refuses(tmp_path, capsys):
    # Minting an id must not launder a name the loader would reject anyway.
    file = _write_agent(tmp_path / "foo.yaml")
    _write_agent(tmp_path / "other.yaml", name="foo", id=VALID_ID)
    before = file.read_bytes()

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 1
    assert file.read_bytes() == before
    assert "duplicate agent name" in capsys.readouterr().err


def test_file_without_metadata_mapping_refuses(tmp_path, capsys):
    file = tmp_path / "foo.yaml"
    file.write_text("just: a scalar mapping\n")

    rc = main(["agents", "id", "issue", "-f", str(file), "--yes"])

    assert rc == 1
    assert "metadata" in capsys.readouterr().err


# --- `agents apply` coordination: apply detects, `id issue` fixes ---


def test_apply_on_an_id_less_file_points_at_id_issue(tmp_path, capsys):
    file = _write_agent(tmp_path / "foo.yaml")

    rc = main(["agents", "apply", "-f", str(file)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "no issued identity" in err
    assert f"openloop agents id issue -f {file}" in err


def test_apply_on_a_malformed_id_points_at_id_issue(tmp_path, capsys):
    file = _write_agent(tmp_path / "foo.yaml", id="not-an-identity")

    rc = main(["agents", "apply", "-f", str(file)])

    assert rc == 1
    assert "openloop agents id issue" in capsys.readouterr().err


def test_apply_on_a_stamped_file_succeeds_unchanged(tmp_path, capsys):
    file = _write_agent(tmp_path / "foo.yaml", id=VALID_ID)

    rc = main(["agents", "apply", "-f", str(file)])

    assert rc == 0
    assert "ok: agent 'foo'" in capsys.readouterr().out


def test_apply_keeps_the_generic_message_for_other_errors(tmp_path, capsys):
    file = tmp_path / "foo.yaml"
    file.write_text(
        "apiVersion: openloop.team/v1alpha1\nkind: Agent\n"
        f"metadata: {{name: foo, workspace: acme, id: {VALID_ID}}}\n"
        "spec: {}\n"  # no model_policy — a non-identity config error
    )

    rc = main(["agents", "apply", "-f", str(file)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "no issued identity" not in err
    assert "error:" in err


@pytest.mark.parametrize("name", ["dev-platform.yaml", "brand-designer.yaml"])
def test_committed_agent_files_already_have_identity(name, capsys):
    # Dogfood check: the repo's real agent files are stamped, so a re-run is
    # the idempotent keep path.
    rc = main([
        "agents", "id", "issue", "-f", str(REPO_ROOT / "agents" / name), "--yes"
    ])

    assert rc == 0
    assert "already has identity" in capsys.readouterr().out
