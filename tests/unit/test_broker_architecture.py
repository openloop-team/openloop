"""Architecture boundaries that keep the broker ledger unprivileged."""

import ast
from pathlib import Path


BROKER_ROOT = Path(__file__).parents[2] / "src" / "openloop" / "broker"
BROKER_RPC_ROOT = Path(__file__).parents[2] / "src" / "openloop" / "broker_rpc"
APP_MODULE = Path(__file__).parents[2] / "src" / "openloop" / "app.py"

BANNED_IMPORT_PREFIXES = (
    "docker",
    "openloop.app",
    "openloop.broker_runtime",
    "openloop.runtime",
    "openloop.sandbox",
    "openloop.tools",
    "openloop.workers",
    "openhands",
)

BANNED_IMPORT_PARTS = ("haproxy", "kms", "relay", "rpc", "worker")


def test_broker_modules_do_not_import_privileged_or_runtime_layers():
    violations = []
    for path in sorted(BROKER_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                lowered = name.lower()
                if lowered.startswith(BANNED_IMPORT_PREFIXES) or any(
                    part in lowered.split(".") for part in BANNED_IMPORT_PARTS
                ):
                    violations.append((path.name, node.lineno, name))
    assert violations == []


def test_broker_package_has_no_generic_public_mutation_escape_hatch():
    banned_names = {"save", "update", "upsert", "execute", "set_state"}
    violations = []
    for path in sorted(BROKER_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in banned_names:
                    violations.append((path.name, node.lineno, node.name))
    assert violations == []


def test_broker_rpc_imports_core_one_way_and_no_runtime_layers():
    banned = (
        "docker",
        "openhands",
        "openloop.broker_runtime",
        "openloop.tools",
        "openloop.workers",
    )
    violations = []
    for path in sorted(BROKER_RPC_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name.lower().startswith(banned):
                    violations.append((path.name, node.lineno, name))
    assert violations == []


def test_application_does_not_wire_privileged_broker_runtime_yet():
    tree = ast.parse(APP_MODULE.read_text(encoding="utf-8"), filename=str(APP_MODULE))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith("openloop.broker_runtime") for name in imports)


def test_broker_rpc_dispatch_can_reach_only_reviewed_ledger_reads_and_create():
    application = BROKER_RPC_ROOT / "application.py"
    tree = ast.parse(
        application.read_text(encoding="utf-8"), filename=str(application)
    )
    ledger_calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = node.func.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_ledger"
        ):
            ledger_calls.add(node.func.attr)
    assert ledger_calls == {
        "create_authorized_job",
        "inspect_job",
        "inspect_job_authorization",
    }


def test_broker_rpc_client_has_no_generic_public_call_escape_hatch():
    client = BROKER_RPC_ROOT / "client.py"
    tree = ast.parse(client.read_text(encoding="utf-8"), filename=str(client))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BrokerRpcClient"
    ]
    assert len(classes) == 1
    public_async_methods = {
        node.name
        for node in classes[0].body
        if isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_")
    }
    assert public_async_methods == {"create_job", "inspect_job"}
