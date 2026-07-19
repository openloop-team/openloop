"""Architecture boundaries that keep the broker ledger unprivileged."""

import ast
import importlib.util
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "src"
OPENLOOP_ROOT = SOURCE_ROOT / "openloop"
BROKER_ROOT = OPENLOOP_ROOT / "broker"
BROKER_RPC_ROOT = OPENLOOP_ROOT / "broker_rpc"
APP_MODULE = OPENLOOP_ROOT / "app.py"
WIRING_ROOT = OPENLOOP_ROOT / "wiring"
CODING_WORKER_MODULES = (
    OPENLOOP_ROOT / "tools" / "coding_worker.py",
    OPENLOOP_ROOT / "workflows" / "coding_worker.py",
)

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


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _first_party_import_graph() -> dict[str, set[str]]:
    modules = {
        _module_name(path): path for path in sorted(OPENLOOP_ROOT.rglob("*.py"))
    }
    graph: dict[str, set[str]] = {}
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = (
                        module
                        if path.name == "__init__.py"
                        else module.rpartition(".")[0]
                    )
                    target = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                else:
                    target = node.module or ""
                targets.append(target)
                targets.extend(
                    f"{target}.{alias.name}"
                    for alias in node.names
                    if target and alias.name != "*"
                )
            for target in targets:
                parts = target.split(".")
                for end in range(1, len(parts) + 1):
                    candidate = ".".join(parts[:end])
                    if candidate in modules:
                        imported.add(candidate)
        graph[module] = imported
    return graph


def _reachable_imports(
    graph: dict[str, set[str]], roots: set[str]
) -> tuple[set[str], dict[str, str | None]]:
    parents: dict[str, str | None] = {root: None for root in roots}
    pending = list(roots)
    while pending:
        module = pending.pop()
        for imported in graph.get(module, set()):
            if imported in parents:
                continue
            parents[imported] = module
            pending.append(imported)
    return set(parents), parents


def _import_path(target: str, parents: dict[str, str | None]) -> str:
    path = []
    current: str | None = target
    while current is not None:
        path.append(current)
        current = parents[current]
    return " -> ".join(reversed(path))


def test_broker_runtime_cannot_transitively_import_legacy_docker_adapter():
    graph = _first_party_import_graph()
    roots = {
        module
        for module in graph
        if module == "openloop.broker_runtime"
        or module.startswith("openloop.broker_runtime.")
    }
    reachable, parents = _reachable_imports(graph, roots)
    legacy = "openloop.tools.openhands_docker"
    assert legacy not in reachable, _import_path(legacy, parents)


def test_relay_facade_does_not_import_legacy_docker_adapter():
    graph = _first_party_import_graph()
    assert "openloop.tools.openhands_docker" not in graph[
        "openloop.tools.openhands_relay"
    ]


def test_config_imports_openhands_default_from_neutral_profile():
    graph = _first_party_import_graph()
    imports = graph["openloop.config"]
    assert "openloop.openhands.runtime_profile" in imports
    assert "openloop.tools.openhands_docker" not in imports


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


def test_privileged_broker_runtime_is_confined_to_wiring_broker():
    # Step 4 wires the privileged broker runtime through a single reviewed
    # composition seam, `wiring/broker.py`. Every other wiring module and the
    # app shell must reach it only via that seam (`openloop.wiring.broker`),
    # never by importing `broker_control`/`broker_runtime` directly.
    seam = WIRING_ROOT / "broker.py"
    assert seam.exists(), "wiring/broker.py is the required broker composition seam"
    other_modules = [
        path
        for path in sorted(WIRING_ROOT.glob("*.py"))
        if path != seam
    ]
    imports = []
    for path in (APP_MODULE, *other_modules):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    assert not any(
        name.startswith(
            ("openloop.broker_control", "openloop.broker_runtime")
        )
        for name in imports
    )


def test_coding_workers_have_no_broker_start_or_runtime_wiring():
    violations = []
    for path in CODING_WORKER_MODULES:
        source = path.read_text(encoding="utf-8")
        for banned in (
            "broker_control",
            "broker_runtime",
            "start_segment",
        ):
            if banned in source:
                violations.append((path.name, banned))
    assert violations == []


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


def test_broker_rpc_dispatch_uses_only_the_narrow_coordinator_port():
    application = BROKER_RPC_ROOT / "application.py"
    tree = ast.parse(
        application.read_text(encoding="utf-8"), filename=str(application)
    )
    coordinator_calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = node.func.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_segment_coordinator"
        ):
            coordinator_calls.add(node.func.attr)
    assert coordinator_calls == {
        "start_segment",
        "inspect_running_access",
        "quiesce_segment",
        "release_segment",
        "finalize_job",
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
    assert public_async_methods == {
        "create_job",
        "finalize_job",
        "inspect_job",
        "quiesce_segment",
        "release_segment",
        "start_segment",
    }
