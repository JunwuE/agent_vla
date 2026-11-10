"""OS-enforced filesystem/network isolation for generated recovery code."""

import shutil
import sys
from pathlib import Path


def recovery_command(worker, source, directory, mode):
    if mode == "landlock":
        return [
            sys.executable,
            "-I",
            str(worker),
            str(source.resolve()),
            str(worker.with_name("worker_isolation.py").resolve()),
        ]
    if mode == "trusted":
        return [sys.executable, "-I", str(worker), str(source.resolve())]
    binary = shutil.which("bwrap")
    if binary is None:
        raise RuntimeError("bubblewrap is required; refusing unsandboxed generated code")
    args = [binary, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    roots = ["/usr", "/lib", "/lib64", str(Path(sys.prefix).resolve())]
    for root in dict.fromkeys(roots):
        if Path(root).exists():
            args += ["--ro-bind", root, root]
    if Path("/etc/ld.so.cache").exists():
        args += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    # Only this attempt's exported evidence is visible, read-only. Host logs, other attempts,
    # credentials, simulator sockets and the repository are absent from the mount namespace.
    args += [
        "--ro-bind",
        str(directory.resolve()),
        str(directory.resolve()),
        "--ro-bind",
        str(worker.resolve()),
        "/worker.py",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
        sys.executable,
        "-I",
        "/worker.py",
        str(source.resolve()),
    ]
    return args


def probe(mode):
    """Exercise confinement in a child; never restrict the calling supervisor."""
    import json
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vla-isolation-") as temp:
        root = Path(temp)
        source = root / "policy.py"
        source.write_text("def run(ctx):\n    return 'UNRESOLVED'\n")
        command = recovery_command(Path(__file__).with_name("recovery_worker.py"), source, root, mode)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **{k: os.environ[k] for k in ("PATH", "LANG", "LD_LIBRARY_PATH") if k in os.environ},
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        if result.returncode or not result.stdout.strip():
            raise RuntimeError(result.stderr[-500:] or "worker failed before isolation check")
        response = json.loads(result.stdout)
        if response.get("op") != "result":
            raise RuntimeError(response.get("error", "isolation probe failed"))
        return dict(mode=mode, available=True, os_confined=mode != "trusted")
