"""Pre-release smoke check: build, validate, and import the package.

Runs the following steps in order:

1. Build sdist and wheel via ``python -m build``.
2. Check the distributions with ``twine check``.
3. Install the wheel into a temporary virtual environment and verify that
   ``import rapidfuzz_collections`` succeeds.

Requires ``build`` and ``twine`` to be installed (both are in the ``dev`` extra).
Run from the repository root:

    python scripts/check_package.py
"""

import sys
from pathlib import Path
from shutil import rmtree
from subprocess import run
from tempfile import TemporaryDirectory


def _run(args: list[str], *, cwd: Path | None = None) -> None:
    result = run(args, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    """Build, validate, and smoke-import the package."""
    root = Path(__file__).parent.parent.resolve()
    dist_dir = root / "dist"

    if dist_dir.exists():
        rmtree(dist_dir)

    print("==> Building sdist and wheel...")
    _run([sys.executable, "-m", "build", "--sdist", "--wheel", str(root)])

    print("==> Checking distributions with twine...")
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    dist_files = [str(p) for p in wheels + sdists]
    _run([sys.executable, "-m", "twine", "check", *dist_files])

    print("==> Installing wheel into a temporary virtual environment...")
    if not wheels:
        print("ERROR: no wheel found in dist/", file=sys.stderr)
        sys.exit(1)

    wheel_path = wheels[0]
    with TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        _run([sys.executable, "-m", "venv", str(venv_dir)])

        venv_python = venv_dir / "Scripts" / "python.exe" if sys.platform == "win32" else venv_dir / "bin" / "python"
        _run([str(venv_python), "-m", "pip", "install", "--quiet", str(wheel_path)])

        print("==> Verifying import...")
        _run([str(venv_python), "-c", "import rapidfuzz_collections; print('OK:', rapidfuzz_collections.__name__)"])

    print("==> Package check passed.")


if __name__ == "__main__":
    main()
