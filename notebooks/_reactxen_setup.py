from pathlib import Path
import sys
import subprocess
import importlib


def add_reactxen_src_to_path():
    """Add the ReActXen src directory to sys.path so notebooks can import `reactxen`.

    Returns the path added as a string.
    """
    notebooks_dir = Path(__file__).parent
    workspace_root = notebooks_dir.parent
    reactxen_src = workspace_root / "ReActXen" / "src"
    if not reactxen_src.exists():
        raise FileNotFoundError(
            f"Expected ReActXen/src at {reactxen_src} but it does not exist"
        )
    p = str(reactxen_src)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def verify_imports(module_names=None):
    """Try importing a list of modules from reactxen and return a dict of results.

    Does not execute heavy runtime code; only imports modules to verify they are reachable.
    """
    if module_names is None:
        module_names = [
            "reactxen",
            "reactxen.prebuilt.create_reactxen_agent",
            "reactxen.demo.hello_world_math",
            "reactxen.agents.react",
            "reactxen.utils.model_inference",
        ]

    results = {}
    for m in module_names:
        try:
            importlib.import_module(m)
            results[m] = (True, "ok")
        except Exception as e:
            results[m] = (False, str(e))
    return results


def install_editable(path=None, pip_executable=sys.executable):
    """Optionally install the ReActXen package in editable mode using pip.

    This will run: `python -m pip install -e <path>` where <path> defaults to the ReActXen folder.
    """
    notebooks_dir = Path(__file__).parent
    workspace_root = notebooks_dir.parent
    target = Path(path) if path else (workspace_root / "ReActXen")
    if not target.exists():
        raise FileNotFoundError(f"Install target does not exist: {target}")
    cmd = [pip_executable, "-m", "pip", "install", "-e", str(target)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    p = add_reactxen_src_to_path()
    print("Added:", p)
    res = verify_imports()
    for k, v in res.items():
        print(k, v)
