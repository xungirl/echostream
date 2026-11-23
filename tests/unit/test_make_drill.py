"""make_drill must turn a function body into `raise NotImplementedError` and keep the docstring."""
import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_drill", ROOT / "scripts" / "make_drill.py")
make_drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_drill)


@pytest.mark.parametrize("name", list(make_drill.DRILLS))
def test_every_drill_stubs_cleanly(name, tmp_path):
    f, fn, _, _ = make_drill.DRILLS[name]
    target = tmp_path / Path(f).name
    shutil.copy(ROOT / f, target)
    make_drill.stub(target, fn)
    src = target.read_text()
    compile(src, str(target), "exec")                     # still valid Python
    assert f'raise NotImplementedError("drill: implement {fn}")' in src
    node = make_drill.find(__import__("ast").parse(src), fn)
    assert len([n for n in node.body if not isinstance(n, __import__("ast").Expr)]) == 1


@pytest.mark.parametrize("name", list(make_drill.GO_DRILLS))
def test_every_go_drill_stubs_cleanly(name, tmp_path):
    f, fn, _, _ = make_drill.GO_DRILLS[name]
    target = tmp_path / Path(f).name
    shutil.copy(ROOT / f, target)
    before = target.read_text()
    make_drill.stub_go(target, fn)
    src = target.read_text()
    assert f'panic("drill: implement {fn}")' in src and src != before
    assert src.count("\nfunc ") == before.count("\nfunc ")             # nothing else removed
    gofmt = shutil.which("gofmt")
    if gofmt:                                                           # still valid Go syntax
        import subprocess
        assert subprocess.run([gofmt, "-e", str(target)], capture_output=True).returncode == 0
