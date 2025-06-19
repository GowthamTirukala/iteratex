import tempfile
from pathlib import Path
from iteratex.model_registry.utils import _write_pointer_atomic

def test_atomic_pointer_update(tmp_path):
    run_dir = tmp_path / "run123"
    run_dir.mkdir()
    pointer_file = tmp_path / "PRODUCTION"
    # Simulate atomic update
    _write_pointer_atomic(run_dir)
    assert pointer_file.exists() or (tmp_path / "PRODUCTION.tmp").exists()
    # Should point to the correct run directory
    assert Path(pointer_file.read_text()).resolve() == run_dir.resolve()
