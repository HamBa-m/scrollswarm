"""
Unit tests for ScrollSwarm's lightweight stdlib diagnostic logging system.
"""

import importlib
import logging
import scrollswarm
from scrollswarm import ScrollSwarmEnv
from scrollswarm.geometry import _load_cpp_extension


def test_package_logger_uses_null_handler():
    """Verifies scrollswarm registers NullHandler without polluting caller log handlers."""
    pkg_logger = logging.getLogger("scrollswarm")
    assert any(isinstance(h, logging.NullHandler) for h in pkg_logger.handlers)


def test_env_lifecycle_logging_and_zero_hot_path_overhead(caplog):
    """
    Verifies lifecycle INFO/DEBUG logs on __init__/reset and zero log emissions
    during the high-frequency step() hot loop.
    """
    with caplog.at_level(logging.DEBUG, logger="scrollswarm"):
        env = ScrollSwarmEnv(num_agents=3, max_cycles=30, backend="numpy")
        init_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "agents=3" in r.message and "backend=numpy->numpy" in r.message
            for r in init_records
        )

        caplog.clear()
        env.reset(seed=42)
        reset_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("seed=42" in r.message for r in reset_records)

        # Hot loop: step() must emit zero log records
        caplog.clear()
        for _ in range(20):
            if not env.agents:
                break
            actions = {a: 1 for a in env.agents}
            env.step(actions)
        assert len(caplog.records) == 0
        env.close()


def test_cpp_extension_load_warning_vs_debug(tmp_path, monkeypatch, caplog):
    """
    Verifies broken compiled C++ binaries emit WARNING whereas an unbuilt
    extension emits only DEBUG.
    """

    def _fail_import(name: str):
        raise ImportError(f"simulated load failure for {name}")

    monkeypatch.setattr(importlib, "import_module", _fail_import)

    with caplog.at_level(logging.DEBUG, logger="scrollswarm.geometry"):
        assert _load_cpp_extension(geometry_dir=tmp_path) is None
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        assert any(
            r.levelno == logging.DEBUG and "not built" in r.message
            for r in caplog.records
        )

        caplog.clear()
        (tmp_path / "_geometry_cpp.cp312-win_amd64.pyd").write_bytes(b"")
        assert _load_cpp_extension(geometry_dir=tmp_path) is None
        assert any(
            r.levelno == logging.WARNING and "failed to load" in r.message
            for r in caplog.records
        )
