from __future__ import annotations

import os
import subprocess
import sys


def test_cli_returns_nonzero_when_stage2_abort_is_raised() -> None:
    script = """
import sys
import fortran2rust.cli as cli


def _fail_stage2(_args):
    raise RuntimeError('Stage 2 Fortran benchmark executable did not run correctly')


cli._run_non_interactive = _fail_stage2
sys.argv = ['fortran2rust', '--non-interactive']
sys.exit(cli.main())
"""
    env = os.environ.copy()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    env['PYTHONPATH'] = src_dir + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(
        [sys.executable, '-c', script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert 'Stage 2 Fortran benchmark executable did not run correctly' in result.stderr


def test_cli_returns_zero_when_non_interactive_completes() -> None:
    script = """
import sys
import fortran2rust.cli as cli


def _ok_stage2(_args):
    return None


cli._run_non_interactive = _ok_stage2
sys.argv = ['fortran2rust', '--non-interactive']
sys.exit(cli.main())
"""
    env = os.environ.copy()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    env['PYTHONPATH'] = src_dir + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(
        [sys.executable, '-c', script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
