# SPDX-License-Identifier: Apache-2.0
"""The ``climval check`` validation gate.

A configuration file declares checks (reference, candidate, variable,
alignment options and thresholds). :func:`run_file` runs them and returns a
:class:`GateReport` with an exit code: 0 when every check passes, 1 when a
threshold fails, 2 when the configuration or the data could not be used.

>>> from climval.gate import run_file
>>> report = run_file("climval.yml")
>>> report.exit_code
0
"""

from climval.gate.config import CheckSpec, GateConfig, InputSpec, load_config, parse_config
from climval.gate.run import (
    EXIT_ERROR,
    EXIT_FAIL,
    EXIT_PASS,
    CheckOutcome,
    GateReport,
    run,
    run_check,
    run_file,
)

__all__ = [
    "load_config",
    "parse_config",
    "run",
    "run_file",
    "run_check",
    "GateConfig",
    "CheckSpec",
    "InputSpec",
    "GateReport",
    "CheckOutcome",
    "EXIT_PASS",
    "EXIT_FAIL",
    "EXIT_ERROR",
]
