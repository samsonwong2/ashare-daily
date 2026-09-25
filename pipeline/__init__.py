"""One-shot simple-pulse mean pipeline orchestrator.

Split out of ``从更新txt到最终monitor_最短命令清单_20260403.py`` in P2-1.
See ``docs/PIPELINE_ORCHESTRATOR.md`` for operator-facing documentation.

Module layout:

    constants     - defaults + mutable _DRY_RUN / _LOG_DIR flags
    infra         - file lock, stage filename sanitizer, _run_command
    paths         - _resolve_paths + filename helpers
    data_prep     - code list, qlib raw panel, AkShare spot append
    route_a       - Route A full / rolling-refresh + manifest
    bridge        - bridge.py invocation
    risk_controls - apply_risk_controls (L2 cap + L3 regime/vol/cluster + anchor)
    stages        - run_backtest, run_monitor
    cli           - build_parser, validate_args, main
"""
