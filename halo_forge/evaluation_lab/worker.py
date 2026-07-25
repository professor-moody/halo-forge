"""Detached worker entry point for persistent CLI evaluation jobs."""

from __future__ import annotations

import argparse

from halo_forge.run_db import get_database

from .service import EvaluationLabService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one queued Halo Forge evaluation")
    parser.add_argument("evaluation_id")
    args = parser.parse_args()

    service = EvaluationLabService(get_database())
    try:
        completed = service.jobs.run_queued(args.evaluation_id)
        return 0 if completed.status == "completed" else 1
    finally:
        service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
