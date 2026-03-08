#!/usr/bin/env python3
"""Regression tests for training workflow UI polish and action hierarchy."""

from __future__ import annotations

from pathlib import Path

from ui.services.training_presentation import (
    build_launch_presentation,
    build_training_run_presentation,
)


def test_guided_recovery_is_primary_action_when_available():
    presentation = build_training_run_presentation(
        job_status="failed",
        quality_status="low_yield",
        quality_summary="Most samples were dropped before updates.",
        recovery_status="ready",
        recovery_action="Apply Suggested Fix",
        recovery_summary="Lower the threshold and retry.",
        failure_reason="No optimizer steps executed.",
        final_reason="no_optimizer_steps",
        has_launch_context=True,
        can_resume_latest=True,
        weights_updated=False,
    )

    assert presentation.primary_action is not None
    assert presentation.primary_action.id == "guided_fix"
    assert [action.id for action in presentation.secondary_actions] == [
        "review_details",
        "run_again",
        "resume_latest",
        "edit_config",
    ]


def test_completed_healthy_run_promotes_review_over_rerun():
    presentation = build_training_run_presentation(
        job_status="completed",
        quality_status="healthy",
        quality_summary="Weights updated and the run ended cleanly.",
        recovery_status="unavailable",
        recovery_action="",
        recovery_summary="",
        failure_reason="",
        final_reason="completed",
        has_launch_context=True,
        can_resume_latest=False,
        weights_updated=True,
    )

    assert presentation.primary_action is not None
    assert presentation.primary_action.id == "review_details"
    assert [action.id for action in presentation.secondary_actions] == [
        "run_again",
        "edit_config",
    ]


def test_completed_run_without_loaded_metrics_still_reads_as_completed():
    presentation = build_training_run_presentation(
        job_status="completed",
        quality_status="",
        quality_summary="",
        recovery_status="unavailable",
        recovery_action="",
        recovery_summary="This run did not produce a recognized recovery pattern.",
        failure_reason="",
        final_reason="",
        has_launch_context=True,
        can_resume_latest=False,
        weights_updated=False,
    )

    assert presentation.headline_status == "Run completed"
    assert presentation.primary_action is not None
    assert presentation.primary_action.id == "review_details"


def test_failed_run_without_guidance_defaults_to_edit_config():
    presentation = build_training_run_presentation(
        job_status="failed",
        quality_status="no_signal",
        quality_summary="Dataset formatting prevented useful updates.",
        recovery_status="advisory_only",
        recovery_action="Inspect dataset formatting",
        recovery_summary="Missing text fields prevented usable samples.",
        failure_reason="missing_text",
        final_reason="input_validation",
        has_launch_context=False,
        can_resume_latest=False,
        weights_updated=False,
    )

    assert presentation.primary_action is not None
    assert presentation.primary_action.id == "edit_config"


def test_launch_presentation_uses_outlook_and_recommendation():
    presentation = build_launch_presentation(
        mode_label="RAFT",
        quality_status="low_yield",
        quality_summary="This run is at risk of producing very little usable training signal.",
        suggested_adjustments=["Increase samples per prompt before launching."],
        yield_safety_note="Watch sample budget and thresholds to avoid starving updates.",
    )

    assert presentation.confidence_tone == "danger"
    assert presentation.headline_status == "RAFT launch is at risk of low signal"
    assert presentation.recommended_adjustment == "Increase samples per prompt before launching."


def test_training_monitor_and_results_sources_expose_polished_hierarchy():
    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "Ready to launch" in training_source
    assert "Recommended adjustment:" in training_source
    assert "Changed from" in training_source

    helper_source = Path("ui/services/training_presentation.py").read_text(encoding="utf-8")
    assert "Review Quality" in helper_source
    assert "Run Again" in helper_source
    assert "Edit Config" in helper_source

    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "Training Decision" in monitor_source
    assert "_current_training_presentation" in monitor_source
    assert "_trigger_monitor_action" in monitor_source

    results_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "Outcome, cause, and next step" in results_source
    assert "Recommended next step" in results_source
    assert "_trigger_training_action" in results_source
