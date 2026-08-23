"""V10 guidance, semantic preview, and corpus-scenario contracts."""

from __future__ import annotations

import pytest

from halo_forge.own_data.guidance import (
    advise_scenarios,
    build_readiness_report,
    guided_example_descriptors,
    semantic_previews,
)
from halo_forge.own_data.mapping import build_preparation_plan
from halo_forge.own_data.models import ScenarioAdviceRequest
from halo_forge.own_data.registry import TRAINING_SCENARIOS


def test_plain_language_advisor_explains_corpus_fit_without_selecting_it():
    result = advise_scenarios(
        ScenarioAdviceRequest(
            goal="Adapt a model to our Markdown and PDF manuals",
            modality="text",
            source_layout="pdf",
        )
    ).to_dict()

    first = result["recommendations"][0]
    assert first["scenario_id"] == "corpus-adaptation"
    assert first["scenario_revision_id"] == "corpus-adaptation@1"
    assert first["requires_confirmation"] is True
    assert first["why_fit"]
    assert any("pdf" in reason.lower() for reason in first["why_fit"])
    assert result["requires_confirmation"] is True
    assert "never selects" in result["explanation"]


def test_field_evidence_distinguishes_preference_from_instruction_sft():
    result = advise_scenarios(
        {
            "goal": "Train from reviewed winner and loser answers",
            "modality": "text",
            "source_layout": "jsonl",
            "source_fields": ["prompt", "chosen", "rejected"],
        }
    ).to_dict()
    assert result["recommendations"][0]["scenario_id"] == "preference-pairs"
    assert result["recommendations"][0]["confidence"] == "high"
    assert {
        "prompt matches prompt",
        "chosen matches chosen",
        "rejected matches rejected",
    } <= set(result["recommendations"][0]["why_fit"])


def test_working_example_gallery_covers_every_verified_scenario():
    descriptors = guided_example_descriptors()
    covered = {item.scenario_id for item in descriptors}
    expected = {
        item.id
        for item in TRAINING_SCENARIOS.list(include_unavailable=False)
    }
    assert covered == expected
    corpus = next(item for item in descriptors if item.scenario_id == "corpus-adaptation")
    assert corpus.fixture_format == "markdown"
    assert corpus.trainer_modes == ("cpt",)
    assert "document" in corpus.expected_outcome.lower()


def test_corpus_defaults_align_recipe_and_reviewed_split_summary():
    scenario = TRAINING_SCENARIOS.get("corpus-adaptation")
    kinds = [step["kind"] for step in scenario.default_recipe["steps"]]
    assert kinds == [
        "map",
        "normalize",
        "document_clean",
        "document_filter",
        "validate",
        "dedup",
        "dedup",
        "split",
        "contamination",
    ]
    plan = build_preparation_plan(
        {
            "id": "inspection-corpus",
            "total_records": 10,
            "preview": [
                {
                    "document_id": "doc-1",
                    "document_hash": "hash-1",
                    "text": "# Heading\n\nBody.",
                    "source_ref": "guide.md",
                }
            ],
        },
        {
            "version": 2,
            "scenario_revision_id": scenario.revision_id,
            "confirmed": True,
            "mappings": {
                "document_id": {"kind": "direct", "source": "document_id"},
                "document_hash": {
                    "kind": "direct",
                    "source": "document_hash",
                },
                "text": {"kind": "direct", "source": "text"},
                "source_ref": {"kind": "direct", "source": "source_ref"},
            },
        },
    )
    assert plan.split_policy == {
        "method": "grouped",
        "group_field": "source_ref",
        "group_by_asset_hash": False,
        "ratios": {"train": 0.9, "validation": 0.1},
        "seed": 42,
        "protected_splits": [],
    }


@pytest.mark.parametrize(
    ("schema", "canonical", "presentation_key"),
    [
        (
            "chat",
            {"messages": [{"role": "user", "content": "Hello"}]},
            "turns",
        ),
        (
            "preference",
            {"prompt": "P", "chosen": "A", "rejected": "B"},
            "chosen",
        ),
        (
            "tool",
            {
                "messages": [{"role": "assistant", "content": ""}],
                "tools": [{"name": "lookup"}],
            },
            "tools",
        ),
        (
            "vlm",
            {"image": "image.png", "prompt": "What?", "response": "A gauge."},
            "image",
        ),
        (
            "audio",
            {"audio": "clip.wav", "task": "transcribe", "transcript": "Hello"},
            "audio",
        ),
        (
            "corpus",
            {
                "document_id": "doc-1",
                "document_hash": "a" * 64,
                "text": "# Heading\n\nBody",
                "source_ref": "guide.md",
                "source_spans": [{"section": "Heading"}],
            },
            "text",
        ),
    ],
)
def test_semantic_preview_is_modality_aware(schema, canonical, presentation_key):
    result = semantic_previews(
        {
            "inspection_id": "inspection-1",
            "scenario_revision_id": "scenario@1",
            "items": [
                {
                    "ordinal": 3,
                    "source": {"raw": True},
                    "canonical": canonical,
                    "issues": [],
                }
            ],
        },
        canonical_schema=schema,
    )
    assert result["sampled"] is True
    assert result["items"][0]["kind"] == schema
    assert presentation_key in result["items"][0]["presentation"]
    assert result["items"][0]["provenance"]["inspection_id"] == "inspection-1"


def test_readiness_report_has_direct_actions_and_truthful_sampled_estimates():
    inspection = {
        "id": "inspection-1",
        "row_count": 20,
        "media_summary": {"referenced": 1, "missing": 1},
        "extraction_summary": {"failed": 1},
    }
    mapping = {
        "inspection_id": "inspection-1",
        "items": [
            {
                "ordinal": 0,
                "source": {"prompt": "ok", "answer": "yes"},
                "canonical": {"prompt": "ok", "response": "yes"},
                "issues": [],
            },
            {
                "ordinal": 1,
                "source": {"prompt": "bad"},
                "canonical": {},
                "issues": [
                    {
                        "code": "required_field_missing",
                        "message": "response is required",
                        "severity": "error",
                    }
                ],
            },
        ],
    }
    preparation = {
        "recipe": {
            "steps": [
                {
                    "kind": "split",
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                }
            ]
        }
    }
    report = build_readiness_report(
        inspection,
        mapping,
        preparation,
        canonical_schema="sft",
        scenario_revision_id="instruction-sft@1",
    ).to_dict()

    assert report["ready"] is False
    assert report["sampled"] is True
    assert report["summary"]["token_count_is_estimated"] is True
    assert report["rejected_examples"][0]["ordinal"] == 1
    blocker_action_ids = {item["action_id"] for item in report["blockers"]}
    action_ids = {item["id"] for item in report["actions"]}
    assert blocker_action_ids <= action_ids
    assert {"inspect_rejected", "set_media_root"} <= action_ids
    assert report["extraction"]["failed"] == 1
    assert report["minimum_data"]["scientific_quality_threshold"] is None


def test_corpus_scenario_preserves_documents_and_has_no_trainer_test_binding():
    scenario = TRAINING_SCENARIOS.get("corpus-adaptation")
    assert scenario.canonical_schema == "corpus"
    assert scenario.trainer_modes == ("cpt",)
    assert {"txt", "markdown", "html", "pdf", "docx"} <= set(
        scenario.source_layouts
    )
    split = next(
        step for step in scenario.default_recipe["steps"] if step["kind"] == "split"
    )
    assert split == {
        "kind": "split",
        "method": "grouped",
        "group_field": "source_ref",
        "ratios": {"train": 0.9, "validation": 0.1},
        "seed": 42,
    }
    normalize = next(
        step
        for step in scenario.default_recipe["steps"]
        if step["kind"] == "normalize"
    )
    assert normalize["collapse_whitespace"] is False
    assert scenario.proof_budget["proof_run"] is False
