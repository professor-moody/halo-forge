from halo_forge.spec_registry import (
    get_spec_descriptor,
    serialized_spec_descriptors,
    validate_structured_spec,
)


def test_dataset_recipe_descriptors_cover_guided_steps():
    descriptors = serialized_spec_descriptors("dataset_recipe_step")
    assert {value["id"] for value in descriptors} == {
        "map",
        "validate",
        "filter",
        "dedup",
        "score",
        "split",
        "failure_mining",
    }
    assert get_spec_descriptor("dataset-recipe-step", "dedup").version == "1"


def test_structured_recipe_validation_reuses_recipe_contract():
    valid = validate_structured_spec(
        "dataset_recipe_step",
        "filter",
        {"field": "metadata.topic", "op": "eq", "value": "audio"},
    )
    assert valid["valid"] is True
    assert valid["value"]["kind"] == "filter"

    invalid = validate_structured_spec(
        "dataset_recipe_step",
        "filter",
        {"field": "metadata.topic", "op": "arbitrary_python"},
    )
    assert invalid["valid"] is False
    assert "Unsupported safe filter operator" in invalid["errors"][0]["message"]


def test_benchmark_descriptor_requires_adapter_specific_fields():
    result = validate_structured_spec(
        "benchmark_suite_item",
        "item",
        {"id": "gsm-example", "adapter_id": "lm-eval"},
    )
    assert result == {
        "valid": False,
        "value": {"id": "gsm-example", "adapter_id": "lm-eval"},
        "errors": [{"field": "task", "message": "Task is required for lm-eval items"}],
    }


def test_acquisition_and_annotation_descriptors_validate_domain_contracts():
    acquisition = serialized_spec_descriptors("acquisition_strategy")
    assert {value["id"] for value in acquisition} >= {
        "regression",
        "low_margin",
        "diversity",
        "random",
    }
    invalid_diversity = validate_structured_spec(
        "acquisition_strategy", "diversity", {"quota": 20}
    )
    assert invalid_diversity["valid"] is False

    annotation = validate_structured_spec(
        "annotation_task",
        "categorical",
        {"modality": "text", "labels": ["correct", "incorrect"]},
    )
    assert annotation["valid"] is True
    assert annotation["value"]["definition"]["output_adapter_id"] == "metadata.v1"
