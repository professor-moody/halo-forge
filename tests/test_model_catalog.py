from __future__ import annotations


def test_model_catalog_filters_by_mode_backend_modality_provider_status():
    from halo_forge.models.catalog import list_models

    code = list_models({"mode": "raft", "backend": "mps", "modality": "code"})
    assert code
    assert all("raft" in item["trainer_support"] for item in code)
    assert all("mps" in item["backend_support"] for item in code)
    assert all("code" in item["modalities"] for item in code)

    liquid = list_models({"provider": "Liquid AI", "status": "experimental"})
    assert liquid
    assert all(item["provider"] == "Liquid AI" for item in liquid)
    assert all(item["status"] == "experimental" for item in liquid)


def test_get_model_returns_catalog_entry():
    from halo_forge.models.catalog import get_model

    item = get_model("Qwen/Qwen2.5-Coder-3B")
    assert item is not None
    assert item["provider"] == "Qwen"
    assert item["memory_tier"] == "small"
    assert item["estimated_memory_gb"] > 0
    assert item["risk_level"] in {"safe", "caveated", "experimental"}


def test_recommended_models_respects_backend():
    from halo_forge.models.catalog import recommended_models

    items = recommended_models(mode="sft", backend="mlx")
    assert items
    assert all("mlx" in item["backend_support"] for item in items)
    assert items[0]["recommended_first_run"] is True


def test_public_api_model_catalog_shape():
    from halo_forge.public_api.service import PublicApiService

    service = PublicApiService()
    payload = service.list_model_catalog(provider="Liquid AI")

    assert payload["catalog_version"]
    assert payload["total"] == len(payload["items"])
    assert payload["items"]
    assert payload["facets"]["providers"] == ["Liquid AI"]
    assert "risk_levels" in payload["facets"]


def test_public_api_training_models_use_catalog(monkeypatch):
    from halo_forge.public_api.service import PublicApiService

    class FakeBackend:
        name = "mps"

    monkeypatch.setattr("halo_forge.backend.get_backend", lambda: FakeBackend())
    service = PublicApiService()

    items = service.list_suggested_models(mode="raft", modality="code")
    assert items
    assert all("raft" in item["trainer_support"] for item in items)
    assert all("code" in item["modalities"] for item in items)


def test_quickstarts_and_templates_reference_catalog_models():
    from halo_forge.models.catalog import get_model
    from halo_forge.training.templates import TEMPLATES
    from ui.services.quickstart_presets import list_quickstart_presets

    missing: list[str] = []
    for template in TEMPLATES:
        if template.model_hint and get_model(template.model_hint) is None:
            missing.append(template.model_hint)
    for preset in list_quickstart_presets("training"):
        model = preset.values.get("model")
        if model and get_model(str(model)) is None:
            missing.append(str(model))

    assert sorted(set(missing)) == []
