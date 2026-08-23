"""Error-contract tests for the public API.

Internal faults (``AttributeError``, ``RuntimeError``, ``OSError``, ...) must
surface as opaque 500s, while halo-forge's own domain errors keep their 4xx
mapping and their caller-facing message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase

SCHEMA_ROUTE = "/api/public/annotation-schemas"


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Yield a TestClient whose server errors are returned instead of re-raised."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(database=database, review_storage_root=tmp_path / "reviews")
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    reset_store_for_tests(None)
    app = app_module.create_app(serve_frontend=False)
    # raise_server_exceptions=False makes the TestClient behave like a real ASGI
    # server: the 500 response the app produced is returned to the caller.
    with TestClient(app, raise_server_exceptions=False) as client:
        client.service = service  # type: ignore[attr-defined]
        yield client
    reset_store_for_tests(None)


def _fail_with(client: Any, monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> Any:
    """Point the annotation-schema route at a service call that raises ``exc``."""

    def _boom(payload: dict[str, Any]) -> dict[str, Any]:
        raise exc

    monkeypatch.setattr(client.service, "create_annotation_schema", _boom)
    return client.post(SCHEMA_ROUTE, json={"name": "irrelevant"})


def test_internal_attribute_error_returns_500_without_leaking_message(
    api_client: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A programming bug in a service must not be reported as a 409 conflict."""
    secret = "'NoneType' object has no attribute 'connection_string_9f3a'"
    with caplog.at_level(logging.ERROR, logger="halo_forge.public_api.app"):
        response = _fail_with(api_client, monkeypatch, AttributeError(secret))

    assert response.status_code == 500, response.text
    assert secret not in response.text
    assert response.json() == {"detail": "Internal server error"}
    assert any("Traceback" in record.exc_text for record in caplog.records if record.exc_text)


def test_internal_runtime_error_returns_500(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare RuntimeError is not a domain error and must be a 500."""
    response = _fail_with(api_client, monkeypatch, RuntimeError("sqlite database is locked"))

    assert response.status_code == 500, response.text
    assert "sqlite" not in response.text


def test_internal_os_error_returns_500(api_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filesystem failures must not be echoed back with a 4xx status."""
    response = _fail_with(api_client, monkeypatch, OSError("[Errno 13] /Users/secret/runs.db"))

    assert response.status_code == 500, response.text
    assert "/Users/secret" not in response.text


def test_domain_conflict_error_still_returns_409(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Genuine review-domain conflicts keep their 409 mapping and message."""
    from halo_forge.review_lab.errors import ReviewConflictError

    response = _fail_with(api_client, monkeypatch, ReviewConflictError("revision already exists"))

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "revision already exists"


def test_domain_validation_error_still_returns_400(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Domain validation errors keep their 400 mapping and message."""
    from halo_forge.review_lab.errors import ReviewValidationError

    response = _fail_with(api_client, monkeypatch, ReviewValidationError("modality is required"))

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "modality is required"


def test_domain_eligibility_error_still_returns_403(
    api_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protected-source rejections keep their 403 mapping."""
    from halo_forge.review_lab.errors import ReviewEligibilityError

    response = _fail_with(api_client, monkeypatch, ReviewEligibilityError("source is protected"))

    assert response.status_code == 403, response.text


def test_is_domain_error_discriminates_project_exceptions() -> None:
    """The predicate accepts halo-forge exceptions and rejects builtins."""
    pytest.importorskip("fastapi")
    from halo_forge.product_lab.service import ProductLabError
    from halo_forge.public_api.app import is_domain_error
    from halo_forge.review_lab.errors import ReviewConflictError

    assert is_domain_error(ReviewConflictError("x")) is True
    assert is_domain_error(ProductLabError("x")) is True
    assert is_domain_error(AttributeError("x")) is False
    assert is_domain_error(RuntimeError("x")) is False
    assert is_domain_error(MemoryError()) is False
    assert is_domain_error(OSError("x")) is False
