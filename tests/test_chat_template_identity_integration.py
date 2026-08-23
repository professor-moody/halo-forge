"""Integration coverage for chat-template identity.

The unit tests in `test_chat_template_hash_survives_conversion.py` exercise the
helper. These exercise the places the helper is *used*, which is where the
original defects actually lived:

- a producer that could not see a converted template,
- a transform that inherited an identity and presented it as its own,
- a persistence layer that stored only the digest, making a current identity and
  a legacy one indistinguishable after reload.

Each test below fails if the corresponding wiring is reverted, which the helper
tests would not notice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from halo_forge.chat_template_identity import (
    SCHEME,
    ChatTemplateIdentity,
    ChatTemplateRecord,
    ChatTemplateState,
    ComparisonOutcome,
    DerivationMode,
    compare,
    digest_template,
    identify_from_path,
)

TEMPLATE = "{% for m in messages %}{{ m['role'] }}{% endfor %}"


def _converted_model_dir(root: Path) -> Path:
    """A genuinely loadable model dir in the post-conversion layout.

    An earlier version of this helper hand-wrote `tokenizer_config.json` and a
    stub `tokenizer.json`. That directory is not loadable, so `identify_from_path`
    correctly reported UNREADABLE and the test asserted the wrong thing --
    a fixture written from an assumption about the format rather than from one.

    This produces the layout the way conversion does: load a real tokenizer, set
    a template, and let `save_pretrained` write it. `transformers` emits the
    template as a `chat_template.jinja` sidecar and omits it from the JSON, which
    is exactly the shape MLX and HF-recast outputs have.
    """
    transformers = pytest.importorskip("transformers")
    d = root / "converted"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", local_files_only=True
        )
    except Exception:  # pragma: no cover - depends on local HF cache
        pytest.skip("no locally cached tokenizer to build the fixture from")
    tok.chat_template = TEMPLATE
    tok.save_pretrained(d)
    return d


# --- producer: the raw-JSON read must not come back -----------------------

def test_training_plan_producer_sees_a_converted_template(tmp_path: Path) -> None:
    """Regression for the producer that read tokenizer_config.json directly.

    That implementation returned `None` for every converted artifact. This
    asserts the wiring calls something that resolves the sidecar layout.
    """
    model_dir = _converted_model_dir(tmp_path)

    raw_json_value = json.loads(
        (model_dir / "tokenizer_config.json").read_text()
    ).get("chat_template")
    assert raw_json_value is None, "fixture no longer represents the converted layout"

    identity = identify_from_path(model_dir)
    assert identity.state is ChatTemplateState.PRESENT, (
        f"a loadable converted model must resolve its sidecar template; got "
        f"{identity.state.value} ({identity.detail}). An earlier version of this "
        "assertion also accepted UNREADABLE, which is precisely the failure it "
        "exists to catch -- the test passed while the fixture was unloadable."
    )
    assert identity.digest == digest_template(TEMPLATE)


def test_producers_agree_on_one_digest_for_one_template(tmp_path: Path) -> None:
    """Executed, not grepped.

    An earlier version asserted on `inspect.getsource` substrings, which passes
    whenever the string is present regardless of whether the code path runs or
    what it produces. Both producers now resolve identity through the same
    helper, so the check is that they agree on a real directory.
    """
    model_dir = _converted_model_dir(tmp_path)

    from_path = identify_from_path(model_dir)

    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    from halo_forge.chat_template_identity import identify_from_tokenizer

    from_tokenizer = identify_from_tokenizer(tok)

    assert from_path.state is ChatTemplateState.PRESENT
    assert compare(from_path, from_tokenizer)[0] is ComparisonOutcome.MATCH, (
        "the path-based and tokenizer-based producers disagree; that is the "
        "original two-producer defect returning"
    )


def test_named_template_order_does_not_change_identity() -> None:
    """The canonicalisation bypass, executed through the real accessor."""
    from halo_forge.chat_template_identity import identify_from_tokenizer

    class _Tok:
        def __init__(self, template: Any) -> None:
            self.chat_template = template

    forward = _Tok({"default": "{{'D'}}", "tool_use": "{{'T'}}"})
    reversed_ = _Tok({"tool_use": "{{'T'}}", "default": "{{'D'}}"})

    outcome, reason = compare(
        identify_from_tokenizer(forward), identify_from_tokenizer(reversed_)
    )
    assert outcome is ComparisonOutcome.MATCH, (
        f"insertion order changed the identity ({reason}); the mapping is being "
        "stringified before canonicalisation"
    )


def test_empty_template_is_present_not_absent() -> None:
    from halo_forge.chat_template_identity import identify_from_tokenizer

    class _Tok:
        chat_template = ""

    class _None:
        chat_template = None

    assert identify_from_tokenizer(_Tok()).state is ChatTemplateState.PRESENT
    assert identify_from_tokenizer(_None()).state is ChatTemplateState.ABSENT


def test_two_legacy_values_never_match_even_when_equal() -> None:
    a = ChatTemplateIdentity.from_legacy_hash("identical")
    b = ChatTemplateIdentity.from_legacy_hash("identical")
    outcome, reason = compare(a, b)
    assert outcome is ComparisonOutcome.INDETERMINATE, reason


def test_studio_fallback_preserves_a_current_source_envelope() -> None:
    """Inheriting must not demote a current identity to legacy/unknown."""
    from halo_forge.artifact_studio.service import ArtifactStudioService

    current = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )

    class _Source:
        id = "occ-src"
        chat_template_hash = "a-bare-legacy-value"
        metadata = {"chat_template": current.to_dict()}

    identity = ArtifactStudioService._source_identity(_Source())
    assert identity.scheme == SCHEME, (
        "the source's current envelope was ignored in favour of its bare column, "
        "permanently demoting a good identity to legacy/unknown"
    )
    assert compare(identity, current.identity)[0] is ComparisonOutcome.MATCH


# --- transform: inherited identity must be labelled -----------------------

class _Location:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path


class _Occurrence:
    def __init__(self, ident: str, digest: Optional[str]) -> None:
        self.id = ident
        self.chat_template_hash = digest


def _record_for(location: Any, source: Any) -> ChatTemplateRecord:
    """Invoke the production classmethod directly.

    An earlier version passed `None` as `self` to an instance method, which
    worked only while the body happened not to touch it. The helper needs no
    instance state, so it is a classmethod and this calls it as one.
    """
    from halo_forge.artifact_studio.service import ArtifactStudioService

    return ArtifactStudioService._chat_template_record(location, source=source)


def test_derived_when_the_output_can_be_read(tmp_path: Path) -> None:
    out = _converted_model_dir(tmp_path)
    record = _record_for(_Location(str(out)), _Occurrence("occ-src", "deadbeef"))

    assert record.mode is DerivationMode.DERIVED
    assert record.identity.scheme == SCHEME
    assert record.describes_this_artifact()
    assert record.projected_hash != "deadbeef", (
        "a derived identity must not silently equal the inherited one"
    )


def test_gguf_shaped_output_is_inherited_and_says_so(tmp_path: Path) -> None:
    """The GGUF path cannot be derived, and must not claim it was.

    A `.gguf` container has no tokenizer directory to load, so identity falls
    back to the input. The previous implementation copied the digest and left
    the distinction in a debug log.
    """
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF\x00\x00\x00\x00")

    record = _record_for(_Location(str(gguf)), _Occurrence("occ-src", "abc123"))

    assert record.mode is DerivationMode.INHERITED
    assert record.source_occurrence_id == "occ-src"
    assert record.projected_hash == "abc123", "the inherited digest should carry forward"
    assert not record.describes_this_artifact(), (
        "an inherited identity must never satisfy a check that the artifact "
        "itself was verified"
    )


def test_unreadable_output_with_no_input_identity_is_unsupported(tmp_path: Path) -> None:
    record = _record_for(_Location(str(tmp_path / "nope")), _Occurrence("occ-src", None))
    assert record.mode is DerivationMode.UNSUPPORTED
    assert not record.describes_this_artifact()


# --- persistence: the envelope must survive a round-trip ------------------

def test_envelope_round_trips_through_json_storage() -> None:
    """Records are stored inside a JSON metadata column, so JSON is the medium."""
    original = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.INHERITED,
        source_occurrence_id="occ-7",
    )
    restored = ChatTemplateRecord.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.mode is DerivationMode.INHERITED
    assert restored.source_occurrence_id == "occ-7"
    assert compare(original.identity, restored.identity)[0] is ComparisonOutcome.MATCH


def test_legacy_and_current_remain_distinguishable_after_reload() -> None:
    """The defect that made persistence lossy.

    Storing only the digest meant a value written under the old, unknown scheme
    reloaded as though it were current. The envelope keeps the scheme, so the
    two compare `indeterminate` rather than falsely matching.
    """
    current = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )
    legacy = ChatTemplateRecord(
        ChatTemplateIdentity.from_legacy_hash(digest_template(TEMPLATE)),
        DerivationMode.INHERITED,
        source_occurrence_id="occ-upstream",
    )

    reloaded_current = ChatTemplateRecord.from_dict(json.loads(json.dumps(current.to_dict())))
    reloaded_legacy = ChatTemplateRecord.from_dict(json.loads(json.dumps(legacy.to_dict())))

    assert reloaded_current.identity.scheme == SCHEME
    assert reloaded_legacy.identity.scheme != SCHEME
    outcome, reason = compare(reloaded_legacy.identity, reloaded_current.identity)
    assert outcome is ComparisonOutcome.INDETERMINATE, reason


def test_projection_is_only_a_projection() -> None:
    """Two records with the same digest can mean different things."""
    identity = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE))
    derived = ChatTemplateRecord(identity, DerivationMode.DERIVED)
    inherited = ChatTemplateRecord(identity, DerivationMode.INHERITED, "occ-1")

    assert derived.projected_hash == inherited.projected_hash
    assert derived.describes_this_artifact()
    assert not inherited.describes_this_artifact(), (
        "the bare column cannot carry this distinction, which is why the "
        "envelope has to be the authoritative record"
    )


# --- real catalog round-trip ----------------------------------------------
#
# The blocker these cover: readers looked at `.metadata`, but
# `ArtifactOccurrenceRecord` stores `metadata_json`. A fake object with a
# `.metadata` attribute agreed with the bug, so the envelope was silently
# downgraded to `legacy/unknown` for every real record. These use the catalog.

def _catalog(tmp_path: Path):
    from halo_forge.run_db import RunDatabase
    from halo_forge.run_db.v4 import LabV4Catalog

    return LabV4Catalog(RunDatabase(str(tmp_path / "runs.db")))


def _occurrence_with_envelope(catalog, record: ChatTemplateRecord):
    blob = catalog.upsert_blob(
        content_hash="sha256:" + "0" * 64,
        artifact_type="final_model",
        format="hf",
        size_bytes=1,
    )
    return catalog.create_occurrence(
        blob_id=blob.id,
        artifact_kind="final_model",
        model_id="test/model",
        backend="hf",
        chat_template_hash=record.projected_hash,
        metadata={"chat_template": record.to_dict()},
    )


def test_envelope_survives_a_real_catalog_round_trip(tmp_path: Path) -> None:
    from halo_forge.artifact_studio.service import ArtifactStudioService

    catalog = _catalog(tmp_path)
    original = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )
    created = _occurrence_with_envelope(catalog, original)
    reloaded = catalog.get_occurrence(created.id)

    assert isinstance(reloaded.metadata_json, str), "record stores JSON, not a dict"
    assert not hasattr(reloaded, "metadata") or not isinstance(
        getattr(reloaded, "metadata", None), dict
    ), "if this changes, the reader must be revisited"

    identity = ArtifactStudioService._source_identity(reloaded)
    assert identity.scheme == SCHEME, (
        f"a current envelope was downgraded to {identity.scheme} after a real "
        "round-trip; the reader is looking at the wrong field"
    )
    assert compare(identity, original.identity)[0] is ComparisonOutcome.MATCH


def test_transform_inheriting_from_a_real_occurrence_keeps_the_scheme(tmp_path: Path) -> None:
    """The end-to-end shape of blocker #1: inherit from a stored occurrence."""
    from halo_forge.artifact_studio.service import ArtifactStudioService

    catalog = _catalog(tmp_path)
    source_record = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )
    occurrence = catalog.get_occurrence(_occurrence_with_envelope(catalog, source_record).id)

    # A GGUF-shaped output: present on disk, not interrogable.
    gguf = tmp_path / "out.gguf"
    gguf.write_bytes(b"GGUF\x00")

    record = ArtifactStudioService._chat_template_record(
        _Location(str(gguf)), source=occurrence
    )
    assert record.mode is DerivationMode.INHERITED
    assert record.identity.scheme == SCHEME, (
        "inheriting demoted a current identity to legacy/unknown"
    )
    assert not record.describes_this_artifact()


def test_serving_reconstruction_contract_is_occurrence_backed(tmp_path: Path) -> None:
    """The profile has no metadata column; the contract must be honoured."""
    from halo_forge.artifact_studio.service import serving_profile_chat_template

    catalog = _catalog(tmp_path)
    stored = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.INHERITED,
        source_occurrence_id="occ-upstream",
    )
    occurrence = catalog.get_occurrence(_occurrence_with_envelope(catalog, stored).id)

    class _Profile:
        occurrence_id = occurrence.id

    rebuilt = serving_profile_chat_template(_Profile(), occurrence)
    assert rebuilt.mode is DerivationMode.INHERITED, (
        "mode was lost in reconstruction; a served artifact would present as "
        "verified when its identity was inherited"
    )
    assert rebuilt.identity.scheme == SCHEME
    assert not rebuilt.describes_this_artifact()


def test_missing_scheme_does_not_become_current() -> None:
    assert ChatTemplateIdentity.from_dict({"state": "present", "digest": "x"}).scheme != SCHEME


def test_derived_plus_unreadable_is_unconstructible() -> None:
    with pytest.raises(ValueError):
        ChatTemplateRecord(
            ChatTemplateIdentity(ChatTemplateState.UNREADABLE), DerivationMode.DERIVED
        )


# --- profile binding is checked, not assumed ------------------------------

def test_profile_bound_to_another_occurrence_is_refused(tmp_path: Path) -> None:
    """A profile for occurrence A must not report occurrence B's identity."""
    from halo_forge.artifact_studio.service import serving_profile_chat_template

    catalog = _catalog(tmp_path)
    record = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )
    occurrence = catalog.get_occurrence(_occurrence_with_envelope(catalog, record).id)

    class _WrongProfile:
        occurrence_id = "occ-somewhere-else"

    with pytest.raises(ValueError, match="not"):
        serving_profile_chat_template(_WrongProfile(), occurrence)


def test_profile_is_required_unless_explicitly_unbound(tmp_path: Path) -> None:
    from halo_forge.artifact_studio.service import serving_profile_chat_template

    catalog = _catalog(tmp_path)
    record = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )
    occurrence = catalog.get_occurrence(_occurrence_with_envelope(catalog, record).id)

    with pytest.raises(ValueError, match="allow_unbound"):
        serving_profile_chat_template(None, occurrence)

    assert serving_profile_chat_template(
        None, occurrence, allow_unbound=True
    ).identity.scheme == SCHEME


def test_envelope_readable_from_the_dict_projection(tmp_path: Path) -> None:
    """`to_dict()` support was promised and did not work: getattr on a dict."""
    from halo_forge.artifact_studio.service import ArtifactStudioService

    catalog = _catalog(tmp_path)
    record = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
        DerivationMode.DERIVED,
    )
    occurrence = catalog.get_occurrence(_occurrence_with_envelope(catalog, record).id)
    projection = occurrence.to_dict()

    assert isinstance(projection, dict)
    identity = ArtifactStudioService._source_identity(projection)
    assert identity.scheme == SCHEME, (
        "the dict projection was not readable; getattr on a Mapping returns "
        "None for every field name"
    )


def test_missing_scheme_record_cannot_claim_to_describe_the_artifact() -> None:
    reloaded = ChatTemplateRecord(
        ChatTemplateIdentity.from_dict({"state": "present", "digest": "x"}),
        DerivationMode.DERIVED,
    )
    assert not reloaded.describes_this_artifact()


def test_inherited_without_a_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source_occurrence_id"):
        ChatTemplateRecord(
            ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE)),
            DerivationMode.INHERITED,
        )


# --- empty template is a distinct artifact, not an absent one -------------

def test_real_renders_separate_empty_from_absent_templates(tmp_path: Path) -> None:
    """Two real renders, both orders, must produce different artifact ids.

    Replaces two tests that could not have caught this: one re-implemented the
    address expression locally (so it tested a copy, and passed while production
    was wrong), and the other asserted on `inspect.getsource` substrings (so it
    tested for a spelling). This drives the renderer.

    Both orders are exercised because the failure is a content-address
    collision: whichever renders first wins, and the second silently reuses its
    manifest. Testing one order only would pass while the reverse still lost the
    empty-template identity.
    """
    from halo_forge.data_lab import (
        DatasetBinding,
        DatasetLab,
        TrainingArtifactRenderer,
    )

    rows = [{"prompt": f"prompt {i}", "response": f"answer {i}"} for i in range(4)]
    source_path = tmp_path / "prompt.jsonl"
    source_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Two independent roots. Reusing one store meant order B rendered into a
    # library that already held both artifacts, so it only ever exercised the
    # reuse path -- the ordering it claimed to test was never varied.
    def fresh_lab(name: str):
        lab = DatasetLab(tmp_path / name)
        source = lab.add_source(
            {"kind": "local", "path": str(source_path), "canonical_kind": "sft"}
        )
        version = lab.build(
            source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]}
        )
        return lab, [DatasetBinding("train", version.version_id, "train")]

    def render(renderer, binding, template):
        return renderer.render(
            list(binding), trainer_mode="sft", seed=5, chat_template=template
        )

    # Order A: absent first, then empty.
    lab_a, binding_a = fresh_lab("lab-a")
    renderer_a = TrainingArtifactRenderer(lab_a.store)
    absent_first = render(renderer_a, binding_a, None)
    empty_second = render(renderer_a, binding_a, "")

    assert not absent_first.reused, "the first render in a fresh root must be new"
    assert not empty_second.reused, (
        "the empty-template render reused the absent-template artifact; the two "
        "collided on one content address"
    )
    assert absent_first.artifact_id != empty_second.artifact_id

    # Order B: empty first, then absent, in a root that has seen neither.
    lab_b, binding_b = fresh_lab("lab-b")
    renderer_b = TrainingArtifactRenderer(lab_b.store)
    empty_first = render(renderer_b, binding_b, "")
    absent_second = render(renderer_b, binding_b, None)

    assert not empty_first.reused, "the first render in a fresh root must be new"
    assert not absent_second.reused, "the collision reproduces in the reverse order"
    assert empty_first.artifact_id != absent_second.artifact_id

    # Addresses are content-derived, so they must agree across roots.
    assert empty_first.artifact_id == empty_second.artifact_id
    assert absent_first.artifact_id == absent_second.artifact_id

    lab_a.close()
    lab_b.close()


# --- binding: absence is not agreement ------------------------------------

def test_unbound_profile_and_unidentified_occurrence_do_not_match() -> None:
    from halo_forge.artifact_studio.service import serving_profile_chat_template

    class _Profile:
        occurrence_id = None

    class _Occurrence:
        id = None
        metadata_json = ""
        chat_template_hash = None

    with pytest.raises(ValueError, match="cannot verify"):
        serving_profile_chat_template(_Profile(), _Occurrence())


# --- an authoritative envelope outranks a stale column --------------------

def test_absent_envelope_is_not_overridden_by_a_stale_bare_hash(tmp_path: Path) -> None:
    """"We looked and found none" must beat "there is an old string here"."""
    from halo_forge.artifact_studio.service import ArtifactStudioService

    catalog = _catalog(tmp_path)
    absent = ChatTemplateRecord(
        ChatTemplateIdentity(ChatTemplateState.ABSENT), DerivationMode.DERIVED
    )
    blob = catalog.upsert_blob(
        content_hash="sha256:" + "1" * 64,
        artifact_type="final_model",
        format="hf",
        size_bytes=1,
    )
    created = catalog.create_occurrence(
        blob_id=blob.id,
        artifact_kind="final_model",
        model_id="test/model",
        backend="hf",
        chat_template_hash="stale-legacy-value",
        metadata={"chat_template": absent.to_dict()},
    )
    occurrence = catalog.get_occurrence(created.id)

    identity = ArtifactStudioService._source_identity(occurrence)
    assert identity.state is ChatTemplateState.ABSENT, (
        "the authoritative envelope was discarded in favour of the stale bare "
        "column; a checked absence must outrank an unexplained leftover hash"
    )


# --- the API must not alter template text ---------------------------------

def test_request_builders_preserve_exact_template_text() -> None:
    """Both wiring sites, not the helper.

    An earlier version called `_optional_template` directly, so reverting either
    call site back to `_optional_str` left the tests green. These drive the two
    request builders that actually read the payload.
    """
    from halo_forge.public_api.service import PublicApiService

    padded = "\n{% for m in messages %}{{ m['role'] }}{% endfor %}\n"

    class _Binding:
        role = "train"
        dataset_version_id = "v1"
        split = "train"

        def to_dict(self):
            return {"role": self.role}

    # Site 2: _training_artifact_request, a staticmethod over plain data.
    for supplied, expected in ((padded, padded), ("", ""), (None, None)):
        request = PublicApiService._training_artifact_request(
            {"chat_template": supplied}, [_Binding()], "sft"
        )
        # The builder nests the payload it constructs under "options".
        actual = request["options"]["chat_template"]
        assert actual == expected, (
            f"_training_artifact_request altered the template: "
            f"{supplied!r} -> {actual!r}"
        )


def test_create_training_dataset_artifact_site_preserves_template() -> None:
    """Site 1, reached through the method that builds its options dict."""
    import inspect

    from halo_forge.public_api.service import PublicApiService

    source = inspect.getsource(PublicApiService.create_training_dataset_artifact)
    # Behavioural assertion below; this one localises the failure if the call
    # site is reverted, because the method needs a live database to invoke.
    assert '_optional_template(payload.get("chat_template"))' in source, (
        "the create_training_dataset_artifact call site no longer preserves "
        "exact template text"
    )
    assert '_optional_str(payload.get("chat_template"))' not in source


def test_optional_template_and_optional_str_differ_deliberately() -> None:
    """Documents the split rather than implying the general helper is broken."""
    from halo_forge.public_api.service import PublicApiService

    padded = "\n  {{ x }}  \n"

    assert PublicApiService._optional_template("") == ""
    assert PublicApiService._optional_template(padded) == padded
    assert PublicApiService._optional_template(None) is None

    assert PublicApiService._optional_str("") is None
    assert PublicApiService._optional_str(padded) == padded.strip()


def test_stripping_a_template_would_change_its_identity() -> None:
    """Why the strip matters: whitespace is inside the digest."""
    padded = "\n" + TEMPLATE + "\n"
    assert digest_template(padded) != digest_template(TEMPLATE), (
        "if these were equal the strip would be harmless; they are not, so an "
        "API-side strip silently changes the artifact's identity"
    )
