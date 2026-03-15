from unittest.mock import MagicMock, patch

from processors import Processor, get_processor

from tasks.sql.processor import SQLProcessor

# ── Processor base class ─────────────────────────────────────────────────────


def test_base_processor_preprocess_is_passthrough() -> None:
    proc = Processor()
    messages: list[dict[str, str]] = [{"role": "user", "content": "hi"}]
    assert proc.preprocess(messages, {}) is messages


def test_base_processor_postprocess_returns_content_unchanged() -> None:
    proc = Processor()
    content, attrs = proc.postprocess("SELECT 1")
    assert content == "SELECT 1"
    assert attrs == {}


def test_base_processor_postprocess_stream_returns_empty_attrs() -> None:
    proc = Processor()
    assert proc.postprocess_stream("SELECT 1") == {}


# ── Registry ─────────────────────────────────────────────────────────────────


def test_get_processor_returns_default_for_unregistered() -> None:
    proc = get_processor("unknown-adapter")
    assert isinstance(proc, Processor)
    assert not isinstance(proc, SQLProcessor)


def test_get_processor_returns_registered_processor() -> None:
    proc = get_processor("sql")
    assert isinstance(proc, SQLProcessor)


# ── SQLProcessor.preprocess ──────────────────────────────────────────────────


def test_sql_preprocess_empty_schema_returns_messages_unchanged() -> None:
    proc = SQLProcessor()
    messages: list[dict[str, str]] = [{"role": "user", "content": "hi"}]
    result = proc.preprocess(messages, {})
    assert result is messages


def test_sql_preprocess_missing_schema_returns_messages_unchanged() -> None:
    proc = SQLProcessor()
    messages: list[dict[str, str]] = [{"role": "user", "content": "hi"}]
    result = proc.preprocess(messages, {"other_key": "value"})
    assert result is messages


def test_sql_preprocess_adds_system_message() -> None:
    proc = SQLProcessor()
    mock_prompt = MagicMock()
    mock_prompt.format.return_value = "SQL system prompt"

    with patch("tasks.sql.processor.mlflow.genai.load_prompt", return_value=mock_prompt):
        messages: list[dict[str, str]] = [{"role": "user", "content": "query"}]
        result = proc.preprocess(messages, {"schema": "CREATE TABLE foo (id INT)"})

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "SQL system prompt"
    assert result[1] == messages[0]
    assert len(result) == 2


def test_sql_preprocess_prepends_to_existing_system_message() -> None:
    proc = SQLProcessor()
    mock_prompt = MagicMock()
    mock_prompt.format.return_value = "injected"

    with patch("tasks.sql.processor.mlflow.genai.load_prompt", return_value=mock_prompt):
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "existing"},
            {"role": "user", "content": "q"},
        ]
        result = proc.preprocess(messages, {"schema": "schema"})

    assert result[0]["role"] == "system"
    assert result[0]["content"] == "injected\n\nexisting"
    assert result[1] == messages[1]
    assert len(result) == 2


# ── SQLProcessor.postprocess ─────────────────────────────────────────────────


def test_sql_postprocess_valid_sql() -> None:
    proc = SQLProcessor()
    content, attrs = proc.postprocess("SELECT 1")
    assert content == "SELECT 1"
    assert attrs == {"sql_valid": True}


def test_sql_postprocess_invalid_sql() -> None:
    proc = SQLProcessor()
    content, attrs = proc.postprocess("not valid sql !!!")
    assert content == "not valid sql !!!"
    assert attrs == {"sql_valid": False}


# ── SQLProcessor.postprocess_stream ──────────────────────────────────────────


def test_sql_postprocess_stream_valid_sql() -> None:
    proc = SQLProcessor()
    attrs = proc.postprocess_stream("SELECT 1")
    assert attrs == {"sql_valid": True}


def test_sql_postprocess_stream_invalid_sql() -> None:
    proc = SQLProcessor()
    attrs = proc.postprocess_stream("not valid sql !!!")
    assert attrs == {"sql_valid": False}
