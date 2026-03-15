import pytest
from thinking import (
    ThinkingExtractor,
    ThinkingStripper,
    extract_thinking,
    strip_thinking,
)

# ── strip_thinking ────────────────────────────────────────────────────────────


def test_strip_thinking_removes_single_block() -> None:
    assert strip_thinking("<think>thoughts</think>answer") == "answer"


def test_strip_thinking_passthrough() -> None:
    assert strip_thinking("plain text") == "plain text"


def test_strip_thinking_content_before_and_after() -> None:
    assert strip_thinking("before<think>thinking</think>after") == "beforeafter"


def test_strip_thinking_multi_block() -> None:
    result = strip_thinking("<think>alpha</think>mid<think>beta</think>end")
    assert result == "midend"


def test_strip_thinking_strips_outer_whitespace() -> None:
    assert strip_thinking("  <think>x</think>  result  ") == "result"


def test_strip_thinking_no_tag_passthrough() -> None:
    text = "SELECT * FROM table WHERE id = 1"
    assert strip_thinking(text) == text


# ── extract_thinking ──────────────────────────────────────────────────────────


def test_extract_thinking_no_block_returns_none() -> None:
    clean, reasoning = extract_thinking("plain text")
    assert clean == "plain text"
    assert reasoning is None


def test_extract_thinking_single_block() -> None:
    clean, reasoning = extract_thinking("<think>r</think>a")
    assert clean == "a"
    assert reasoning == "r"


def test_extract_thinking_multiple_blocks_joined() -> None:
    clean, reasoning = extract_thinking("<think>first</think>mid<think>second</think>end")
    assert clean == "midend"
    assert reasoning == "first\nsecond"


def test_extract_thinking_strips_outer_whitespace_from_clean() -> None:
    clean, reasoning = extract_thinking("  <think>thought</think>  result  ")
    assert clean == "result"
    assert reasoning == "thought"


# ── ThinkingStripper ──────────────────────────────────────────────────────────


def test_thinking_stripper_full_block_in_one_chunk() -> None:
    stripper = ThinkingStripper()
    assert stripper.process("<think>thought</think>answer") == "answer"


def test_thinking_stripper_no_tag_passthrough() -> None:
    stripper = ThinkingStripper()
    assert stripper.process("no tags here") == "no tags here"


def test_thinking_stripper_tag_split_across_chunks() -> None:
    """Key regression: <think> tag split over two SSE chunks."""
    stripper = ThinkingStripper()
    result1 = stripper.process("hello <thi")
    result2 = stripper.process("nk>thoughts</think>world")
    assert result1 == "hello "
    assert result2 == "world"


def test_thinking_stripper_close_tag_split_across_chunks() -> None:
    """</think> tag split over two SSE chunks."""
    stripper = ThinkingStripper()
    result1 = stripper.process("<think>thoughts</thi")
    result2 = stripper.process("nk>answer")
    assert result1 == ""
    assert result2 == "answer"


def test_thinking_stripper_multiple_chunks_no_think() -> None:
    stripper = ThinkingStripper()
    parts = ["hello ", "world ", "foo"]
    output = "".join(stripper.process(part) for part in parts)
    assert output == "hello world foo"


def test_thinking_stripper_think_entirely_in_first_chunk() -> None:
    stripper = ThinkingStripper()
    result1 = stripper.process("<think>hidden</think>")
    result2 = stripper.process("visible")
    assert result1 == ""
    assert result2 == "visible"


# ── ThinkingExtractor ─────────────────────────────────────────────────────────


def test_thinking_extractor_full_block_in_one_chunk() -> None:
    extractor = ThinkingExtractor()
    clean, reasoning = extractor.process("<think>thought</think>answer")
    assert clean == "answer"
    assert reasoning == "thought"


def test_thinking_extractor_no_tag_passthrough() -> None:
    extractor = ThinkingExtractor()
    clean, reasoning = extractor.process("no tags")
    assert clean == "no tags"
    assert reasoning == ""


def test_thinking_extractor_tag_split_across_chunks() -> None:
    """<think> tag split over two SSE chunks."""
    extractor = ThinkingExtractor()
    clean1, reasoning1 = extractor.process("hello <thi")
    clean2, reasoning2 = extractor.process("nk>thoughts</think>world")
    assert clean1 == "hello "
    assert reasoning1 == ""
    assert clean2 == "world"
    assert reasoning2 == "thoughts"


def test_thinking_extractor_close_tag_split_across_chunks() -> None:
    """</think> tag split over two SSE chunks."""
    extractor = ThinkingExtractor()
    clean1, reasoning1 = extractor.process("<think>thoughts</thi")
    clean2, reasoning2 = extractor.process("nk>answer")
    assert clean1 == ""
    assert reasoning1 == ""
    assert clean2 == "answer"
    assert reasoning2 == "thoughts"


def test_thinking_extractor_multiple_chunks_no_think() -> None:
    extractor = ThinkingExtractor()
    parts = ["hello ", "world ", "foo"]
    clean_output = ""
    reasoning_output = ""
    for part in parts:
        clean, reasoning = extractor.process(part)
        clean_output += clean
        reasoning_output += reasoning
    assert clean_output == "hello world foo"
    assert reasoning_output == ""


def test_thinking_extractor_think_entirely_in_first_chunk() -> None:
    extractor = ThinkingExtractor()
    clean1, reasoning1 = extractor.process("<think>hidden</think>")
    clean2, reasoning2 = extractor.process("visible")
    assert clean1 == ""
    assert reasoning1 == "hidden"
    assert clean2 == "visible"
    assert reasoning2 == ""


# ── Custom tag tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("open_tag", "close_tag"),
    [
        ("<reasoning>", "</reasoning>"),
        ("[THINK]", "[/THINK]"),
    ],
)
def test_extract_thinking_custom_tags(open_tag: str, close_tag: str) -> None:
    text = f"{open_tag}internal{close_tag}answer"
    clean, reasoning = extract_thinking(text, open_tag, close_tag)
    assert clean == "answer"
    assert reasoning == "internal"


def test_strip_thinking_custom_tags() -> None:
    clean = strip_thinking(
        "<reasoning>thoughts</reasoning>result",
        open_tag="<reasoning>",
        close_tag="</reasoning>",
    )
    assert clean == "result"


def test_thinking_extractor_custom_tags() -> None:
    extractor = ThinkingExtractor(open_tag="[R]", close_tag="[/R]")
    clean, reasoning = extractor.process("[R]hidden[/R]visible")
    assert clean == "visible"
    assert reasoning == "hidden"


def test_thinking_stripper_custom_tags() -> None:
    stripper = ThinkingStripper(open_tag="[R]", close_tag="[/R]")
    assert stripper.process("[R]hidden[/R]visible") == "visible"


def test_thinking_extractor_custom_tags_split_across_chunks() -> None:
    extractor = ThinkingExtractor(open_tag="<reasoning>", close_tag="</reasoning>")
    clean1, reasoning1 = extractor.process("hello <reason")
    clean2, reasoning2 = extractor.process("ing>thoughts</reasoning>world")
    assert clean1 == "hello "
    assert reasoning1 == ""
    assert clean2 == "world"
    assert reasoning2 == "thoughts"


def test_thinking_stripper_custom_tags_split_across_chunks() -> None:
    stripper = ThinkingStripper(open_tag="<reasoning>", close_tag="</reasoning>")
    result1 = stripper.process("hello <reason")
    result2 = stripper.process("ing>thoughts</reasoning>world")
    assert result1 == "hello "
    assert result2 == "world"
