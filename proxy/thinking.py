import re

from config import settings


def extract_thinking(
    text: str,
    open_tag: str = settings.thinking_open_tag,
    close_tag: str = settings.thinking_close_tag,
) -> tuple[str, str | None]:
    """
    Extract thinking blocks from generated text.

    Parameters
    ----------
    text : str
        Raw model output, possibly containing thinking blocks.
    open_tag : str
        Opening tag for thinking blocks.
    close_tag : str
        Closing tag for thinking blocks.

    Returns
    -------
    tuple[str, str | None]
        ``(clean_content, reasoning_content)`` where ``reasoning_content`` is
        ``None`` if no thinking blocks were present, or the joined block
        content if one or more were found.
    """
    pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
    thinking_blocks = re.findall(pattern, text, re.DOTALL)
    clean = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    if thinking_blocks:
        return clean, "\n".join(block.strip() for block in thinking_blocks)
    return clean, None


def strip_thinking(
    text: str,
    open_tag: str = settings.thinking_open_tag,
    close_tag: str = settings.thinking_close_tag,
) -> str:
    """
    Strip chain-of-thought blocks from generated text.

    Parameters
    ----------
    text : str
        Raw model output, possibly containing thinking blocks.
    open_tag : str
        Opening tag for thinking blocks.
    close_tag : str
        Closing tag for thinking blocks.

    Returns
    -------
    str
        Text with thinking blocks removed and outer whitespace stripped.
    """
    clean, _ = extract_thinking(text, open_tag, close_tag)
    return clean


class ThinkingExtractor:
    """
    Stateful streaming filter that separates thinking content from
    regular content across SSE chunks.

    Unlike ``ThinkingStripper`` which discards thinking content, this class
    captures it so callers can emit it as ``delta.reasoning_content``.
    """

    def __init__(
        self,
        open_tag: str = settings.thinking_open_tag,
        close_tag: str = settings.thinking_close_tag,
    ) -> None:
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._inside_think: bool = False
        self._buffer: str = ""

    def process(self, text: str) -> tuple[str, str]:
        """
        Process a chunk, returning clean and reasoning content separately.

        Parameters
        ----------
        text : str
            A chunk of generated text from a streaming SSE event.

        Returns
        -------
        tuple[str, str]
            ``(clean_content, reasoning_content)`` for this chunk. Either may
            be empty if the chunk contained only the other type of content.
        """
        clean_parts: list[str] = []
        reasoning_parts: list[str] = []
        self._buffer += text

        while self._buffer:
            if self._inside_think:
                end_idx = self._buffer.find(self._close_tag)
                if end_idx == -1:
                    # Check for partial close tag at end of buffer
                    has_partial = False
                    for tail_len in range(1, min(len(self._close_tag), len(self._buffer) + 1)):
                        candidate = self._buffer[-tail_len:]
                        if self._close_tag.startswith(candidate):
                            has_partial = True

                    if has_partial:
                        # Keep entire buffer — reasoning before the
                        # partial tag will be emitted once close tag
                        # is confirmed on the next call.
                        pass

                    else:
                        # No partial match — emit all as reasoning
                        reasoning_parts.append(self._buffer)
                        self._buffer = ""

                    break

                # Emit reasoning content before closing tag
                reasoning_parts.append(self._buffer[:end_idx])
                self._buffer = self._buffer[end_idx + len(self._close_tag) :]
                self._inside_think = False

            else:
                start_idx = self._buffer.find(self._open_tag)
                if start_idx == -1:
                    # Check for partial tag at end of buffer
                    partial_match = ""
                    for tail_len in range(1, min(len(self._open_tag), len(self._buffer) + 1)):
                        candidate = self._buffer[-tail_len:]
                        if self._open_tag.startswith(candidate):
                            partial_match = candidate

                    if partial_match:
                        clean_parts.append(self._buffer[: -len(partial_match)])
                        self._buffer = partial_match
                        break

                    clean_parts.append(self._buffer)
                    self._buffer = ""
                    break

                # Emit text before the open tag
                clean_parts.append(self._buffer[:start_idx])
                self._buffer = self._buffer[start_idx + len(self._open_tag) :]
                self._inside_think = True

        return "".join(clean_parts), "".join(reasoning_parts)


class ThinkingStripper:
    """
    Stateful streaming filter that removes thinking blocks across SSE chunks.

    Handles the case where open and close tags are split across multiple
    SSE data lines. Tracks whether we are currently inside a thinking block and buffers
    partial tag matches.
    """

    def __init__(
        self,
        open_tag: str = settings.thinking_open_tag,
        close_tag: str = settings.thinking_close_tag,
    ) -> None:
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._inside_think: bool = False
        self._buffer: str = ""

    def process(self, text: str) -> str:
        """
        Process a chunk of text, stripping thinking content.

        Parameters
        ----------
        text : str
            A chunk of generated text from a streaming SSE event.

        Returns
        -------
        str
            The text with any thinking-block content removed.
        """
        output = []
        self._buffer += text

        while self._buffer:
            if self._inside_think:
                end_idx = self._buffer.find(self._close_tag)
                if end_idx == -1:
                    # Check for partial close tag at end of buffer
                    partial_close = ""
                    for tail_len in range(1, min(len(self._close_tag), len(self._buffer) + 1)):
                        candidate = self._buffer[-tail_len:]
                        if self._close_tag.startswith(candidate):
                            partial_close = candidate

                    self._buffer = partial_close
                    break

                # Skip past closing tag
                self._buffer = self._buffer[end_idx + len(self._close_tag) :]
                self._inside_think = False
            else:
                start_idx = self._buffer.find(self._open_tag)
                if start_idx == -1:
                    # Check for partial tag at end of buffer
                    partial_match = ""
                    for tail_len in range(1, min(len(self._open_tag), len(self._buffer) + 1)):
                        candidate = self._buffer[-tail_len:]
                        if self._open_tag.startswith(candidate):
                            partial_match = candidate

                    if partial_match:
                        output.append(self._buffer[: -len(partial_match)])
                        self._buffer = partial_match
                        break

                    output.append(self._buffer)
                    self._buffer = ""
                    break

                # Emit text before the open tag
                output.append(self._buffer[:start_idx])
                self._buffer = self._buffer[start_idx + len(self._open_tag) :]
                self._inside_think = True

        return "".join(output)
