"""The creature's inner life, as far as it can be read out of its brain."""

from .thoughts import KINDS, ThoughtStream, format_thought, read_thoughts, thoughts_path

__all__ = ["ThoughtStream", "KINDS", "thoughts_path", "read_thoughts", "format_thought"]
