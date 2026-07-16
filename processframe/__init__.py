"""Keyframe embedding deduplication package."""

from .pipeline import DeduplicationResult, build_deduplicated_index

__all__ = ["DeduplicationResult", "build_deduplicated_index"]
