"""Framework adapters for external compression methods."""

from pcml.frameworks.base import BaseAdapter, CompressionResult
from pcml.frameworks.pcgcv2 import PCGCv2Adapter, PCGCv2Config

__all__ = ["BaseAdapter", "CompressionResult", "PCGCv2Adapter", "PCGCv2Config"]
