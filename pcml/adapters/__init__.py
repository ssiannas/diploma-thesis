"""Framework adapters for external compression methods."""

from pcml.adapters.base import BaseAdapter, CompressionResult
from pcml.adapters.pcgcv2 import PCGCv2Adapter, PCGCv2Config

__all__ = ["BaseAdapter", "CompressionResult", "PCGCv2Adapter", "PCGCv2Config"]
