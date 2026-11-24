"""VN + FoldingNet autoencoder with a normalizing-flow prior over its codes (comparison model, 2026-11-24)."""

from .model import VNFoldConfig, VNFoldingAE, chamfer, make_grid

__all__ = ["VNFoldConfig", "VNFoldingAE", "chamfer", "make_grid"]
