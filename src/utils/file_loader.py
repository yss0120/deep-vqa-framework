# src/utils/file_loader.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple, Union

from loguru import logger

from src.data.types import DatasetType


@dataclass
class AssetInfo:
    """File information: path, filename, extension, type (image/video)"""

    path: Path
    filename: str
    extension: str

    @property
    def dataset_type(self) -> DatasetType:
        """Returns the dataset type based on the file extension."""
        return DatasetType.from_extension(self.extension)

    @property
    def is_video(self) -> bool:
        return self.dataset_type == DatasetType.VIDEO

    @property
    def is_image(self) -> bool:
        return self.dataset_type == DatasetType.IMAGE


class CaseInsensitiveAssetResolver:
    """
    File path resolver: Pre-scans directories to build an index, shared by multiple processes.
    """

    def __init__(self, target_dir: Union[str, Path], allowed_extensions: list):
        self.target_dir = Path(target_dir).resolve()
        self.allowed_exts: Set[str] = {f".{ext.lower().lstrip('.')}" for ext in allowed_extensions}

        # A set of video file extensions used to automatically determine attributes during indexing.
        self.video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        self.image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # [Modification]: Dynamic lookup is no longer used; instead, the index is completed all at once during the initialization phase.
        # In this way, during the training phase, each process looks up the same table in memory, eliminating the need for disk operations.
        self.full_registry: Dict[str, AssetInfo] = {}

        logger.info(f"🚀 [Resolver] Scanning directory: {self.target_dir}")

        # Scan only once, mapping all file paths into a memory dictionary.
        # `rglob` can handle any level of nesting.
        for f in self.target_dir.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower()
                if ext in self.allowed_exts:
                    # Create an AssetInfo object
                    asset_info = AssetInfo(path=f, filename=f.name, extension=ext)
                    # Store complete physical information
                    self.full_registry[f.name.lower()] = asset_info
                else:
                    # Skip unnecessary files and do nothing.
                    continue
        logger.info(f"✅ [Resolver] Scan complete, totaling {len(self.full_registry)} files.")

    def resolve(self, label_filename: str) -> Path:
        name_lower = Path(label_filename).name.lower()

        # [Modification]: Direct dictionary lookup, O(1) complexity, no IO contention.
        if name_lower in self.full_registry:
            return self.full_registry[name_lower]  # Returns (Path, is_video)
        # Throw an exception only when not found.
        raise FileNotFoundError(f"❌ File does not exist: {label_filename} (Search directory: {self.target_dir})")

    def resolve_path(self, label_filename: str) -> Path:
        """Returns only the path (compatible with legacy code)"""
        return self.resolve(label_filename).path

    def resolve_with_info(self, label_filename: str) -> Tuple[Path, bool, bool]:
        """Compatible with legacy API: Returns (path, is_video, is_image)"""
        asset = self.resolve(label_filename)
        return asset.path, asset.is_video, asset.is_image
