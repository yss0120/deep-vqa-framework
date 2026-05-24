# src/data/types.py
from enum import Enum


class DatasetType(Enum):
    IMAGE = "image"
    VIDEO = "video"

    @classmethod
    def from_extension(cls, ext: str) -> "DatasetType":
        """根据文件扩展名判断数据集类型"""
        ext = ext.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if ext in video_exts:
            return cls.VIDEO
        elif ext in image_exts:
            return cls.IMAGE
        else:
            raise ValueError(f"Unknown file extension: {ext}")

    @classmethod
    def from_config(cls, config_value: str) -> "DatasetType":
        """从配置文件读取的类型字符串转换"""
        if config_value.lower() == "video":
            return cls.VIDEO
        elif config_value.lower() == "image":
            return cls.IMAGE
        else:
            raise ValueError(f"Unknown dataset type configuration: {config_value}")
