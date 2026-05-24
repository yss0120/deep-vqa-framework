# src/data/eda/statistics.py

import cv2

# TODO:
# 输出dataset的imgs/videos总数、分辨率范围、尺寸大小、颜色空间
# 输出MOS/DMOS分布


def analyze_image_properties(sample_paths, sample_limit: int = 50) -> dict:
    """
    分析图像属性（分辨率统计）

    Args:
        image_paths: 图像文件路径列表
        sample_limit: 采样数量限制

    Returns:
        包含以下字段的字典:
        - total_files: 总文件数
        - min_resolution: 最小分辨率 (width, height)
        - max_resolution: 最大分辨率 (width, height)
    """
    resolutions = []
    for img_path in sample_paths[:sample_limit]:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))

    if resolutions:
        widths, heights = zip(*resolutions)
        return {
            "total_files": len(sample_paths),
            "min_resolution": (min(widths), min(heights)),
            "max_resolution": (max(widths), max(heights)),
        }
    return {}


def analyze_video_properties(video_paths: list) -> dict:
    """
    分析视频属性（从第一个采样视频获取）

    Args:
        video_paths: 视频文件路径列表

    Returns:
        包含以下字段的字典:
        - fps: 帧率
        - frame_count: 总帧数
        - width: 宽度
        - height: 高度
        - total_files: 总文件数
    """
    if not video_paths:
        return {}

    sample_v = video_paths[0]
    cap = cv2.VideoCapture(str(sample_v))

    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"total_files": len(video_paths), "fps": fps, "frame_count": frame_count, "width": width, "height": height}


def compute_mos_statistics(df) -> dict:
    """计算MOS统计信息"""
    return {
        "total_samples": len(df),
        "mos_range": (df["mos"].min(), df["mos"].max()),
        "mos_mean": df["mos"].mean(),
        "mos_std": df["mos"].std(),
    }
