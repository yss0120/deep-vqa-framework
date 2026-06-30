#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立推理脚本：加载训练好的 checkpoint，对单张图片或单个视频文件进行质量评分预测。

设计原则：
- checkpoint 里已经保存了完整的 config（见 src/core/engine.py `_save_checkpoint` 中的
  `state["config"] = self.config`），所以推理时不需要重新走 load_system_config /
  dataset_config.yaml 那一套，直接从 checkpoint 里还原配置和模型结构，避免"训练用的配置"
  和"推理用的配置"再次出现不一致。
- 不依赖 DataEDA / config_loader / trainer / engine / path_manager，只依赖
  src.models.iqavqa_net.IQAVQANet 这一个项目内文件，方便单独交付给后端组部署。
- 后端组在部署时需在项目根目录执行 `uv sync` 同步依赖，然后启动 `api.py` 服务。

模型路由方式：
    forward(self, x: torch.Tensor) -> torch.Tensor
    - 4D Tensor [B, 3, H, W]    -> 图片，走单帧空间特征提取分支
    - 5D Tensor [B, F, 3, H, W] -> 视频，先逐帧提取特征再做时序融合
    模型不接受额外的 mode 参数，完全靠输入 tensor 的维度自动路由。

自动模型选择：
    - 单文件或批量推理时，按文件类型自动选择模型
    - 图片 → IQA 模型 (iqa-models/tid2013_best.pt)
    - 视频 → VQA 模型 (vqa-models/konvid_best.pt)
    - 混合目录 → 分别加载两组模型，各自推理
    - 可通过 -c 手动指定统一模型覆盖自动选择

完整性检查：
    - 图片：可解码、分辨率有效、非全黑/全白、颜色种类 ≥ 2
    - 视频：可解码、实际解码帧数与视频文件声称的帧数偏差不超过 10%、
            黑帧/白帧比例、坏帧比例、跳帧检测（基于时间戳间隔统计）

使用方式：
    # 自动选择模型（推荐）
    uv run python -m deploy.infer -i test.jpg
    uv run python -m deploy.infer -i test.mp4
    uv run python -m deploy.infer -i ./mixed_dir/ -o results.json  # 混合目录自动分组

    # 手动指定模型（所有文件用同一个）
    uv run python -m deploy.infer -c model.pt -i test.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from loguru import logger

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("⚠️ Decord 未安装，视频读取将回退到 OpenCV。")

# ==================== 路径处理 ====================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEPLOY_ROOT = Path(__file__).resolve().parent

from src.models.iqavqa_net import IQAVQANet

# ==================== 文件类型常量 ====================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ==================== 默认模型路径 ====================
DEFAULT_IQA_MODEL = "iqa-models/tid2013_best.pt"
DEFAULT_VQA_MODEL = "vqa-models/konvid_best.pt"

# ==================== 完整性检查阈值 ====================
MIN_COLORS = 2
BLACK_FRAME_THRESHOLD = 8.0
WHITE_FRAME_THRESHOLD = 245.0
BAD_FRAME_DIFF_THRESHOLD = 0.5
MAX_BLACK_WHITE_RATIO = 0.3
MAX_BAD_RATIO = 0.3
MAX_FRAME_DEVIATION = 0.10
FRAME_DROP_INTERVAL_THRESHOLD = 1.5
VIDEO_MEAN_WARNING_THRESHOLD = 0.01


# ==================== 路由函数 ====================
def detect_media_type(file_path: Union[str, Path]) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"不支持的文件类型: {ext}")


# ==================== Checkpoint 加载 ====================
def load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cuda") -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ Checkpoint 不存在: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "config" not in checkpoint:
        raise KeyError(
            "🚨 Checkpoint 中没有保存 config 字段，无法还原模型结构。"
            "请确认该 checkpoint 是由 TrainerEngine._save_checkpoint 保存的。"
        )

    config = checkpoint["config"]
    model = IQAVQANet(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    logger.info(f"✅ [Infer] 已加载 checkpoint: {ckpt_path.name} (epoch={epoch})")
    if metrics:
        logger.info(f"   └─ 该 checkpoint 训练时的验证指标: {metrics}")

    return model, config


# ==================== 反归一化 ====================
def denormalize(score: float, mos_min: Optional[float], mos_max: Optional[float]) -> Optional[float]:
    if mos_min is None or mos_max is None:
        return None
    return float(score) * (mos_max - mos_min) + mos_min


# ==================== 数据预处理 ====================
class Preprocessor:
    def __init__(self, num_frames: int = 8, input_size: int = 224):
        self.num_frames = num_frames
        self.input_size = input_size

    def process(self, file_path: Union[str, Path]) -> torch.Tensor:
        ext = Path(file_path).suffix.lower()
        if ext in VIDEO_EXTS:
            return self._process_video(file_path)
        elif ext in IMAGE_EXTS:
            return self._process_image(file_path)
        else:
            raise ValueError(f"❌ 不支持的文件类型: {ext}")

    def _process_video(self, file_path: Union[str, Path]) -> torch.Tensor:
        if DECORD_AVAILABLE:
            try:
                vr = VideoReader(str(file_path), ctx=cpu(0))
                total_frames = len(vr)

                claimed_frames = self._get_claimed_frame_count(str(file_path))
                if claimed_frames > 0:
                    deviation = abs(total_frames - claimed_frames) / max(claimed_frames, 1)
                    if deviation > MAX_FRAME_DEVIATION:
                        logger.warning(f"⚠️ 帧数偏差 {deviation*100:.1f}%（声称={claimed_frames}, 实际={total_frames}）")

                if total_frames >= self.num_frames:
                    indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
                else:
                    indices = list(range(total_frames))

                frames = vr.get_batch(indices).asnumpy()
                if frames.size == 0:
                    raise ValueError(f"Decord 返回了空帧序列: {file_path}")

                timestamps = self._get_frame_timestamps(vr, indices)
                if len(timestamps) > 2:
                    intervals = np.diff(timestamps)
                    mean_interval = np.mean(intervals)
                    max_interval = np.max(intervals)
                    if max_interval > mean_interval * FRAME_DROP_INTERVAL_THRESHOLD:
                        logger.warning(f"⚠️ 检测到跳帧: max间隔={max_interval:.1f}ms, mean间隔={mean_interval:.1f}ms")

                frame_stats = self._analyze_frames(frames)
                total = len(frames)
                black_ratio = frame_stats["black"] / max(total, 1)
                white_ratio = frame_stats["white"] / max(total, 1)
                bad_ratio = frame_stats["bad"] / max(total, 1)

                if black_ratio > MAX_BLACK_WHITE_RATIO:
                    logger.warning(f"⚠️ 黑帧比例过高: {black_ratio*100:.1f}%")
                if white_ratio > MAX_BLACK_WHITE_RATIO:
                    logger.warning(f"⚠️ 白帧比例过高: {white_ratio*100:.1f}%")
                if bad_ratio > MAX_BAD_RATIO:
                    logger.warning(f"⚠️ 坏帧比例过高: {bad_ratio*100:.1f}%")

                if len(frames) < self.num_frames:
                    logger.warning(f"⚠️ 帧数不足 {len(frames)}/{self.num_frames}，将用最后一帧填充")

                resized = [cv2.resize(f, (self.input_size, self.input_size)) for f in frames]
                video_np = np.stack(resized)
                tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

                if tensor.size(0) < self.num_frames:
                    pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
                    tensor = torch.cat([tensor, pad], dim=0)

                if tensor.mean() < VIDEO_MEAN_WARNING_THRESHOLD:
                    logger.warning(f"⚠️ 视频几乎全黑: {file_path}")

                return tensor

            except Exception as e:
                logger.warning(f"⚠️ [Decord] 解析失败，回退到 OpenCV: {e}")

        return self._process_video_opencv(file_path)

    def _process_video_opencv(self, file_path: Union[str, Path]) -> torch.Tensor:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"❌ 无法打开视频: {file_path}")

        claimed_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        timestamps = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp)
            frame = cv2.resize(frame, (self.input_size, self.input_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"❌ 视频没有读到任何有效帧: {file_path}")

        if claimed_frames > 0:
            actual = len(frames)
            deviation = abs(actual - claimed_frames) / max(claimed_frames, 1)
            if deviation > MAX_FRAME_DEVIATION:
                logger.warning(f"⚠️ 帧数偏差 {deviation*100:.1f}%（声称={claimed_frames}, 实际={actual}）")

        if len(timestamps) > 2:
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            max_interval = np.max(intervals)
            if max_interval > mean_interval * FRAME_DROP_INTERVAL_THRESHOLD:
                logger.warning(f"⚠️ 检测到跳帧: max间隔={max_interval:.1f}ms, mean间隔={mean_interval:.1f}ms")

        frame_stats = self._analyze_frames(frames)
        total = len(frames)
        black_ratio = frame_stats["black"] / max(total, 1)
        white_ratio = frame_stats["white"] / max(total, 1)
        bad_ratio = frame_stats["bad"] / max(total, 1)

        if black_ratio > MAX_BLACK_WHITE_RATIO:
            logger.warning(f"⚠️ 黑帧比例过高: {black_ratio*100:.1f}%")
        if white_ratio > MAX_BLACK_WHITE_RATIO:
            logger.warning(f"⚠️ 白帧比例过高: {white_ratio*100:.1f}%")
        if bad_ratio > MAX_BAD_RATIO:
            logger.warning(f"⚠️ 坏帧比例过高: {bad_ratio*100:.1f}%")

        if len(frames) < self.num_frames:
            logger.warning(f"⚠️ 帧数不足 {len(frames)}/{self.num_frames}，将用最后一帧填充")

        video_np = np.stack(frames)
        tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

        if tensor.size(0) < self.num_frames:
            pad = tensor[-1].unsqueeze(0).repeat(self.num_frames - tensor.size(0), 1, 1, 1)
            tensor = torch.cat([tensor, pad], dim=0)

        if tensor.mean() < VIDEO_MEAN_WARNING_THRESHOLD:
            logger.warning(f"⚠️ 视频几乎全黑: {file_path}")

        return tensor

    def _get_claimed_frame_count(self, file_path: str) -> int:
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return count
        except Exception:
            return 0

    def _get_frame_timestamps(self, vr, indices) -> List[float]:
        timestamps = []
        for idx in indices:
            try:
                ts = vr.get_frame_timestamp(idx)[0] * 1000
                timestamps.append(ts)
            except Exception:
                timestamps.append(idx * 33.33)
        return timestamps

    def _analyze_frames(self, frames) -> Dict[str, int]:
        stats = {"black": 0, "white": 0, "bad": 0}
        prev_gray = None

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            mean_gray = np.mean(gray)

            if mean_gray < BLACK_FRAME_THRESHOLD:
                stats["black"] += 1
            if mean_gray > WHITE_FRAME_THRESHOLD:
                stats["white"] += 1
            if prev_gray is not None:
                diff = np.mean(np.abs(gray - prev_gray))
                if diff < BAD_FRAME_DIFF_THRESHOLD:
                    stats["bad"] += 1
            prev_gray = gray

        return stats

    def _process_image(self, file_path: Union[str, Path]) -> torch.Tensor:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"❌ 图像解码失败: {file_path}")

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"❌ 无效分辨率: {w}x{h}")

        unique_colors = len(np.unique(img))
        if unique_colors < MIN_COLORS:
            raise ValueError(f"❌ 颜色种类过少: {unique_colors}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        if mean_gray < BLACK_FRAME_THRESHOLD:
            logger.warning(f"⚠️ 图像几乎全黑: {file_path} (mean={mean_gray:.1f})")
        if mean_gray > WHITE_FRAME_THRESHOLD:
            logger.warning(f"⚠️ 图像几乎全白: {file_path} (mean={mean_gray:.1f})")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        return torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0


# ==================== 批量推理核心 ====================
@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    file_path: Union[str, Path],
    config: Dict[str, Any],
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> Dict[str, Any]:
    num_frames = config.get("model", {}).get("num_frames", 8)
    input_size = config.get("model", {}).get("input_size", 224)

    preprocessor = Preprocessor(num_frames=num_frames, input_size=input_size)
    data_tensor = preprocessor.process(file_path)
    data_tensor = data_tensor.unsqueeze(0).to(device)

    output = model(data_tensor)
    output = output.float()
    if output.ndim > 1 and output.size(-1) == 1:
        output = output.squeeze(-1)
    raw_score = float(output.flatten()[0].cpu().item())

    if mos_min is None or mos_max is None:
        dataset_info = config.get("dataset_info", {}) or {}
        mos_min = mos_min if mos_min is not None else dataset_info.get("mos_min")
        mos_max = mos_max if mos_max is not None else dataset_info.get("mos_max")

    real_score = denormalize(raw_score, mos_min, mos_max)

    return {
        "file": str(file_path),
        "raw_score": round(raw_score, 6),
        "mos_score": round(real_score, 4) if real_score is not None else None,
        "task_type": config.get("task_type", "unknown"),
        "model_name": config.get("model", {}).get("name", "unknown"),
    }


def predict_batch(
    model: torch.nn.Module,
    input_paths: List[Union[str, Path]],
    config: Dict[str, Any],
    device: str = "cuda",
    mos_min: Optional[float] = None,
    mos_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    results = []
    for path in input_paths:
        try:
            results.append(predict_single(model, path, config, device, mos_min, mos_max))
        except Exception as e:
            logger.error(f"🚨 推理失败: {path} | {e}")
            results.append({"file": str(path), "error": str(e)})
    return results


# ==================== CLI 入口 ====================
def main():
    parser = argparse.ArgumentParser(
        description="Deep VQA/IQA 推理脚本（自动选择 IQA/VQA 模型，支持混合目录）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动选择模型（单文件）
  uv run python -m deploy.infer -i test.jpg
  uv run python -m deploy.infer -i test.mp4

  # 混合目录自动分组推理
  uv run python -m deploy.infer -i ./mixed_dir/ -o results.json

  # 手动指定统一模型
  uv run python -m deploy.infer -c model.pt -i test.jpg
"""
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="模型路径 (可选，不指定则自动选择)")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入文件或目录路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mos_min", type=float, default=None, help="反归一化下限")
    parser.add_argument("--mos_max", type=float, default=None, help="反归一化上限")
    parser.add_argument("-o", "--output", type=str, default=None, help="结果 JSON 输出路径")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()

    if args.cpu:
        args.device = "cpu"

    input_path = Path(args.input)

    # 收集文件
    targets = []
    if input_path.is_dir():
        for ext in VIDEO_EXTS | IMAGE_EXTS:
            targets.extend(input_path.rglob(f"*{ext}"))
        logger.info(f"📁 在目录 {input_path} 中找到 {len(targets)} 个文件")
    else:
        targets = [input_path]

    if not targets:
        logger.error(f"❌ 在 {input_path} 没有找到任何支持的图片/视频文件")
        sys.exit(1)

    # 按类型分组
    image_files, video_files = [], []
    for f in targets:
        ext = f.suffix.lower()
        if ext in IMAGE_EXTS:
            image_files.append(f)
        elif ext in VIDEO_EXTS:
            video_files.append(f)
        else:
            logger.warning(f"⚠️ 跳过不支持的格式: {f}")

    results = []

    if args.checkpoint:
        # 手动模式：所有文件用同一模型
        model_path = Path(args.checkpoint)
        if not model_path.exists():
            logger.error(f"❌ 模型不存在: {model_path}")
            sys.exit(1)
        model, config = load_checkpoint(model_path, device=args.device)
        all_files = image_files + video_files
        if all_files:
            results = predict_batch(model, all_files, config, args.device, args.mos_min, args.mos_max)
    else:
        # 自动模式：图片走 IQA，视频走 VQA
        if image_files:
            iqa_path = _DEPLOY_ROOT / DEFAULT_IQA_MODEL
            if not iqa_path.exists():
                logger.error(f"❌ IQA 模型不存在: {iqa_path}")
            else:
                logger.info(f"🖼️ 加载 IQA 模型，推理 {len(image_files)} 张图片")
                model, config = load_checkpoint(iqa_path, device=args.device)
                results.extend(predict_batch(model, image_files, config, args.device, args.mos_min, args.mos_max))

        if video_files:
            vqa_path = _DEPLOY_ROOT / DEFAULT_VQA_MODEL
            if not vqa_path.exists():
                logger.error(f"❌ VQA 模型不存在: {vqa_path}")
            else:
                logger.info(f"🎬 加载 VQA 模型，推理 {len(video_files)} 个视频")
                model, config = load_checkpoint(vqa_path, device=args.device)
                results.extend(predict_batch(model, video_files, config, args.device, args.mos_min, args.mos_max))

    if not results:
        logger.warning("⚠️ 没有成功推理任何文件")
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("推理结果:")
    for r in results:
        if "error" in r:
            logger.error(f"  ❌ {r['file']}: {r['error']}")
        else:
            mos_str = f"{r['mos_score']:.4f}" if r["mos_score"] is not None else "N/A"
            logger.info(f"  ✅ {r['file']}: raw={r['raw_score']:.4f} | mos={mos_str}")
    logger.info("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 结果已保存到: {out_path}")


if __name__ == "__main__":
    main()