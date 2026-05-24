# src/data/data_eda.py
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.data.eda.integrity import check_media_integrity
from src.data.eda.split import check_fold_distribution, split_train_val_test
from src.data.eda.statistics import analyze_image_properties, analyze_video_properties, compute_mos_statistics
from src.data.metadata_loader_factory import MetadataLoaderFactory
from src.data.types import DatasetType
from src.utils.config_loader import safe_load_yaml
from src.utils.path_manager import PathManager

# Location:
#   1. 面向整个dataset
#   2. 面向model的training


# Targets:
# 一、sampledata空间变换:
#   1. 尺寸变换：
#       - img: Random Crop
#       - video: Resize + Center Crop
#   2. 数据增强：
#       - img: Random Flip, Color Jitter
#       - video: Temporal Augmentation
#   3. 帧采样:
#       - img： N/A
#       - video: Fragment Sampling
#
#
#
# 二、质量分数预处理:
#   1. 归一化:
#       - img：MOS → [0,1]
#       - video: DMOS → [0,1]
#   2. 异常值检测:
#       - img: 3σ 原则剔除
#       - video: 3σ 原则剔除
#   3. 分数均衡:
#       - img：检查各分数区间样本数
#       - video：检查各分数区间样本数
#
#
# 三、视频特有处理:
#   1. 帧采样、时序一致性检查(计算相邻帧差异，检测跳帧、黑帧)
#   2. 探索帧数 vs MOS 的关系  —————— pass
#
#
# 四、数据加载:
#   1. 预加载到RAM: 小数据集直接 np.load()
#   2. 多进程 DataLoader: num_workers=4
#   3. 缓存预处理结果: lmdb / h5py  —————— pass
#
#
# 五、检查是否需要过采样/欠采样处理不平衡


# TODO: 改进 check_filename_label_match 的后缀检测逻辑
# 当前只检查第一个 label 是否包含点，如果数据集混合有后缀和无后缀的 label 会出错
# 建议: 检查大多数 label 是否有后缀，或通过配置明确指定

# TODO: 视频时序一致性检查可能性能较差
# 对于长视频，逐帧读取所有帧可能很慢
# 建议: 添加采样间隔参数（如每隔 N 帧检查一次），或添加 tqdm 进度条

# TODO: 确保 results 目录存在后再写入报告
# _write_integrity_report 中直接拼接 results 路径，但目录可能不存在
# 建议: 在写入前调用 results_dir.mkdir(parents=True, exist_ok=True)


class DataEDA:
    """
    Data exploration and analysis: loading metadata, cleaning samples, statistical analysis, and dataset partitioning.
    """

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self, dataset_name: str, data_dir: Path, dataset_info: Optional[Dict] = None):
        self.dataset_name = dataset_name.lower()
        self.df = None
        self.stats = {}

        self.dataset_config = dataset_info if dataset_info else self._load_dataset_config_from_yaml(None)

        self.is_video = self.dataset_config.get("data_type") == "video"
        self.file_extensions = [ext.lower() for ext in self.dataset_config.get("file_extensions", ["*"])]

        if data_dir:
            self.data_dir = Path(data_dir).resolve()
        else:
            # 从 YAML 配置的 paths.root 和 paths.data 动态构建
            root = PathManager.resolve("dataset", dataset=self.dataset_name)
            sub_path = self.dataset_config.get("paths", {}).get("data", "")
            self.data_dir = (root / sub_path).resolve()

        logger.info(f"🚀 [PathFix] Data directory: {self.data_dir}")
        plt.switch_backend("Agg")

        # 引入无视大小写资产解析网关
        from src.utils.file_loader import CaseInsensitiveAssetResolver

        self.resolver = CaseInsensitiveAssetResolver(target_dir=self.data_dir, allowed_extensions=self.file_extensions)

    def _load_dataset_config_from_yaml(self, config_path: Path) -> Dict:
        if config_path is None:
            config_path = PathManager.get_config_file("dataset_config.yaml")

        # 使用 safe_load_yaml 加载
        full_config = safe_load_yaml(config_path, "Dataset configuration file")

        # 大小写不敏感查找
        lowered_config = {k.lower(): v for k, v in full_config.items()}
        target_key = self.dataset_name.lower()

        if target_key not in lowered_config:
            available = list(full_config.keys())
            logger.error(f"❌ Dataset '{self.dataset_name}' not found in config")
            raise KeyError(f"Dataset '{self.dataset_name}' missing. Available: {available}")

        return full_config[self.dataset_name]  # 注意：返回原始大小写的 key

    # ==================== 一、Basic Analysis ====================

    def load_metadata(self) -> pd.DataFrame:
        meta_file = (
            PathManager.resolve("dataset", dataset=self.dataset_name)
            / self.dataset_config["paths"].get("metadata", "")
            / self.dataset_config["metadata"]["mos_file"]
        )

        if not meta_file.exists():
            logger.error(f"❌ Metadata file not found: {meta_file}")
            return pd.DataFrame()

        try:
            loader = MetadataLoaderFactory.get_loader(self.dataset_name)
            df = loader.load(meta_file)
            self.df = df  # 保存到 self.df
            logger.info(f"✅ [Data Engine] Successfully loaded {len(df)} cleaned samples.")
            return df

        except Exception as e:
            logger.error(f"❌ Parsing failed: {e}")
            return pd.DataFrame()

    def ensure_split_column(self):
        """
        根据 dataset_config 中的比例执行切分。
        如果 dataset_config 中定义了 split 参数，则使用定义的比例；
        否则默认执行 8:1:1 划分。
        """
        # 如果已经存在有效的 split 列，则跳过
        if "split" in self.df.columns and not self.df["split"].isnull().all():
            logger.info("✅ [Split] 已检测到划分列，跳过自动切分。")
            return

        split_cfg = self.dataset_config.get("split", {})

        # 情况1：预定义划分（直接指定 sample_id 列表）
        if "train" in split_cfg and isinstance(split_cfg["train"], list):
            logger.info("⚙️ [Split] 使用预定义划分")
            self.df["split"] = "test"  # 默认 test
            self.df.loc[self.df["sample_id"].isin(split_cfg["train"]), "split"] = "train"
            self.df.loc[self.df["sample_id"].isin(split_cfg.get("val", [])), "split"] = "val"
            # test 保持默认
            return

        # 情况2：按比例划分
        train_ratio = split_cfg.get("train_ratio", 0.8)
        val_ratio = split_cfg.get("val_ratio", 0.1)

        logger.info(f"⚙️ [Split] 按比例切分: Train={train_ratio}, Val={val_ratio}")

        train_df, val_df, test_df = split_train_val_test(self.df, train_ratio=train_ratio, val_ratio=val_ratio, random_state=42)

        self.df["split"] = "train"
        self.df.loc[val_df.index, "split"] = "val"
        self.df.loc[test_df.index, "split"] = "test"

        logger.info(f"✅ 划分完成: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def basic_statistics(self):
        if self.df is None or self.df.empty:
            logger.error("❌ No data loaded. Run load_metadata() first.")
            return

        mos_stats = compute_mos_statistics(self.df)
        logger.info(f"\n{'=' * 50}\n[-] {self.dataset_name} Basic Statistics\n{'=' * 50}")
        logger.info(f"  Total samples in Sheet: {mos_stats['total_samples']}")
        logger.info(f"  MOS Range: [{mos_stats['mos_range'][0]:.3f}, {mos_stats['mos_range'][1]:.3f}]")
        logger.info(f"  MOS Mean: {mos_stats['mos_mean']:.3f} | MOS Std: {mos_stats['mos_std']:.3f}")

        self._analyze_media_properties()  # 统一调用

        self.stats["total_samples"] = mos_stats["total_samples"]
        self.stats["mos_range"] = mos_stats["mos_range"]

    def _analyze_image_properties(self, image_paths):
        props = analyze_image_properties(image_paths)
        if props:
            logger.info(f"    Scanned Physical Files on Disk: {props['total_files']}")
            logger.info(
                f"    Resolution boundary: {props['min_resolution'][0]}x{props['min_resolution'][1]} ~ "
                f"{props['max_resolution'][0]}x{props['max_resolution'][1]}"
            )

    def _analyze_video_properties(self, video_paths):
        props = analyze_video_properties(video_paths)
        if props:
            logger.info(f"    Scanned Physical Videos on Disk: {props['total_files']}")
            logger.info(f"    Sample Video Size: {props['width']}x{props['height']} | FPS: {props['fps']:.2f}")
            logger.info(f"    Total Frame Count: {int(props['frame_count'])}")

    def _analyze_media_properties(self):
        """Unified analysis of media attributes (based on dataset type)"""
        # 收集文件
        media_paths = set()
        for ext in self.file_extensions:
            media_paths.update(self.data_dir.rglob(f"*.{ext.lower()}"))
            media_paths.update(self.data_dir.rglob(f"*.{ext.upper()}"))
            media_paths.update(self.data_dir.rglob(f"*.{ext.capitalize()}"))

        media_paths = list(media_paths)

        if not media_paths:
            logger.warning(f"⚠️ No media files in {self.data_dir}")
            return

        if not hasattr(self, "dataset_type"):
            try:
                first_file = Path(media_paths[0]).name
                asset = self.resolver.resolve(first_file)
                self.dataset_type = asset.dataset_type
            except Exception:
                self.dataset_type = DatasetType.IMAGE if not self.is_video else DatasetType.VIDEO

        # 根据类型调用对应的分析函数
        if self.dataset_type == DatasetType.IMAGE:
            logger.info("\n  Image Properties:")
            self._analyze_image_properties(media_paths)
        elif self.dataset_type == DatasetType.VIDEO:
            logger.info("\n  Video Properties:")
            self._analyze_video_properties(media_paths)
        else:
            logger.warning(f"⚠️ Unknown dataset type: {self.dataset_type}")

    def check_integrity(self, max_samples: int = None, skip_video_check: bool = False) -> Dict:
        """Check file integrity: missing, corrupted, duplicate"""
        logger.info("\n 🔍 [Integrity] Integrity check started...")

        # 如果指定了最大样本数，只检查部分
        df_to_check = self.df if max_samples is None else self.df.head(max_samples)

        missing = []
        corrupted = []
        # 使用 set 优化查找速度，提高性能
        duplicates = self.df[self.df.duplicated(subset=["sample_id"])]["sample_id"].tolist()

        quarantine_dir = self._PROJECT_ROOT / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        valid_indices = []  # 记录合法的索引

        # 添加进度条
        from tqdm import tqdm

        # 预先处理路径解析，避免循环内重复解析
        for i, row in tqdm(df_to_check.iterrows(), total=len(df_to_check), desc="检查文件完整性"):
            sid = str(row["sample_id"])

            # 1. 路径解析校验
            try:
                asset = self.resolver.resolve(sid)
                target_path = asset.path
                is_video = asset.is_video
            except Exception:  # 捕获解析器可能抛出的所有异常
                missing.append(sid)
                continue

            # 2. 快速检查：文件是否存在
            if not target_path.exists():
                missing.append(sid)
                continue

            # 3. 如果需要跳过视频内容检查，只检查存在性
            if skip_video_check and is_video:
                valid_indices.append(i)
                continue

            # 4. 深度检查（图像或视频内容）
            dataset_type = DatasetType.VIDEO if is_video else DatasetType.IMAGE
            is_ok, error, _ = check_media_integrity(target_path, dataset_type)
            # 2. 文件可读性校验 (改为只读模式，减少 I/O 开销)
            if not is_ok:
                corrupted.append(sid)

                # 只在确实文件存在且无法读取时移动
                if target_path.exists():
                    dest = quarantine_dir / target_path.name
                    if dest.exists():
                        dest = quarantine_dir / f"{target_path.stem}_corrupted{target_path.suffix}"
                    shutil.move(str(target_path), str(dest))
                    logger.warning(f"⚠️ Corrupted file has been quarantined: {target_path.name} ({error})")
                continue

            valid_indices.append(i)

        # 3. 内存索引更新：只保留验证通过的行
        self.df = self.df.loc[valid_indices].reset_index(drop=True)

        # 4. 生成审计报告 (使用更加紧凑的格式)
        self._write_integrity_report(missing, corrupted)

        logger.info(
            f"✅ [Integrity] Cleaning complete. Retained samples: {len(self.df)} | Missing samples: {len(missing)} | Corrupted samples: {len(corrupted)}"
        )
        return {"corrupted": corrupted, "missing": missing, "duplicates": duplicates}

    def _write_integrity_report(self, missing: List[str], corrupted: List[str]):
        """Generate an integrity check report"""
        report_path = (self._PROJECT_ROOT / "results") / f"{self.dataset_name}_integrity_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"--- Integrity Audit Report: {self.dataset_name} ---\n")
            f.write(f"Date: {pd.Timestamp.now()}\n")
            f.write(f"Summary: {len(missing)} missing, {len(corrupted)} corrupted.\n\n")
            if missing:
                f.write("Missing Files:\n" + "\n".join(missing) + "\n\n")
            if corrupted:
                f.write("Corrupted Files:\n" + "\n".join(corrupted) + "\n")

    def check_filename_label_match(self) -> bool:
        # 检查磁盘上的文件和 metadata 中的 label 是否完全匹配
        """Compares the disk filename with the tag filename to see if they match (case-insensitive)"""
        if self.df is None:
            return False

        physical_names = set()
        for ext in self.file_extensions:
            physical_names.update({p.name.lower() for p in self.data_dir.rglob(f"*.{ext}")})
            physical_names.update({p.name.lower() for p in self.data_dir.rglob(f"*.{ext.upper()}")})

        label_names = set(self.df["sample_id"].astype(str))

        if len(label_names) > 0 and "." not in list(label_names)[0]:
            physical_names = {Path(name).stem for name in physical_names}

        match = physical_names == label_names
        logger.info(f"  Filename-Label Strict Complement Match: {match}")
        if not match:
            logger.warning(f"    Diff - Excess files on Disk: {len(physical_names - label_names)}")
            logger.warning(f"    Diff - Deficit files on Disk (Missing from labels): {len(label_names - physical_names)}")
        return match

    def visualize_mos_distribution(self, save_dir: Path = Path("results/plots")):
        if self.df is None or self.df.empty:
            return
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].hist(self.df["mos"], bins=25, edgecolor="black", alpha=0.75, color="steelblue")
        axes[0].set_xlabel("MOS / DMOS Score")
        axes[0].set_ylabel("Sample Frequency")
        axes[0].set_title(f"{self.dataset_name} - Continuous Score Distribution", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].boxplot(
            self.df["mos"],
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="crimson", linewidth=1.5),
        )
        axes[1].set_ylabel("MOS Range")
        axes[1].set_title(f"{self.dataset_name} - Statistical Boxplot", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        output_path = save_dir / f"{self.dataset_name}_mos_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 MOS distribution graph saved to: {output_path}")

    # ==================== 四、质量分数预处理 ====================

    def normalize_scores(self) -> pd.DataFrame:
        min_val = self.df["mos"].min()
        max_val = self.df["mos"].max()
        # 防零除防御
        denom = (max_val - min_val) if max_val != min_val else 1.0
        self.df["normalized_score"] = (self.df["mos"] - min_val) / denom
        logger.info("  Score Processing -> Scale normalizes [0, 1] completed.")
        return self.df

    def detect_outliers_3sigma(self) -> pd.DataFrame:
        mean = self.df["mos"].mean()
        std = self.df["mos"].std() if self.df["mos"].std() > 0 else 1.0
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        outliers = self.df[(self.df["mos"] < lower_bound) | (self.df["mos"] > upper_bound)]
        logger.info(f"  Outlier Detection -> Found {len(outliers)} samples drifting beyond 3σ boundary.")
        self.df["is_outlier"] = (self.df["mos"] < lower_bound) | (self.df["mos"] > upper_bound)
        return self.df

    # ==================== 三、K折交叉验证 ====================

    def check_fold_score_distribution(self, n_splits: int = 5) -> List[Dict]:
        """使用分位数进行分层多折，确保每一折回归分布高度一致"""
        return check_fold_distribution(df=self.df, n_splits=n_splits, random_state=42, verbose=True)

    # ==================== 五、视频特有增强处理 ====================

    def check_temporal_consistency_by_path(self, video_path: Path) -> Dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"avg_diff": 0, "has_black_frames": False, "has_jumps": False}

        diffs = []
        black_frames_count = 0
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 8.0:
                black_frames_count += 1

            if prev_frame is not None:
                frame_diff = np.mean(cv2.absdiff(gray, prev_frame))
                diffs.append(frame_diff)
            prev_frame = gray

        cap.release()

        avg_diff = np.mean(diffs) if diffs else 0
        has_jumps = np.std(diffs) > (avg_diff * 2.5) if diffs else False

        return {
            "avg_frame_delta": round(float(avg_diff), 4),
            "black_frame_hits": black_frames_count,
            "has_black_frames": black_frames_count > 0,
            "has_jumps": has_jumps,
        }

    # ==================== 主分析流一键呼叫 ====================

    def run_full_eda(self, save_dir: Optional[Path] = None, skip_integrity: bool = False) -> Dict:
        """Execute the complete data exploration and analysis process"""
        if save_dir is None:
            save_dir = self._PROJECT_ROOT / "results" / "plots"

        save_dir.mkdir(parents=True, exist_ok=True)
        self.df = self.load_metadata()
        if self.df is None or self.df.empty:
            logger.error(f"[EDA] Pipeline initialization failed for: {self.dataset_name}")
            return {}

        self.basic_statistics()
        # TODO:
        if skip_integrity:
            integrity_res = {"corrupted": [], "missing": [], "duplicates": []}
            logger.info("⏭️ [Integrity] Skip integrity check")
        else:
            # 只检查文件存在性，不检查视频内容（快很多）
            integrity_res = self.check_integrity(skip_video_check=True)

        self.ensure_split_column()
        self.check_filename_label_match()

        self.visualize_mos_distribution(save_dir)

        self.normalize_scores()
        self.detect_outliers_3sigma()
        fold_res = self.check_fold_score_distribution()

        if self.is_video:
            logger.info("🎬 Video dataset subtype detected. Sampling standard temporal consistency check...")
            video_samples = []
            for ext in self.file_extensions:
                video_samples.extend(list(self.data_dir.rglob(f"*.{ext}")))
                video_samples.extend(list(self.data_dir.rglob(f"*.{ext.upper()}")))
            if video_samples:
                temporal_report = self.check_temporal_consistency_by_path(video_samples[0])
                logger.info(f"   [Temporal Diagnostics Template] Sample: {video_samples[0].name}")
                logger.info(f"   └─ Frame Delta Diff: {temporal_report['avg_frame_delta']} | Has Jumps/Drop: {temporal_report['has_jumps']}")
                self.stats["sample_temporal_report"] = temporal_report

        self.stats.update({"integrity": integrity_res, "fold_variance": fold_res})
        logger.info(f"🏁 [EDA Complete] Mission perfectly accomplished on {self.dataset_name}.\n")
        return self.stats
