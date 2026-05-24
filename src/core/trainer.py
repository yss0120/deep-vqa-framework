import copy
import gc
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

try:
    from decord import VideoReader, cpu

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("⚠️ Decord is not installed; video loading will use OpenCV fallback.")


from src.core.engine import TrainerEngine
from src.models.iqavqa_net import IQAVQALoss, IQAVQANet


def worker_init_fn(worker_id):
    """In a multi-process DataLoader, set a different random seed for each worker."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class VQA_IQADataset(Dataset):
    """
    A multimodal data loader that supports images and videos.
    Reads DataFrames processed by DataEDA and decodes image or video frames as needed.
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path, config: Dict = None, transform: Any = None, resolver: Any = None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.resolver = resolver

        self.config = config or {}
        self.num_frames = self.config.get("model", {}).get("num_frames", 8)

        self.traditional_cols = [c for c in ["ssim", "vif", "dlm", "vmaf", "niqe"] if c in self.df.columns]

        self.is_video_mode = any(str(sample).endswith((".mp4", ".avi", ".mov", ".mkv")) for sample in self.df["sample_id"].head(10))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        default_payload = {
            "data": torch.zeros((self.num_frames, 3, 224, 224)),
            "label": torch.tensor(0.0, dtype=torch.float32),
        }
        row = self.df.iloc[idx]
        sample_id = str(row["sample_id"]).strip()

        # 1. Physical layer: Get Path and properties
        try:
            target_path, is_video, is_image = self.resolver.resolve_with_info(sample_id)
            str_path = str(target_path.absolute())
        except Exception as e:
            logger.error(f"❌ Addressing failed: {sample_id} | {e}")
            # Debugging Interception: Set a breakpoint immediately when addressing fails to check why the file pointed to by Metadata cannot be found.
            if os.environ.get("DEBUG", "0") == "1":
                breakpoint()
            return default_payload

        # 2. Execution layer: Automatically distribute traffic based on physical attributes.
        data_tensor = None
        try:
            if is_video:
                data_tensor = self._read_video_with_decord(str_path)
            elif is_image:
                data_tensor = self._load_image_logic(str_path)
            else:
                logger.error(f"⚠️ Unknown file type: {str_path}")
        except Exception as e:
            logger.error(f"🚨 Data read path crashed: {str_path} | Reason: {e}")
            # Debugging Interception: Set a breakpoint to check for file corruption or decoder incompatibility.
            if os.environ.get("DEBUG", "0") == "1":
                breakpoint()

        if data_tensor is None:
            logger.critical(f"💀 Silent read failed (returns None)! Sample: {sample_id}")
            if os.environ.get("DEBUG", "0") == "1":
                breakpoint()
            return default_payload

        # 3. Assemble the payload
        label = torch.tensor(row.get("normalized_score", row["mos"]), dtype=torch.float32)
        payload = {"data": data_tensor, "label": label}

        if self.traditional_cols:
            payload["traditional"] = {col: torch.tensor(row[col], dtype=torch.float32) for col in self.traditional_cols}

        return payload

    def _read_video_with_decord(self, str_path: str) -> torch.Tensor:
        """
        High-performance video sampling using decord
        """
        if DECORD_AVAILABLE:
            try:
                # Use CPU context to prevent GPU memory handle deadlock in multi-process environments.
                vr = VideoReader(str_path, ctx=cpu(0))
                total_frames = len(vr)
                max_frames = self.num_frames

                # Uniform sampling index (more representative than range(max_frames))
                if total_frames >= max_frames:
                    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
                else:
                    # NOTE: For very short videos, read and populate them all directly.
                    indices = list(range(total_frames))

                # NOTE: Retrieve all frames at once and directly return an ndarray of (16, H, W, 3).
                frames = vr.get_batch(indices).asnumpy()

                # Directly using a loop to resize is extremely inefficient and error-prone; consider slicing directly.
                # Check if frames are empty.
                if frames.size == 0:
                    raise ValueError(f"Decord returned a sequence of empty frames: {str_path}")

                # Normalization: resize to (224, 224)
                # Decord is already reading RGB, so there's no need for cv2.cvtColor.
                resized_frames = [cv2.resize(f, (224, 224)) for f in frames]
                video_np = np.stack(resized_frames)
                data_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

                # If there are not enough frames, padding is applied to the timeline.
                if data_tensor.size(0) < max_frames:
                    padding = data_tensor[-1].unsqueeze(0).repeat(max_frames - data_tensor.size(0), 1, 1, 1)
                    data_tensor = torch.cat([data_tensor, padding], dim=0)

                return data_tensor

            except Exception as e:
                logger.error(f"🚨 [Decord] Critical video parsing failure: {str_path} | Error: {e}")

        # Fall back to OpenCV
        return self._read_video_with_opencv(str_path)

    def _read_video_with_opencv(self, str_path: str) -> torch.Tensor:
        """OpenCV video readback fallback solution"""
        cap = cv2.VideoCapture(str_path)
        if not cap.isOpened():
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        frames = []
        max_frames = self.num_frames

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            return torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)

        video_np = np.stack(frames)
        data_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

        if data_tensor.size(0) < max_frames:
            padding = data_tensor[-1].unsqueeze(0).repeat(max_frames - data_tensor.size(0), 1, 1, 1)
            data_tensor = torch.cat([data_tensor, padding], dim=0)

        return data_tensor

    def _load_image_logic(self, str_path: str) -> torch.Tensor:
        """Encapsulated image loading logic"""
        img = cv2.imread(str_path)
        if img is None:
            raise ValueError("Decoding failed")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            return self.transform(img)

        img = cv2.resize(img, (224, 224))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class TrainerExecutionPipeline:
    """
    Managing the training process of K-fold cross-validation
    """

    def __init__(self, config: Dict[str, Any], eda_df: pd.DataFrame, data_dir: Path, resolver: Any = None):
        self.config = config
        self.eda_df = eda_df
        self.data_dir = Path(data_dir).resolve()
        self.resolver = resolver

        self.task_type = config.get("task_type", "iqa")
        if "video" in str(data_dir).lower() or self.config.get("dataset_type") == "video":
            self.task_type = "vqa"
        self.is_video = config.get("task_type") == "vqa"

        self.train_cfg = config.get("train", {})
        self.batch_size = self.config.get("preprocessing", {}).get("batch_size", 16)
        self.num_workers = self.config.get("preprocessing", {}).get("num_workers", 4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # FIXME
    def execute_cross_validation(self, evaluator: Any):
        """
        Perform K-fold cross-validation, train independently for each fold, and support breakpoint cleanup and model selection.
        """
        n_splits = self.config.get("preprocessing", {}).get("k_fold", 5)
        logger.info(f"🧱 [Pipeline] Start with {n_splits} fold cross-validation...")

        clean_df = self.eda_df.copy()

        # Outlier filtering
        if "is_outlier" in clean_df.columns:
            clean_df = clean_df[~clean_df["is_outlier"]].reset_index(drop=True)
            logger.info(f"🧹 [Pipeline] Outliers have been filtered out. Remaining samples: {len(clean_df)}")

        # Test set isolation
        test_df = None
        if "split" in clean_df.columns:
            active_df = clean_df[clean_df["split"].isin(["train", "val"])].reset_index(drop=True)
            test_df = clean_df[clean_df["split"] == "test"]
            logger.info(f"🛡️ The test set has been isolated: (Size: {len(test_df)})")
        else:
            active_df = clean_df
            logger.warning("⚠️ The 'split' column was not detected, indicating a risk of data leakage.")

        original_base_filename = getattr(evaluator, "base_filename", "model")

        # Save the original configuration for test set evaluation.
        original_config = copy.deepcopy(self.config)
        for key in ["current_fold", "model"]:
            if key in original_config:
                original_config[key] = self.config.get(key)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(active_df)):
            current_fold = fold + 1
            logger.info(f"🌀 [Fold {current_fold}/{n_splits}] 开始...")

            self.config["current_fold"] = current_fold
            evaluator.base_filename = f"{original_base_filename}_fold{current_fold}"

            train_sub_df = active_df.iloc[train_idx]
            val_sub_df = active_df.iloc[val_idx]

            train_dataset = VQA_IQADataset(train_sub_df, self.data_dir, config=self.config, resolver=self.resolver)
            val_dataset = VQA_IQADataset(val_sub_df, self.data_dir, config=self.config, resolver=self.resolver)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

            model_config = self.config.get("model", {}).copy()
            model_config["num_frames"] = self.config.get("preprocessing", {}).get("num_frames", 8)
            self.config["model"] = model_config

            model = IQAVQANet(config=self.config)
            model = model.to(self.device)

            optimizer_cfg = self.config.get("train", {})
            lr = optimizer_cfg.get("lr", 1e-3)
            weight_decay = optimizer_cfg.get("weight_decay", 1e-4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            epochs = optimizer_cfg.get("epochs", 30)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            criterion = IQAVQALoss(config=self.config)

            engine = TrainerEngine(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=evaluator,
                config=self.config,
                scheduler=scheduler,
                device=self.device,
            )

            engine.fit(train_loader, val_loader)
            logger.info(f"🏁 [Fold {current_fold}] Completed")

            # Variables are automatically garbage collected when the function ends, and do not necessarily need to be manually deleted.
            for var in ["model", "optimizer", "scheduler", "criterion", "engine"]:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()
            gc.collect()

            if self.config.get("train", {}).get("fast_run", False):
                logger.warning("⚡ Run mode, exit after completing the first discount.")
                break

        # Test set evaluation
        if test_df is not None and not test_df.empty:
            logger.info("🧪 Start test set evaluation...")
            self._evaluate_test_set(test_df, evaluator, original_base_filename, n_splits, original_config)

    def _evaluate_test_set(self, test_df: pd.DataFrame, evaluator: Any, base_filename: str, n_splits: int, original_config: Dict = None):
        """Test set evaluation"""
        # Use the original configuration (without pollution such as current_fold).
        config_to_use = original_config if original_config else self.config

        # Ensure that the model configuration contains `num_frames`
        model_config = config_to_use.get("model", {}).copy()
        model_config["num_frames"] = config_to_use.get("preprocessing", {}).get("num_frames", 8)
        config_to_use["model"] = model_config

        model = None

        try:
            model = IQAVQANet(config=self.config).to(self.device)
            save_dir = Path(self.config.get("logging", {}).get("save_dir", "./results/model_outputs"))
            best_model_path = save_dir / f"{base_filename}_fold{n_splits}_best.pt"

            if best_model_path.exists():
                logger.info(f"💾 Load the best model: {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=self.device)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                logger.warning("⚠️ No best model path was found; random weights were used.")

            model.eval()

            test_dataset = VQA_IQADataset(test_df, self.data_dir, config=config_to_use, resolver=self.resolver)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

            y_true_list, y_pred_list = [], []

            with torch.no_grad():
                for batch in test_loader:
                    data = batch["data"].to(self.device)
                    target = batch["label"].to(self.device)
                    output = model(data)
                    y_true_list.append(target.cpu())
                    y_pred_list.append(output.cpu())

            y_true = torch.cat(y_true_list).numpy()
            y_pred = torch.cat(y_pred_list).numpy()

            evaluator.evaluate(y_true, y_pred)

        finally:
            # Clean up the model and video memory
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()
