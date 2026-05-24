from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# NOTE: The torch dependency in Docker might be an older version, preventing the use of new APIs.
# from torch.amp import autocast, GradScaler


class TrainerEngine:
    """
    General Training/Validation Engine

    ## Responsibilities:

    - Training loop scheduling (forward/backward propagation, gradient update)

    - Mixed Precision Training (AMP) and gradient pruning

    - Validation set evaluation and metric collection

    - Checkpoint saving and early stopping control

    ## Not concerned with:

    - Specific dataset format

    - Model architecture details

    - Loss function implementation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        evaluator: Any,
        config: Dict[str, Any],
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self._validate_config(config)
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluator = evaluator
        self.config = config
        self.scheduler = scheduler

        train_cfg = config.get("train", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.grad_clip = train_cfg.get("grad_clip", None)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)

        self.use_amp = self.config.get("system", {}).get("amp", True)
        # self.scaler = GradScaler(device_type=self.device_type, enabled=self.use_amp)
        self.scaler = GradScaler() if self.use_amp else None

        # Early Stopping State Machine Initialization
        early_stop_cfg = train_cfg.get("early_stop", {})
        self.early_stop_enabled = early_stop_cfg.get("enabled", True)
        self.patience = early_stop_cfg.get("patience", 10)
        self.early_stop_monitor = early_stop_cfg.get("monitor", "val_loss").lower()  # Unify the lowercase at the source
        self.early_stop_mode = early_stop_cfg.get("mode", "min")

        self.early_stop_counter = 0
        self.best_early_stop_score = float("inf") if self.early_stop_mode == "min" else float("-inf")

        # Checkpoint State Machine Monitoring Initialization
        checkpoint_cfg = train_cfg.get("checkpoint", {})
        self.checkpoint_monitor = checkpoint_cfg.get("monitor", "val_srocc").lower()  # Unify the lowercase at the source
        self.checkpoint_mode = checkpoint_cfg.get("mode", "max")
        self.best_checkpoint_score = float("inf") if self.checkpoint_mode == "min" else float("-inf")

        logger.info(
            f"🚀 [Engine] Training engine initialization successful."
            f"Device: {self.device} | Monitoring metrics: {self.checkpoint_monitor} | "
            f"Adaptive AMP: {self.use_amp}"
        )

    def _validate_config(self, config: Dict[str, Any]):
        """Check if the configuration file contains required fields; if any are missing, an error will be reported."""
        required_keys = {"train": ["epochs"], "logging": ["save_dir"]}

        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"🚨 The configuration file is missing a first-level node: [{section}]")

            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"🚨 The configuration file is missing necessary parameters: [{section}.{key}], Please check the YAML file.")

        logger.info("✅ [System] Configuration item verification passed, everything is ready.")

    def resume_training(self, checkpoint_path: str) -> int:
        """Resume training from the checkpoint; if it fails, start from scratch."""
        from pathlib import Path

        ckpt_file = Path(checkpoint_path)

        if not ckpt_file.exists():
            logger.warning(f"⚠️ The breakpoint does not exist in {checkpoint_path}; reinitialize the training.")
            return 1

        logger.info(f"🔄 [Engine] Loading checkpoint: {ckpt_file} ...")
        checkpoint = torch.load(ckpt_file, map_location=self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        logger.info("   ├─ [Model] The neural network tensor weights have been loaded.")

        if "optimizer" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("   ├─ [Optimizer] The optimizer status (momentum, gradient squared, etc.) has been restored.")

        if "scheduler" in checkpoint and self.scheduler and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("   ├─ [Scheduler] The Scheduler status has been restored.")

        resume_epoch = checkpoint.get("epoch", 0) + 1

        if "metrics" in checkpoint:
            hist_metrics = {k.lower(): v for k, v in checkpoint["metrics"].items()}
            self.best_checkpoint_score = hist_metrics.get(self.checkpoint_monitor, self.best_checkpoint_score)
            if self.early_stop_enabled:
                self.best_early_stop_score = hist_metrics.get(self.early_stop_monitor, self.best_early_stop_score)

        logger.info(f"   └─ [Time] Continue training from the {resume_epoch}th round.")
        return resume_epoch

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Training loop that runs a single epoch"""
        self.model.train()
        running_loss = 0.0

        # Get gradient accumulation steps
        accum_steps = self.gradient_accumulation_steps

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for batch_idx, batch_data in enumerate(pbar):
            if isinstance(batch_data, dict):
                inputs = batch_data["data"].to(self.device)
                labels = batch_data["label"].to(self.device)
            else:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

            # 1. Forward propagation
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)

            # 2. Loss Calculation
            with autocast(enabled=False):
                outputs_f32 = outputs.float()
                if outputs_f32.ndim > 1 and outputs_f32.size(-1) == 1:
                    outputs_f32 = outputs_f32.squeeze(-1)
                labels = labels.view_as(outputs_f32).float()
                loss_dict = self.criterion(outputs_f32, labels, model=self.model)
                total_loss = loss_dict["total_loss"]

                # NOTE: Divide by the cumulative number of steps to make the accumulated loss consistent with the normal loss magnitude.
                total_loss = total_loss / accum_steps

            # 3. Backpropagation (not updated immediately)
            self.scaler.scale(total_loss).backward()

            # 4. Updated once every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += total_loss.item() * accum_steps  # Recover the original loss

            # Log output
            log_interval = self.config.get("logging", {}).get("log_interval", 10)
            if batch_idx % log_interval == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{total_loss.item() * accum_steps:.4f}",
                        "MSE": f"{float(loss_dict.get('mse_loss', 0.0)):.4f}",
                        "Rank": f"{float(loss_dict.get('rank_loss', 0.0)):.4f}",
                    }
                )

        epoch_loss = running_loss / len(train_loader)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Evaluate the model on the validation set and return the various metrics."""
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_trues = []
        traditional_metrics_payload = {}

        for batch_data in tqdm(val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]"):
            if isinstance(batch_data, dict):
                inputs = batch_data["data"].to(self.device)
                labels = batch_data["label"].to(self.device)
                if "traditional" in batch_data:
                    for k, v in batch_data["traditional"].items():
                        traditional_metrics_payload.setdefault(k, []).extend(v.numpy())
            else:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)

            with autocast(enabled=False):
                outputs_f32 = outputs.float()

                if outputs_f32.ndim > 1 and outputs_f32.size(-1) == 1:
                    outputs_f32 = outputs_f32.squeeze(-1)
                labels = labels.view_as(outputs_f32).float()

                loss_dict = self.criterion(outputs_f32, labels, model=self.model)
                total_loss = loss_dict["total_loss"]

            running_loss += total_loss.item()
            all_preds.extend(outputs_f32.cpu().numpy().flatten())
            all_trues.extend(labels.cpu().numpy().flatten())

        val_loss = running_loss / len(val_loader)
        traditional_metrics = {k: np.array(v) for k, v in traditional_metrics_payload.items()} if traditional_metrics_payload else None

        eval_fn = getattr(self.evaluator, "evaluate", None)
        if eval_fn is None:
            for alt_name in ["compute", "compute_metrics", "run", "calculate"]:
                alt_fn = getattr(self.evaluator, alt_name, None)
                if alt_fn is not None:
                    eval_fn = alt_fn
                    break

        if eval_fn is None:
            raise AttributeError("🚨 The Evaluator class lacks a standard evaluation function path!")

        metrics = eval_fn(
            y_true=np.array(all_trues),
            y_pred=np.array(all_preds),
            epoch=epoch,
            val_loss=val_loss,
            traditional_metrics=traditional_metrics,
        )

        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 1):
        """Main training loop"""
        logger.info(f"⏱️  [System] Training begins, starting epoch: {start_epoch}")
        checkpoint_cfg = self.config.get("train", {}).get("checkpoint", {})

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            raw_metrics = self.evaluate(val_loader, epoch)

            # Convert indicator names to lowercase
            metrics = {k.lower(): v for k, v in raw_metrics.items()}

            val_loss = raw_metrics.get("val_loss", metrics.get("val_loss", 0.0))
            if val_loss == 0.0 and "loss" in metrics:
                val_loss = metrics["loss"]

            metrics["val_loss"] = val_loss
            metrics["loss"] = val_loss

            for indicator in ["srocc", "plcc", "rmse"]:
                if indicator in metrics:
                    metrics[f"val_{indicator}"] = metrics[indicator]

            val_srocc = metrics.get("val_srocc", 0.0)
            val_plcc = metrics.get("val_plcc", 0.0)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # 动态传入监控指标
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"📊 [Epoch {epoch}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}"
                )
            else:
                logger.info(f"📊 [Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}")

            # Early cessation method
            if self.early_stop_enabled:
                if self.early_stop_monitor in metrics:
                    score = metrics[self.early_stop_monitor]
                elif self.early_stop_monitor == "val_loss":
                    score = val_loss
                else:
                    logger.warning(f"The monitoring metric '{self.early_stop_monitor}' does not exist; use val_loss instead.")
                    score = val_loss

                if self._is_improved(score, self.best_early_stop_score, self.early_stop_mode):
                    self.best_early_stop_score = score
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.patience:
                    logger.warning(
                        f"🛑 The [Early Stop] monitoring metric '{self.early_stop_monitor}' has shown no improvement "
                        f"for several consecutive {self.early_stop_counter} rounds, triggering an overfitting prevention interception!"
                    )
                    break

            # Save model checkpoints
            ckpt_score = metrics.get(self.checkpoint_monitor, 0.0)

            if self._is_improved(ckpt_score, self.best_checkpoint_score, self.checkpoint_mode):
                self.best_checkpoint_score = ckpt_score
                if checkpoint_cfg.get("save_best", True):
                    self._save_checkpoint(epoch, val_loss, metrics, is_best=True)

            if checkpoint_cfg.get("save_last", True) and epoch == self.epochs:
                self._save_checkpoint(epoch, val_loss, metrics, is_best=False)

            # Clean up video memory fragmentation
            torch.cuda.empty_cache()

    def _is_improved(self, current: float, best: float, mode: str) -> bool:
        if mode == "min":
            return current < best
        elif mode == "max":
            return current > best
        return False

    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, Any], is_best: bool):
        """Save model checkpoints to the configured directory."""
        import re
        import shutil
        from pathlib import Path

        save_dir = Path(self.config.get("logging", {}).get("save_dir", "./results/model_outputs"))
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cfg = self.config.get("train", {}).get("checkpoint", {})
        top_k = checkpoint_cfg.get("save_top_k", 3)

        # NOTE：Make sure that the base_filename obtained here has already been aligned.
        base_name = self.evaluator.base_filename

        def extract_score(path: Path):

            # NOTE：Use `re.escape` to prevent special characters, and use regular expressions to match index values ​​in filenames.
            match = re.search(rf"{re.escape(self.checkpoint_monitor)}(-?\d+\.\d+)", path.name.lower())
            return float(match.group(1)) if match else (-float("inf") if self.checkpoint_mode == "max" else float("inf"))

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config,
        }

        if is_best:
            current_score = metrics.get(self.checkpoint_monitor, 0.0)
            # NOTE：Directly construct an absolute path
            pt_name = f"{base_name}_fold{self.config.get('current_fold', '1')}_best_epoch{epoch}_{self.checkpoint_monitor}{current_score:.4f}.pt"
            target_path = save_dir / pt_name

            torch.save(state, target_path)
            logger.info(f"🏆 [Checkpoint] The weights have been saved to ──> {target_path.resolve()}")

            all_best_pts = list(save_dir.glob(f"{base_name}_fold{self.config.get('current_fold', '1')}_best_epoch*.pt"))
            if len(all_best_pts) > top_k:
                reverse_flag = True if self.checkpoint_mode == "max" else False
                all_best_pts.sort(key=extract_score, reverse=reverse_flag)
                for low_pt in all_best_pts[top_k:]:
                    low_pt.unlink()
                    logger.warning(f"🗑️  [Checkpoint] Delete old model files and keep only the K most recent ones ──> {low_pt.name}")

            # NOTE: Refresh Best Soft Links
            all_best_pts = list(save_dir.glob(f"{base_name}_fold{self.config.get('current_fold', '1')}_best_epoch*.pt"))
            if all_best_pts:
                all_best_pts.sort(key=extract_score, reverse=(self.checkpoint_mode == "max"))
                best_of_best = all_best_pts[0]
                standard_best_path = save_dir / f"{base_name}_fold{self.config.get('current_fold', '1')}_best.pt"
                shutil.copy(best_of_best, standard_best_path)

        else:
            pt_name = f"{base_name}_best_epoch{epoch}_{self.checkpoint_monitor}{current_score:.4f}.pt"
            target_path = save_dir / pt_name
            torch.save(state, target_path)
            logger.info(f"💾 [Checkpoint] The model snapshot has been saved to ──> {target_path.resolve()}")
