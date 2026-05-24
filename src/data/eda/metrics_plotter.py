# src/data/eda/metrics_plotter.py
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Targets: (可视化)
#   1. 绘制loss曲线，残差图，每个epoch的 loss output，RMSE、R2 score
#   2. 绘制SSIM、VIF、DLM、VMAF、NIQE   <-- 🚀 传统指标对账热力分布图/箱线图
#
# Notes:
#   1. 存储的img注意文件名不要乱七八糟，要保留生成时间、model的信息 <-- 🚀 已锁定年月日_task_model规范
#   2. 以py文件所在的文件路径为基准：
#      img文件存储到../results/plots/中   <-- 🚀 完美对齐你的 results/plots 目录


class MetricsPlotter:
    """Draw training curves, residual plots, and index distributions, etc."""

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

    def __init__(self, task_type: str = "iqa", model_name: str = "resnet50", plots_dir: Path = None):
        """Initialization: Set output directory, filename prefix, and plotting backend."""
        self.task_type = task_type
        self.model_name = model_name
        self.current_date = time.strftime("%Y%m%d")

        if plots_dir:
            self.plots_dir = Path(plots_dir)
        else:
            self.plots_dir = self._PROJECT_ROOT / "results" / "plots"

        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Use the Agg backend to avoid errors in environments without a GUI.
        plt.switch_backend("Agg")
        sns.set_style("whitegrid")

        # Filename prefix: {date}_{task}_{model}
        self.base_prefix = f"{self.current_date}_{self.task_type}_{self.model_name}"

    def plot_training_history(self, csv_path: Path, version: str = "v1"):
        """Plot the training history curve (2x2 subplot)"""
        if not csv_path.exists():
            logger.warning(f"⚠️  History file not found: {csv_path}")
            raise FileNotFoundError(f"Missing history logs at {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"⚠️ CSV file is empty: {csv_path}")
                return
        except Exception as e:
            logger.error(f"❌ Failed to read CSV: {csv_path}, error: {e}")
            raise
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss Curve (Train vs Val)
        if "train_loss" in df.columns and "val_loss" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="train_loss",
                ax=axes[0, 0],
                label="Train Loss",
                marker="o",
                markersize=4,
                linewidth=2,
                color="tab:blue",
            )
            sns.lineplot(
                data=df,
                x="epoch",
                y="val_loss",
                ax=axes[0, 0],
                label="Val Loss",
                marker="s",
                markersize=4,
                linewidth=2,
                color="tab:orange",
            )
            axes[0, 0].set_title("Loss Convergence Curve", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()

        # 2. Plot PLCC and SROCC curves
        if "plcc" in df.columns and "srocc" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="plcc",
                ax=axes[0, 1],
                label="PLCC (Pearson)",
                marker="o",
                markersize=4,
                color="forestgreen",
                linewidth=2,
            )
            sns.lineplot(
                data=df,
                x="epoch",
                y="srocc",
                ax=axes[0, 1],
                label="SROCC (Spearman)",
                marker="s",
                markersize=4,
                color="royalblue",
                linewidth=2,
            )
            axes[0, 1].set_title("Alignment with Human MOS (PLCC/SROCC)", fontsize=12, fontweight="bold")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Correlation Coefficient")
            axes[0, 1].legend()

            # Set the lower limit of the vertical axis to min (minimum value of the indicator, -0.05) to avoid the curve from being suspended.
            min_corr = min(df["plcc"].min(), df["srocc"].min(), -0.05)
            axes[0, 1].set_ylim(min_corr - 0.05, 1.05)
        else:
            axes[0, 0].text(0.5, 0.5, "Loss Data Missing", ha="center", va="center", color="gray")
            axes[0, 0].set_title("Loss Convergence (Missing Data)")
            axes[0, 0].axis("off")  # Turn off axes and hide blank subplots when data is missing.

        # 3.
        # Left axis RMSE
        # right axis R²
        if "rmse" in df.columns and "r2" in df.columns:
            ax1 = axes[1, 0]
            sns.lineplot(
                data=df,
                x="epoch",
                y="rmse",
                ax=ax1,
                label="RMSE",
                marker="o",
                markersize=4,
                color="crimson",
                linewidth=2,
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("RMSE (Lower is better)", color="crimson")
            ax1.tick_params(axis="y", labelcolor="crimson")
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            sns.lineplot(
                data=df,
                x="epoch",
                y="r2",
                ax=ax2,
                label="R²",
                marker="s",
                markersize=4,
                color="darkorange",
                linewidth=2,
            )
            ax2.set_ylabel("R² Score (Goodness of Fit)", color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange")
            ax2.legend(loc="upper right")
            axes[1, 0].set_title("Error & Prediction Accuracy", fontsize=12, fontweight="bold")

        # 4. KROCC curve
        if "krocc" in df.columns:
            sns.lineplot(
                data=df,
                x="epoch",
                y="krocc",
                ax=axes[1, 1],
                label="KROCC (Kendall)",
                marker="^",
                markersize=4,
                color="purple",
                linewidth=2,
            )
            axes[1, 1].set_title("Kendall Tau Rank Correlation", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("KROCC")
            min_krocc = min(df["krocc"].min(), -0.05)
            axes[1, 1].set_ylim(min_krocc - 0.05, 1.05)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, "KROCC Metric Not Cached", ha="center", va="center", fontsize=12, color="gray")
            axes[1, 1].axis("off")

        plt.suptitle(f"Framework History | Model: {self.model_name} ({version})", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.base_prefix}_{version}_training_history.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Saved training history curve: {save_path}")

    def plot_residuals(self, csv_path: Path, version: str = "v1"):
        """Residual Analysis: Scatter Plot, Regression Comparison, Error Distribution Histogram"""
        if not csv_path.exists():
            logger.warning(f"⚠️  File not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"⚠️ CSV file is empty: {csv_path}")
                return
        except Exception as e:
            logger.error(f"❌ Failed to read CSV: {csv_path}, error: {e}")
            return

        if "pred" not in df.columns or "true" not in df.columns:
            logger.warning(f"⚠️  CSV missing 'pred' or 'true' columns: {csv_path}")
            return

        y_true = np.asarray(df["true"], dtype=np.float64)
        y_pred = np.asarray(df["pred"], dtype=np.float64)
        residuals = y_true - y_pred

        # Third Subplot: Residual Distribution Histogram + KDE
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Scatter test of residual variance
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color="dodgerblue", edgecolors="w", linewidths=0.3)
        axes[0].axhline(y=0, color="crimson", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Predicted Quality Score (MOS)")
        axes[0].set_ylabel("Residuals (True - Pred)")
        axes[0].set_title("Residuals Scatter (Homoscedasticity)", fontsize=11, fontweight="bold")

        # 2. Actual Value vs. Predicted Value: Linear Regression Check
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=20, color="darkgreen", edgecolors="w", linewidths=0.3)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "crimson", linestyle="--", linewidth=1.5, label="Perfect Alignment")
        axes[1].set_xlabel("True Human MOS")
        axes[1].set_ylabel("Predicted Network MOS")
        axes[1].set_title("Linearity Alignment (MOS Accuracy)", fontsize=11, fontweight="bold")
        axes[1].legend()

        # 3. One-dimensional marginal Gaussian probability density distribution of residuals
        sns.histplot(residuals, kde=True, ax=axes[2], color="purple", edgecolor="black", alpha=0.6, bins=20)
        axes[2].axvline(x=0, color="crimson", linestyle="--", linewidth=1.5)
        axes[2].set_xlabel("Residual Error Value")
        axes[2].set_ylabel("Density / Count")
        axes[2].set_title("Error Distribution (Normality Check)", fontsize=11, fontweight="bold")

        plt.suptitle(f"Residual & Linearity Diagnostics | Model: {self.model_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.base_prefix}_{version}_residuals.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Saved residual diagnostics plot: {save_path}")

    def plot_traditional_metrics_distribution(self, data_df: pd.DataFrame, dataset_name: str, score_column: str = "mos"):
        """Find the score column based on score_column, supporting case-insensitive matching."""
        target_metrics = ["ssim", "vif", "dlm", "vmaf", "niqe"]
        available_metrics = []
        for m in target_metrics:
            # Try case-insensitive matching
            matches = [col for col in data_df.columns if col.lower() == m.lower()]
            if matches:
                available_metrics.append(matches[0])  # Use actual column names

        if not available_metrics:
            logger.info("ℹ️  No traditional benchmarking metrics (SSIM/VIF/etc.) found in the current dataframe.")
            return

        # Reconciliation route check
        if score_column not in data_df.columns:
            # Try case-insensitive matching
            possible_cols = ["mos", "true", "score", "target", "label"]
            matched = None
            for col in possible_cols:
                matches = [c for c in data_df.columns if c.lower() == col.lower()]
                if matches:
                    matched = matches[0]
                    break

            if matched:
                resolved_score = matched
                logger.warning(f"⚠️  Specified '{score_column}' not found. Using '{resolved_score}' instead.")
            else:
                logger.error(f"❌ The rating column could not be found. Available columns: {data_df.columns.tolist()}")
                return
        else:
            resolved_score = score_column

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5), squeeze=False)

        for i, metric in enumerate(available_metrics):
            sns.regplot(
                data=data_df,
                x=metric,
                y=resolved_score,
                ax=axes[0, i],
                scatter_kws={"alpha": 0.4, "s": 15, "color": "purple"},
                line_kws={"color": "crimson", "linestyle": "-."},
            )
            axes[0, i].set_title(f"{metric.upper()} vs {resolved_score.upper()}", fontsize=11, fontweight="bold")
            axes[0, i].set_xlabel(metric.upper())
            axes[0, i].set_ylabel(resolved_score.upper())

        plt.suptitle(f"Traditional Academic Benchmarks Distribution on [{dataset_name}]", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"{self.current_date}_{dataset_name}_traditional_benchmarks.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Saved traditional benchmarks distribution report: {save_path}")

    def plot_comparison(self, metrics_csv_dict: dict, dataset_name: str):
        """
        Draw a bar chart comparing PLCC, SROCC, and RMSE for multiple models.
        """
        if not metrics_csv_dict:
            logger.warning("⚠️  No metrics files provided for cross-model benchmarking.")
            return

        models = []
        plcc_list, srocc_list, rmse_list = [], [], []

        for m_name, info in metrics_csv_dict.items():
            csv_path = info.get("csv_path")
            if not csv_path or not csv_path.exists():
                logger.debug(f"CSV not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    logger.warning(f"Empty CSV: {csv_path}")
                    continue
            except Exception as e:
                logger.error(f"Failed to read CSV {csv_path}: {e}")
                continue

            last_row = df.iloc[-1]
            models.append(f"{m_name}\n({info.get('version', 'v1')})")
            plcc_list.append(last_row.get("plcc", 0))
            srocc_list.append(last_row.get("srocc", 0))
            rmse_list.append(last_row.get("rmse", 0))

        if not models:
            logger.warning("⚠️  No valid metrics rows successfully loaded.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_payload = [
            {"data": plcc_list, "title": "PLCC Comparison (Higher is better)", "color": "seagreen", "ax": axes[0]},
            {"data": srocc_list, "title": "SROCC Comparison (Higher is better)", "color": "royalblue", "ax": axes[1]},
            {"data": rmse_list, "title": "RMSE Comparison (Lower is better)", "color": "crimson", "ax": axes[2]},
        ]

        for payload in metrics_payload:
            ax = payload["ax"]
            # Perform drawing
            bars = ax.bar(models, payload["data"], color=payload["color"], alpha=0.75, edgecolor="black", linewidth=0.5)
            ax.set_title(payload["title"], fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

            # Vertical axis range:
            # - RMSE: 0 to maximum value × 1.2
            # - PLCC/SROCC: 0 to 1.05
            if "RMSE" in payload["title"]:
                ax.set_ylim(0, max(payload["data"]) * 1.2)
            else:
                ax.set_ylim(0, 1.05)

            # Draw numerical labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (height * 0.01),
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="semibold",
                )

        plt.suptitle(f"Cross-Model Benchmark Arena on Dataset: {dataset_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / f"comparison_arena_{dataset_name}_{self.current_date}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Saved tournament comparison leaderboard: {save_path}")
