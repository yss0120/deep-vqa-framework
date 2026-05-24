# src/data/eda/split.py
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split

# TODO:
# 按照dataset_config.yaml的数据集划分kv，划分数据集
# 对trainset、valset、testset分层K折交叉验证: 5折 (Notes: 检查每折的分数分布是否一致)


# NOTE：这部分实参到时候利用src.data.eda里面的method实现，而不是使用src.utils.config里面的函数
def split_train_val_test(df, train_ratio: float = 0.8, val_ratio: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    划分训练/验证/测试集（分层采样）

    Args:
        df: 包含 'mos' 列的 DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_state: 随机种子

    Returns:
        (train_df, val_df, test_df)
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio <= 0:
        raise ValueError(f"无效比例: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # 创建分层标签
    stratify_labels = create_stratified_labels(df)

    # 第一次划分：分出测试集
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=random_state, stratify=stratify_labels)

    # 第二次划分：从 train_val 中分出验证集
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_ratio,
        random_state=random_state,
        stratify=stratify_labels.iloc[train_val.index],
    )

    return train, val, test


def create_stratified_labels(df, bins: int = 10):
    """
    创建分层标签
    优先使用分位数，失败时回退到均匀分箱
    """
    try:
        return pd.qcut(df["mos"], q=bins, labels=False, duplicates="drop")
    except Exception:
        logger.debug("分位数切分失败，回退到均匀分箱")
        return pd.cut(df["mos"], bins=bins, labels=False)


def check_fold_distribution(df, n_splits: int = 5, random_state: int = 42, verbose: bool = True) -> List[Dict]:
    """
    检查K折分布

    Args:
        df: 包含 'mos' 列的 DataFrame
        n_splits: K折数量
        random_state: 随机种子
        verbose: 是否输出日志

    Returns:
        每折的统计信息列表
    """
    if df is None or df.empty:
        return []

    stratify_labels = create_stratified_labels(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        train_m = df.iloc[train_idx]["mos"].mean()
        val_m = df.iloc[val_idx]["mos"].mean()
        train_s = df.iloc[train_idx]["mos"].std()
        val_s = df.iloc[val_idx]["mos"].std()

        fold_stats.append({"fold": fold, "train_mean": train_m, "val_mean": val_m, "train_std": train_s, "val_std": val_s})

    if verbose:
        logger.info(f"Cross-Validation -> Stratified {n_splits}-Fold distribution checked")

    return fold_stats
