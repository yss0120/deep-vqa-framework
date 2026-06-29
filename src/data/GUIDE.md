# `src/data/` — 数据模块指南

该模块负责**数据集的加载、清洗、探索分析、预处理和标准化**，是整个训练流程的**数据入口**。

---

## 📁 目录结构

```bash
src/data/
├── data_eda.py                 # 主入口：DataEDA（核心类）
├── metadata_loader_factory.py    # 元数据加载器工厂
├── metadata_loaders.py         # 具体数据集加载器（Konvid、T2VQA、TID2013 等）
├── types.py                    # 类型定义（DatasetType）
└── eda/                        # 数据探索与分析子模块
      ├── integrity.py            # 文件完整性检查
      ├── statistics.py           # 统计分析
      ├── split.py                # 数据集划分
      └── metrics_plotter.py      # 可视化工具

```

---

## 🎯 核心职责

1. **元数据加载** — 支持多种格式（CSV、空格分隔、竖线分隔等）
2. **数据清洗与完整性检查** — 缺失文件、损坏文件、黑帧/白帧、跳帧
3. **统计分析与可视化** — MOS 分布、图像/视频属性
4. **数据集划分** — 分层采样（Train/Val/Test）
5. **分数预处理** — 归一化、离群值检测
6. **路径解析** — 大小写不敏感 + 后缀智能补全

---

## 🔄 数据流（推荐使用流程）

```text
main.py
    ↓
eda = DataEDA(dataset_name, data_dir, dataset_info)
    ↓
eda.load_metadata()
    ↓
MetadataLoaderFactory.get_loader(dataset_name) → 选择对应 Loader
    ↓
Loader.load(meta_file) → 返回标准化 DataFrame
    ↓
eda.run_full_eda(skip_integrity=False)
    ↓
    ├── basic_statistics()
    ├── check_integrity()          # 可选跳过视频深度检查
    ├── ensure_split_column()
    ├── check_filename_label_match()
    ├── visualize_mos_distribution()
    ├── normalize_scores()
    ├── detect_outliers_3sigma()
    └── check_fold_score_distribution()
    ↓
返回统计信息 + 处理后的 DataFrame
    ↓
Trainer / Dataset 使用 eda.df + resolver 进行训练
```

---

## 📋 核心类说明

### 1. `DataEDA`（`data_eda.py`）

**核心类**，负责整个 EDA 流程。

#### 主要方法

| 方法 | 功能 | 推荐参数 |
|------|------|----------|
| `load_metadata()` | 加载并解析元数据 | - |
| `run_full_eda(skip_integrity=False)` | 一键执行完整 EDA 流程 | `skip_integrity=True` 可显著加速 |
| `check_integrity(skip_video_check=True)` | 文件完整性检查 | 建议默认跳过视频深度检查 |
| `ensure_split_column()` | 添加 `split` 列（分层划分） | - |
| `normalize_scores()` | MOS 归一化到 [0, 1] | 使用 train 集的 min/max |
| `detect_outliers_3sigma()` | 3σ 离群值标记 | - |
| `basic_statistics()` | 打印基本统计信息 | - |
| `visualize_mos_distribution()` | 保存 MOS 分布图 | - |

---

### 2. `MetadataLoaderFactory`（`metadata_loader_factory.py`）

**工厂模式**，统一管理各数据集加载器。

**新增数据集方法**：
```python
# 在 _REGISTRY 中添加一行即可
_REGISTRY = {
    "konvid-1k": KonvidLoader,
    "t2vqa-db": T2VqaLoader,
    "tid2013": Tid2013Loader,
    "your_new_dataset": YourNewLoader,   # ← 新增
}
```

---

### 3. 数据加载器（`metadata_loaders.py`）

所有加载器继承 `BaseMetadataLoader`，必须返回包含以下列的 DataFrame：

- `sample_id`（文件名或 stem）
- `mos`（质量分数）

**已支持数据集**：
- **TID2013**（图像）
- **KoNViD-1k**（视频）
- **T2VQA**（视频）

**通用解析能力**：
- 支持逗号、竖线、空格、制表符等多种分隔符
- 自动清理引号、空白
- 失败时回退到逐行解析

---

### 4. `DatasetType`（`types.py`）

```python
class DatasetType(Enum):
    IMAGE = "image"
    VIDEO = "video"
```

用于统一区分图像/视频处理逻辑。

---

## 🔧 重要工具

- **`CaseInsensitiveAssetResolver`**：大小写不敏感的文件路径解析器，支持自动补全扩展名。
- **Quarantine 机制**：损坏文件自动移动到 `quarantine/` 目录。
- **结果输出**：
  - `results/plots/` — 可视化图表
  - `results/{dataset}_integrity_report.txt` — 完整性审计报告

---

## 📌 使用示例

```python
from src.data.data_eda import DataEDA
from src.utils.config_loader import load_config

config = load_config("configs/datasets/konvid.yaml")

eda = DataEDA(
    dataset_name="konvid-1k",
    data_dir=Path("/data/KoNViD-1k"),
    dataset_info=config["dataset"]
)

# 一键运行完整流程
stats = eda.run_full_eda(skip_integrity=False)

print(f"保留样本数: {len(eda.df)}")
print(f"MOS 范围: {stats['mos_range']}")
```

---

## ⚠️ 注意事项

## ⚠️ 注意事项

- **性能**：视频深度完整性检查（Decord 解码）较慢，默认开启 `skip_video_check=True`，只检查文件存在性
- **`skip_integrity` vs `skip_video_check`**：
  - `skip_integrity=True`：跳过所有文件检查（适用于快速测试）
  - `skip_video_check=True`：检查文件存在性，但不做视频内容解码（推荐用于大型数据集）
- **路径匹配**：`check_filename_label_match()` 对后缀敏感，建议在配置中明确 `file_extensions`
- **目录创建**：`results/` 目录会自动创建，但建议在项目初始化时统一确保
- **扩展性**：新增数据集只需实现 `BaseMetadataLoader` 并注册到工厂即可
- **归一化**：`normalize_scores()` 默认使用训练集的 min/max，避免数据泄露

---

## TODO

1. 增加更多数据集的 Loader（如 LIVE-VQC、YouTube-UGC 等）
2. 考虑将 `check_temporal_consistency` 集成到 `integrity.py` 中

---

**文档维护提示**：每次新增数据集或重要功能后，请同步更新本 GUIDE.md。
