# `config/` — 配置系统指南

本目录采用**分层 + 模块化**的设计，统一管理项目的所有配置参数。

---

## 📁 文件结构

```bash
config/
├── basic.yaml                 # 全局基础配置（系统、训练、日志等）
├── dataset_config.yaml        # 数据集配置（TID2013、KoNViD-1k、T2VQA 等）
├── paths.yaml                 # 路径模板定义（DSL）
├── models/                    # 模型专属配置
│   ├── resnet_iqa.yaml
│   ├── timeswin_vqa.yaml
│   └── ...
└── GUIDE.md
```

---

## 🎯 配置加载流程

```text
load_system_config(model_name, dataset_name)
    ↓
1. basic.yaml（基础）
2. models/xxx.yaml（模型覆盖）
3. dataset_config.yaml（数据集信息）
    ↓
deep_update 递归合并
    ↓
返回完整 config 字典
```

---

## 📋 核心配置文件说明

### 1. `basic.yaml` — 全局基础配置

包含项目通用的运行时参数：

- **system**：设备、AMP、num_workers 等
- **preprocessing**：随机种子、K-Fold、batch_size
- **train**：优化器、学习率调度、Early Stop、Checkpoint
- **evaluation**：测试策略、评估指标
- **logging**：日志间隔、TensorBoard 等

---

### 2. `dataset_config.yaml` — 数据集配置

每个数据集一个顶级 Key（支持大小写不敏感查找）。

#### 主要字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `name` | 数据集显示名称 | "KoNViD-1k" |
| `data_type` | `image` 或 `video` | "video" |
| `file_extensions` | 支持的文件后缀 | `["mp4"]` |
| `paths` | 目录结构定义 | `root`, `data`, `metadata` |
| `metadata` | 元数据文件信息 | `mos_file`, `format`, `delimiter` |
| `split` | 划分比例 | `train_ratio`, `val_ratio` |
| `preprocessing` | 预处理参数 | `num_frames`, `frame_size` 等 |

**已支持数据集**：
- **TID2013**（传统 IQA）
- **KoNViD-1k**（真实场景 VQA）
- **T2VQA-DB**（文本到视频质量评估）

---

### 3. `paths.yaml` — 路径管理系统

采用**模板 + 变量插值**的 DSL 设计。

#### 核心结构

- `roots`：基础路径前缀
- `resolvers`：路径模板定义

**使用示例**：
```python
from src.utils.path_manager import PathManager

dataset_dir = PathManager.resolve("dataset", dataset="konvid-1k")
log_dir = PathManager.resolve("train_logs", dataset="konvid-1k", mkdir=True)
```

**优势**：
- 集中管理所有路径
- 自动创建目录
- 消除硬编码和相对路径问题

---

### 4. `models/` 目录 — 模型专属配置

每个模型一个 YAML 文件，覆盖 `basic.yaml` 中的默认值。

**推荐命名规范**：`{backbone}_{task}.yaml`（如 `resnet_iqa.yaml`、`timeswin_vqa.yaml`）

#### 典型字段

- `task_type`：`iqa` 或 `vqa`
- `model`：backbone、预训练、dropout、输入尺寸等
- `preprocessing`：batch_size、num_workers
- `train`：epochs、lr、Early Stop、Checkpoint
- `loss`：不同任务的损失权重

---

## 🔧 最佳实践

1. **新增数据集**：
   - 在 `dataset_config.yaml` 中新增一个顶级 Key
   - 实现对应的 `MetadataLoader` 并注册到工厂

2. **新增模型**：
   - 在 `config/models/` 下新增 YAML 文件
   - 在 `PathManager` 或 `config_loader` 的 MODEL_MAP 中注册（自动发现优先）

3. **修改通用参数**：
   - 优先修改 `basic.yaml`
   - 模型特定参数在对应 model yaml 中覆盖

4. **路径使用**：
   - 统一使用 `PathManager.resolve()`，**禁止**手动拼接路径

5. **配置校验**：
   - 系统会在加载时自动进行 Schema 校验，缺少关键字段会直接报错

---

## 📌 使用示例

```python
from src.utils.config_loader import load_system_config
from src.utils.path_manager import PathManager

# 加载完整配置
config = load_system_config(
    model_cfg_name="timeswin_vqa",
    dataset_name="konvid-1k"
)

# 使用路径管理器
data_dir = PathManager.resolve("dataset", dataset=config["dataset_name"])
model_dir = PathManager.resolve("model_outputs", dataset=config["dataset_name"], mkdir=True)
```

---

## ⚠️ 注意事项

- **大小写**：数据集名称和 model key 加载时会转为小写处理
- **合并规则**：后加载的配置会覆盖前面的同名字段（model > basic）
- **路径模板**：修改 `paths.yaml` 后需同步更新 `PathManager` 中的兼容方法
- **数据集路径**：建议保持 `datasets/{dataset_name}/` 结构

---

**维护提示**：每次新增数据集或模型配置后，请同步更新本 GUIDE.md 中的“已支持数据集”列表。