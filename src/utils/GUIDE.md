# `src/utils/` — 工具模块指南

该模块提供项目中**全局共享的基础设施**：配置加载、路径管理、日志系统、文件解析等，是整个框架的**底层支撑**。

---

## 📁 目录结构

```bash
src/utils/
├── config_loader.py          # 配置分层加载 + Schema 校验
├── path_manager.py           # 路径解析引擎（推荐使用）
├── file_loader.py            # 文件路径解析器（CaseInsensitiveAssetResolver）
└── logging_utils.py          # 日志系统初始化与装饰器
```

---

## 🎯 核心组件

### 1. `PathManager`（`path_manager.py`）

**最推荐**的路径管理方式，使用 DSL 风格解析路径。

#### 核心方法

| 方法 | 功能 | 示例 |
|------|------|------|
| `PathManager.resolve(key, **kwargs)` | 核心路径解析引擎 | `PathManager.resolve("dataset", dataset="konvid-1k")` |
| `PathManager.get_config_file()` | 获取配置文件路径 | - |
| `PathManager.get_dataset_dir(dataset_name)` | 获取数据集目录 | - |
| `PathManager.get_results_dir()` | 获取结果目录 | - |

**优势**：
- 集中管理所有路径模板（位于 `config/paths.yaml`）
- 自动创建目录（`mkdir=True`）
- 强锚定项目根目录，杜绝相对路径问题
- 支持动态变量插值

---

### 2. `config_loader.py` — 配置加载引擎

**功能**：分层加载 `basic.yaml` + `model/*.yaml` + `dataset_config.yaml`

#### 核心函数

| 函数 | 功能 |
|------|------|
| `load_system_config(model_cfg_name, dataset_name)` | 一键加载完整配置（推荐入口） |
| `safe_load_yaml(path)` | 安全的 YAML 加载器 |
| `deep_update()` | 递归字典合并 |
| `warn_orphan_keys(config)` | 检测配置中的孤儿 Key（未在 Schema 中定义的字段） |
| `validate_config_schema(config)` | Schema 校验（已定义，可根据需要启用） |

**关键特性**：
- 严格的 `REQUIRED_CONFIG_PATHS` 校验（防止隐蔽 Bug）
- 孤儿 Key 警告（`warn_orphan_keys`）
- 支持自动发现 `config/models/` 下的模型配置文件

---

### 3. `file_loader.py` — 文件路径解析器

**核心类**：`CaseInsensitiveAssetResolver`

#### 特性

- **大小写不敏感** 查找
- **预构建内存索引**（`rglob` 一次扫描，之后 O(1) 查询）
- 支持图像和视频自动类型识别
- 返回 `AssetInfo` 数据类（包含 `path`、`is_video` 等信息）

**推荐使用方式**：
```python
resolver = CaseInsensitiveAssetResolver(data_dir, allowed_extensions=[".mp4", ".jpg"])

asset = resolver.resolve("0001.mp4")        # 返回 AssetInfo
path = resolver.resolve_path("0001")        # 仅路径
```

---

### 4. `logging_utils.py` — 日志系统

**功能**：统一初始化项目日志。

#### 主要功能

- `log_prepare(model_name, dataset_name)`：初始化控制台 + 文件 + CSV 日志
- 自动生成带时间戳的日志文件名，返回 `base_fn` 用于后续文件命名（如 checkpoint 前缀）
- CSV 日志安全转义（处理换行、引号）
- `time_it` 装饰器：函数耗时统计
- `atexit` 钩子：程序退出时记录总结日志

---

## 🔄 使用流程示例

```python
from src.utils.config_loader import load_system_config
from src.utils.path_manager import PathManager
from src.utils.logging_utils import log_prepare
from src.utils.file_loader import CaseInsensitiveAssetResolver

# 1. 日志初始化
base_fn = log_prepare(model_name="IQAVQANet", dataset_name="konvid-1k")

# 2. 加载配置
config = load_system_config(model_cfg_name="timeswin_vqa", dataset_name="konvid-1k")

# 3. 路径管理
dataset_dir = PathManager.resolve("dataset", dataset="konvid-1k")
plots_dir = PathManager.resolve("plots", dataset="konvid-1k", mkdir=True)

# 4. 文件解析器
resolver = CaseInsensitiveAssetResolver(
    target_dir=dataset_dir,
    allowed_extensions=config["dataset_info"]["file_extensions"]
)
```

---

## 📋 配置体系说明

项目采用**三层配置叠加**策略：

1. `config/basic.yaml` — 通用基础配置
2. `config/models/xxx.yaml` — 模型特定配置（backbone、超参等）
3. `config/dataset_config.yaml` — 数据集信息

最终通过 `deep_update` 合并，并进行严格 Schema 校验。

---

## ⚙️ 最佳实践

- **路径**：优先使用 `PathManager.resolve()`，避免手动拼接
- **配置**：始终使用 `load_system_config()` 加载，不要直接 `safe_load_yaml`
- **文件查找**：使用 `CaseInsensitiveAssetResolver`，不要手动 `rglob`
- **日志**：在 `main.py` 最开始调用 `log_prepare()`
- **调试**：开启 `logger.debug` 可查看详细路径解析和配置合并过程

---

## 🛠️ TODO

- 在 `paths.yaml` 中持续补充新的路径模板
- 未来可考虑加入 `ConfigManager` 类，利用 `Pydantic` 来统一管理配置对象
- 可增加配置热重载支持（开发模式）
