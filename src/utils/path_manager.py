import pdb
from pathlib import Path

import yaml
from loguru import logger


class PathManager:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _CONFIG_FILE = _PROJECT_ROOT / "config/paths.yaml"

    # 预加载 YAML 配置（只读一次，全局共享，效率最高）
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as _f:
            _REGISTRY = yaml.safe_load(_f)

        if _REGISTRY is None:
            raise ValueError("paths.yaml 文件为空")
        if "resolvers" not in _REGISTRY:
            raise KeyError("paths.yaml 缺少 'resolvers' 字段")

    except FileNotFoundError:
        raise RuntimeError(
            f"[PathManager] 路径配置文件不存在: {_CONFIG_FILE}\n   请确保项目根目录下有 config/paths.yaml 文件\n   当前项目根目录: {_PROJECT_ROOT}"
        )
    except yaml.YAMLError as e:
        raise RuntimeError(f"[PathManager] paths.yaml 格式错误: {_CONFIG_FILE}\n   错误详情: {e}\n   请检查 YAML 语法，特别是缩进和特殊字符")
    except Exception as e:
        raise RuntimeError(f"[PathManager] 加载路径配置失败: {e}")

    @classmethod
    def resolve(cls, key: str, mkdir: bool = False, **kwargs) -> Path:
        """
        核心 DSL 寻址引擎：替代一切诡异的路径拼接。
        示例：PathManager.resolve("train_logs", dataset="TID2013", mkdir=True)
        """
        template = cls._REGISTRY["resolvers"].get(key)
        if not template:
            available = list(cls._REGISTRY["resolvers"].keys())
            logger.error(f"❌ Key '{key}' 没在 paths.yaml 的 resolvers 中定义！")
            logger.debug(f"[PathManager] 可用的 keys: {available}")
            # Help: 检查 paths.yaml 中的 resolvers 字段是否包含此 key
            raise KeyError(f"Key '{key}' not found. Available: {available}")

        # 💎【核心修复】：这里是关键！
        # 我们创建一个临时字典，把 roots 里的每一项“摊平”解包出来。
        # 这样你在模板里写 {datasets} 就能直接拿到值，不再需要 {roots.datasets} 了。
        roots_ctx = cls._REGISTRY.get("roots", {})
        fmt_args = {**roots_ctx, **kwargs}

        try:
            path_str = template.format(**fmt_args)
        except KeyError as e:
            logger.error(f"❌ 路径模板 '{template}' 中缺少必要的变量: {e}")
            logger.debug(f"[PathManager] 当前参数: {fmt_args.keys()}")
            # Help: 检查模板中需要的变量是否都已传入
            # Help: 对比 kwargs 和模板中的 {变量名}
            raise KeyError(f"❌ 路径模板 '{template}' 中缺少必要的变量: {e}")

        # 3. 强行锚定在项目根目录下，并转换为绝对路径，彻底消除“相对路径漂移”
        final_path = (cls._PROJECT_ROOT / path_str).resolve()
        logger.debug(f"🔍 [PathManager] Resolved '{key}' to: {final_path}")

        # 4. 自动化福利：如果是目录且要求 mkdir，顺手帮用户建好
        if mkdir:
            try:
                final_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.error(f"❌ 权限不足，无法创建目录: {final_path}")
                # Help: 检查目录权限
                pdb.set_trace()
                raise
            except OSError as e:
                logger.error(f"❌ 创建目录失败: {final_path}, 错误: {e}")
                pdb.set_trace()
                raise

        return final_path

    # =========================================================================
    # 兼容过渡层：以下保留旧接口，但底层全部重构为新引擎，确保旧代码不报错
    # =========================================================================

    @classmethod
    def get_config_file(cls, filename: str) -> Path:
        """【兼容旧代码】获取配置文件"""
        # 如果用户传的是 models/timeswin_vqa，自动去匹配 model_config 路由
        if "dataset_config" in filename:
            return cls._PROJECT_ROOT / "config" / "dataset_config.yaml"

        if "models/" in filename:
            name = Path(filename).stem
            return cls.resolve("model_config", name=name)

        # 否则默认走 basic_config 路由
        return cls.resolve("basic_config")

    @classmethod
    def get_dataset_dir(cls, dataset_name: str) -> Path:
        """【兼容旧代码】获取指定数据集的根目录"""
        return cls.resolve("dataset", dataset=dataset_name)

    @classmethod
    def get_results_dir(cls, dataset_name: str, sub_dir: str = "") -> Path:
        """【兼容旧代码】获取指定数据集的结果归档目录"""
        # 旧代码传的 sub_dir（如 "train_logs", "plots"）刚好对应我们的新 key
        if sub_dir in cls._REGISTRY["resolvers"]:
            return cls.resolve(sub_dir, dataset=dataset_name, mkdir=True)

        # 兜底逻辑：如果是一些奇奇怪怪的临时子目录
        base_results = cls.resolve("dataset", dataset=dataset_name).parent.parent / "results"
        path = base_results / dataset_name / sub_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_project_root(cls) -> Path:
        """获取工程绝对根路径"""
        return cls._PROJECT_ROOT
